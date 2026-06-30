"""
FastAPI application entry point for the AR Image Recognition system.
"""
import os
import traceback
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from database import connect_db, disconnect_db, get_images_collection
from routes import router

load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

async def _sync_faiss_from_db():
    """
    On startup, ensure FAISS index matches MongoDB exactly.
    - Removes stale entries (in FAISS but not in DB)
    - Adds missing entries (in DB but not in FAISS)
    - Does a full rebuild if stale entries are detected (since FAISS doesn't support removal easily)
    """
    from routes import faiss_index as fi_module
    from embeddings import extract_embedding
    import faiss_index as fi_pkg

    collection = get_images_collection()
    if collection is None:
        print("[WARN] FAISS sync skipped: DB not connected")
        return

    # Gather all DB content IDs
    db_ids = set()
    db_docs = []
    cursor = collection.find()
    async for doc in cursor:
        cid = doc.get("contentId")
        if cid:
            db_ids.add(cid)
            db_docs.append(doc)

    faiss_ids = set(fi_module.id_to_idx.keys())
    stale_ids = faiss_ids - db_ids
    missing_ids = db_ids - faiss_ids

    print(f"FAISS: {len(faiss_ids)} entries | DB: {len(db_ids)} entries")
    print(f"  Stale (in FAISS, not DB): {len(stale_ids)} | Missing (in DB, not FAISS): {len(missing_ids)}")

    needs_rebuild = len(stale_ids) > 0 or len(missing_ids) > 0

    if not needs_rebuild:
        print(f"[OK] FAISS index is in sync ({fi_module.total} entries)")
        return

    # Full rebuild needed
    print("[SYNC] Rebuilding FAISS index from DB...")
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index.bin")
    FAISS_MAPPING_PATH = os.getenv("FAISS_MAPPING_PATH", "data/id_mapping.json")

    # Create fresh index
    new_index = fi_pkg.FaissIndex.__new__(fi_pkg.FaissIndex)
    new_index.index_path = FAISS_INDEX_PATH
    new_index.mapping_path = FAISS_MAPPING_PATH
    new_index.id_to_idx = {}
    new_index.idx_to_id = {}
    new_index._next_idx = 0
    new_index.is_numpy = fi_module.is_numpy

    from embeddings import EMBEDDING_DIM
    if fi_module.is_numpy:
        new_index.index = fi_pkg.NumpyFlatIndex(EMBEDDING_DIM)
    else:
        import faiss
        new_index.index = faiss.IndexFlatIP(EMBEDDING_DIM)

    indexed = 0
    for doc in db_docs:
        content_id = doc.get("contentId")
        image_path = doc.get("imagePath", "")
        abs_path = None

        if image_path.startswith("http"):
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.get(image_path, timeout=15)
                    if resp.status_code == 200:
                        ext = os.path.splitext(image_path.split("?")[0])[1] or ".jpg"
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=UPLOAD_DIR)
                        tmp.write(resp.content)
                        tmp.close()
                        abs_path = tmp.name
                    else:
                        print(f"  [WARN] Download failed for {content_id}: HTTP {resp.status_code}")
            except Exception as e:
                print(f"  [WARN] Download error for {content_id}: {e}")
        else:
            abs_path = os.path.join(os.getcwd(), image_path.lstrip("/"))
            if not os.path.exists(abs_path):
                print(f"  [WARN] Local file missing for {content_id}: {abs_path}")
                abs_path = None

        if abs_path:
            try:
                embedding = extract_embedding(abs_path)
                new_index.add(embedding, content_id)
                indexed += 1
                print(f"  [OK] Indexed: {content_id} ({doc.get('originalImageName', '?')})")
            except Exception as e:
                print(f"  [FAIL] Embedding failed for {content_id}: {e}")
            finally:
                if image_path.startswith("http") and os.path.exists(abs_path):
                    os.remove(abs_path)

    # Replace the global faiss_index in routes module
    import routes
    routes.faiss_index = new_index
    print(f"[SYNC] Rebuild complete: {indexed}/{len(db_docs)} images indexed")


async def _sync_order_hashes():
    """
    On startup, check if there are any orders with framePhoto but missing framePhotoHash or framePhotoDHash,
    download them, compute hashes, and save them.
    """
    from database import get_orders_collection
    from routes import calculate_file_hash, calculate_image_dhash
    import httpx
    
    col = get_orders_collection()
    if col is None:
        return
        
    cursor = col.find({
        "framePhoto": {"$exists": True, "$ne": ""},
        "$or": [
            {"framePhotoHash": {"$exists": False}},
            {"framePhotoHash": ""},
            {"framePhotoDHash": {"$exists": False}},
            {"framePhotoDHash": ""}
        ]
    })
    
    orders_to_sync = []
    async for doc in cursor:
        orders_to_sync.append(doc)
        
    if not orders_to_sync:
        return
        
    print(f"[SYNC] Found {len(orders_to_sync)} orders missing framePhoto hashes. Syncing...")
    
    async with httpx.AsyncClient() as client:
        for doc in orders_to_sync:
            order_id = doc.get("orderId")
            photo_url = doc.get("framePhoto")
            
            # Download photo to a temp file
            try:
                if photo_url.startswith("http"):
                    resp = await client.get(photo_url, timeout=20)
                    if resp.status_code == 200:
                        ext = os.path.splitext(photo_url.split("?")[0])[1] or ".jpg"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            tmp.write(resp.content)
                            tmp_path = tmp.name
                    else:
                        print(f"  [WARN] Download failed for order {order_id} photo: {resp.status_code}")
                        continue
                else:
                    tmp_path = os.path.join(os.getcwd(), photo_url.lstrip("/"))
                    if not os.path.exists(tmp_path):
                        continue
                        
                file_hash = calculate_file_hash(tmp_path)
                dhash = calculate_image_dhash(tmp_path)
                
                await col.update_one(
                    {"orderId": order_id},
                    {"$set": {"framePhotoHash": file_hash, "framePhotoDHash": dhash}}
                )
                print(f"  [OK] Hash synced for order {order_id}")
                
                if photo_url.startswith("http") and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception as e:
                print(f"  [FAIL] Hash sync error for order {order_id}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application starting up...")
    await connect_db()
    for sub in ["images", "videos", "audio", "pdfs"]:
        os.makedirs(os.path.join(UPLOAD_DIR, sub), exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Auto-sync FAISS index with MongoDB
    try:
        await _sync_faiss_from_db()
    except Exception as e:
        print(f"[WARN] FAISS sync failed (non-fatal): {e}")
        traceback.print_exc()

    # Sync order hashes
    try:
        await _sync_order_hashes()
    except Exception as e:
        print(f"[WARN] Order hashes sync failed (non-fatal): {e}")

    print("Server ready.")
    yield
    await disconnect_db()

app = FastAPI(
    title="AR Image Recognition System",
    version="1.1.0",
    lifespan=lifespan,
)

# Dynamic CORS — allow local IP and localhost
def get_local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

local_ip = get_local_ip()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
        "http://localhost:3003",
        "http://127.0.0.1:3003",
        f"http://{local_ip}:3000",
        f"http://{local_ip}:3001",
        f"http://{local_ip}:3002",
        f"http://{local_ip}:3003",
        f"http://{local_ip}.nip.io:3000",
        f"http://{local_ip}.nip.io:3001",
        f"http://{local_ip}.nip.io:3002",
        f"http://{local_ip}.nip.io:3003",
        f"http://{local_ip}.nip.io:5000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"GLOBAL ERROR: {str(exc)}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "detail": str(exc)},
        headers={
            "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
            "Access-Control-Allow-Credentials": "true",
        }
    )

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Incoming: {request.method} {request.url.path}")
    response = await call_next(request)
    return response

app.include_router(router)

@app.get("/")
async def root():
    return {"name": "AR Image Recognition System", "version": "1.1.0"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
