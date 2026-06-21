"""
FastAPI application entry point for the AR Image Recognition system.
"""
import os
import traceback
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

    needs_rebuild = len(stale_ids) > 0 or len(missing_ids) > 0

    if not needs_rebuild:
        print("[OK] FAISS index is in sync")
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
        if image_path:
            try:
                embedding = extract_embedding(image_path)
                new_index.add(embedding, content_id)
                indexed += 1
            except Exception as e:
                print(f"  [FAIL] Embedding failed for {content_id}: {e}")

    # Replace the global faiss_index in routes module
    import routes
    routes.faiss_index = new_index
    print(f"[SYNC] Rebuild complete: {indexed}/{len(db_docs)} images indexed")
    print("[OK] FAISS index is in sync")


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
        "http://localhost",
        "https://localhost",
        "capacitor://localhost",
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
    return {"name": "AR Image Recognition System", "version": "1.1.0", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
