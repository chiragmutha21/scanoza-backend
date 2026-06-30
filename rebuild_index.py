
import asyncio
import os
from database import connect_db, get_images_collection
from embeddings import extract_embedding, extract_augmented_embeddings
import faiss_index
import httpx
import tempfile
from dotenv import load_dotenv

load_dotenv()

async def rebuild():
    print("Connecting to DB...")
    await connect_db()
    
    collection = get_images_collection()
    if collection is None:
        print("Failed to connect to DB.")
        return

    # Initialize index
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index.bin")
    FAISS_MAPPING_PATH = os.getenv("FAISS_MAPPING_PATH", "data/id_mapping.json")
    
    # Backup old files (Handling Windows FileExistsError)
    if os.path.exists(FAISS_INDEX_PATH):
        if os.path.exists(FAISS_INDEX_PATH + ".bak"):
            os.remove(FAISS_INDEX_PATH + ".bak")
        os.rename(FAISS_INDEX_PATH, FAISS_INDEX_PATH + ".bak")
    if os.path.exists(FAISS_MAPPING_PATH):
        if os.path.exists(FAISS_MAPPING_PATH + ".bak"):
            os.remove(FAISS_MAPPING_PATH + ".bak")
        os.rename(FAISS_MAPPING_PATH, FAISS_MAPPING_PATH + ".bak")

    idx = faiss_index.FaissIndex(FAISS_INDEX_PATH, FAISS_MAPPING_PATH)
    
    print("Fetching documents from DB...")
    cursor = collection.find()
    count = 0
    async for doc in cursor:
        content_id = doc.get("contentId")
        image_path = doc.get("imagePath")
        if not content_id or not image_path:
            continue
        
        print(f"Indexing {content_id} ({image_path})...")
        try:
            # Handle Cloudinary URLs
            if image_path.startswith("http"):
                async with httpx.AsyncClient() as client:
                    resp = await client.get(image_path)
                    if resp.status_code == 200:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                            tmp.write(resp.content)
                            tmp_path = tmp.name
                        
                        embeddings = extract_augmented_embeddings(tmp_path)
                        os.remove(tmp_path)
                    else:
                        print(f"Failed to download {image_path}")
                        continue
            else:
                abs_path = os.path.join(os.getcwd(), image_path.lstrip("/"))
                if not os.path.exists(abs_path):
                    print(f"File missing: {abs_path}")
                    continue
                embeddings = extract_augmented_embeddings(abs_path)

            for emb in embeddings:
                idx.add(emb, content_id)
            count += 1
        except Exception as e:
            print(f"Failed to index {content_id}: {e}")

    if count == 0:
        idx.save()

    print(f"Done! Indexed {count} images.")

if __name__ == "__main__":
    asyncio.run(rebuild())
