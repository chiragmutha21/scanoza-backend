import asyncio
import os
import json
from database import connect_db, get_images_collection
from embeddings import extract_augmented_embeddings
import faiss_index
from dotenv import load_dotenv

load_dotenv()

async def sync():
    print("Connecting to DB...")
    await connect_db()
    
    collection = get_images_collection()
    if collection is None:
        print("Failed to connect to DB.")
        return

    # Load metadata backup
    backup_path = "data/metadata_backup.json"
    if not os.path.exists(backup_path):
        print("Backup file not found.")
        return
        
    with open(backup_path, "r") as f:
        backup_data = json.load(f)
        
    contents = backup_data.get("contents", [])
    print(f"Found {len(contents)} items in backup.")

    # Initialize index
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index.bin")
    FAISS_MAPPING_PATH = os.getenv("FAISS_MAPPING_PATH", "data/id_mapping.json")
    idx = faiss_index.FaissIndex(FAISS_INDEX_PATH, FAISS_MAPPING_PATH)
    
    count = 0
    for doc in contents:
        content_id = doc.get("contentId")
        image_path = doc.get("imagePath")
        if not content_id or not image_path:
            continue
            
        # Check if already exists in DB
        existing = await collection.find_one({"contentId": content_id})
        if not existing:
            # Prepare document for DB insertion (matching the current schema)
            # Remove mock _id to let MongoDB generate its own
            db_doc = dict(doc)
            if "_id" in db_doc:
                del db_doc["_id"]
            
            # Normalize email
            db_doc["userEmail"] = (db_doc.get("userEmail") or "").strip().lower()
            
            print(f"Inserting {content_id} into MongoDB...")
            await collection.insert_one(db_doc)
        else:
            print(f"Document {content_id} already exists in DB.")

        # Index in FAISS if not already indexed
        if content_id not in idx.id_to_idx:
            print(f"Extracting embeddings for {content_id} ({image_path})...")
            try:
                # Ensure the path is absolute/resolvable
                local_path = image_path
                if not os.path.isabs(local_path):
                    local_path = os.path.join(os.getcwd(), image_path.lstrip("/\\"))
                
                if not os.path.exists(local_path):
                    print(f"Warning: local file {local_path} not found. Skipping FAISS indexing.")
                    continue
                    
                embeddings = extract_augmented_embeddings(local_path)
                for emb in embeddings:
                    idx.add(emb, content_id)
                print(f"Successfully indexed {content_id} in FAISS.")
                count += 1
            except Exception as e:
                print(f"Failed to index {content_id}: {e}")
        else:
            print(f"FAISS index already has vectors for {content_id}.")

    print(f"Sync complete. Added {count} new images to FAISS index.")

if __name__ == "__main__":
    asyncio.run(sync())
