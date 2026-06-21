import json
import os
from database import get_images_collection, connect_db, disconnect_db
import asyncio

async def audit():
    # 1. Check FAISS mapping
    mapping_path = "data/id_mapping.json"
    faiss_ids = set()
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            data = json.load(f)
            faiss_ids = set(data.get("id_to_idx", {}).keys())
    
    print(f"IDs in FAISS Index: {len(faiss_ids)}")
    for fid in faiss_ids:
        print(f"  - {fid}")

    # 2. Check MongoDB
    await connect_db()
    coll = get_images_collection()
    mongo_ids = set()
    if coll is not None:
        async for doc in coll.find({}, {"contentId": 1}):
            mongo_ids.add(doc["contentId"])
    
    print(f"\nIDs in MongoDB: {len(mongo_ids)}")
    for mid in mongo_ids:
        print(f"  - {mid}")

    # 3. Find Desync
    missing = faiss_ids - mongo_ids
    if missing:
        print(f"\nCRITICAL: {len(missing)} IDs found in FAISS but MISSING in MongoDB (will cause 'Metadata Lost'):")
        for m in missing:
            print(f"  !! {m}")
    else:
        print("\nSync Check: FAISS and MongoDB IDs match.")

    await disconnect_db()

if __name__ == "__main__":
    asyncio.run(audit())
