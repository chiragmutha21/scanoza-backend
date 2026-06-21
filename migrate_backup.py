import asyncio
import os
import json
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

LOCAL_DB_PATH = "data/metadata_backup.json"

async def migrate():
    if not os.path.exists(LOCAL_DB_PATH):
        print("No local backup file found.")
        return

    with open(LOCAL_DB_PATH, "r") as f:
        local_data = json.load(f)

    uri = os.getenv("MONGO_URI")
    client = AsyncIOMotorClient(uri, tlsAllowInvalidCertificates=True)
    
    try:
        db = client.get_default_database()
        print(f"Migrating to cloud DB: {db.name}")

        # Migrate contents
        contents = local_data.get("contents", [])
        if contents:
            coll = db["contents"]
            for doc in contents:
                # Remove mock _id if present
                if "_id" in doc and str(doc["_id"]).startswith("mock_"):
                    del doc["_id"]
                
                # Check if already exists
                exists = await coll.find_one({"contentId": doc["contentId"]})
                if not exists:
                    await coll.insert_one(doc)
                    print(f" - Migrated content: {doc['contentId']}")
                else:
                    print(f" - Content already exists: {doc['contentId']}")

        # Migrate attachments
        attachments = local_data.get("attached_contents", [])
        if attachments:
            coll = db["attached_contents"]
            for doc in attachments:
                if "_id" in doc and str(doc["_id"]).startswith("mock_"):
                    del doc["_id"]
                
                exists = await coll.find_one({"attachmentId": doc["attachmentId"]})
                if not exists:
                    await coll.insert_one(doc)
                    print(f" - Migrated attachment: {doc['attachmentId']}")
                else:
                    print(f" - Attachment already exists: {doc['attachmentId']}")

        print("Migration complete.")
    except Exception as e:
        print(f"Error during migration: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(migrate())
