import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

async def list_ids():
    uri = os.getenv("MONGO_URI")
    client = AsyncIOMotorClient(uri, tlsAllowInvalidCertificates=True)
    try:
        db = client.get_default_database()
        coll = db["contents"]
        docs = await coll.find({}, {"contentId": 1, "_id": 0}).to_list(length=100)
        print("ContentIDs in Cloud MongoDB:")
        for d in docs:
            print(f" - {d.get('contentId')}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(list_ids())
