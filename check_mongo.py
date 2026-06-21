import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

async def check():
    uri = os.getenv("MONGO_URI")
    print(f"Connecting to: {uri}")
    client = AsyncIOMotorClient(uri, tlsAllowInvalidCertificates=True)
    try:
        await client.admin.command('ping')
        db = client.get_default_database()
        print(f"Connected to DB: {db.name}")
        coll = db["contents"]
        count = await coll.count_documents({})
        print(f"Documents in 'contents' collection: {count}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(check())
