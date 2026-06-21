import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

async def test():
    print(f"Testing connection to: {MONGO_URI}")
    try:
        # Try simplest connection first
        client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        await client.admin.command('ping')
        print("Success with default settings!")
        return
    except Exception as e:
        print(f"Failed with default: {e}")

    try:
        # Try with certifi
        import certifi
        client = AsyncIOMotorClient(MONGO_URI, tls=True, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
        await client.admin.command('ping')
        print("Success with certifi!")
        return
    except Exception as e:
        print(f"Failed with certifi: {e}")

if __name__ == "__main__":
    asyncio.run(test())
