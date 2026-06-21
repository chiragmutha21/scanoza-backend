
import asyncio
from database import connect_db, get_images_collection
from dotenv import load_dotenv
load_dotenv()

async def check():
    await connect_db()
    col = get_images_collection()
    async for doc in col.find():
        print(f"ID: {doc.get('contentId')} | Path: '{doc.get('imagePath')}'")

if __name__ == "__main__":
    asyncio.run(check())
