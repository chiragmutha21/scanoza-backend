import asyncio
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from database import connect_db, get_images_collection

async def main():
    await connect_db()
    col = get_images_collection()
    if col is None:
        print("Could not get collection.")
        return
    
    count = await col.count_documents({})
    print(f"Total Images in DB: {count}")
    
    cursor = col.find({})
    async for doc in cursor:
        print(f"ContentID: {doc.get('contentId')} | Name: {doc.get('originalImageName')}")

if __name__ == "__main__":
    asyncio.run(main())
