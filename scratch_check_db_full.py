import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

async def debug_db():
    uri = os.getenv("MONGO_URI")
    print(f"Connecting to URI: {uri}")
    client = AsyncIOMotorClient(uri, tlsAllowInvalidCertificates=True)
    try:
        # Check default database name
        db_default = client.get_default_database()
        print(f"Default DB: {db_default.name if db_default is not None else 'None'}")
        
        # List all databases
        dbs = await client.list_database_names()
        print(f"Databases on cluster: {dbs}")
        
        for db_name in dbs:
            if db_name in ["admin", "local", "config"]:
                continue
            db = client[db_name]
            cols = await db.list_collection_names()
            print(f"Database: {db_name} | Collections: {cols}")
            for col_name in cols:
                coll = db[col_name]
                count = await coll.count_documents({})
                print(f"  - Collection: {col_name} | Count: {count}")
                docs = await coll.find({}).to_list(length=5)
                for doc in docs:
                    print(f"    * ID: {doc.get('contentId')} | Name: {doc.get('originalImageName')} | Email: {doc.get('userEmail')}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(debug_db())
