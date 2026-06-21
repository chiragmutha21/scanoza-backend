"""
MongoDB connection using Motor (async driver).
Includes a local JSON fallback if MongoDB is unreachable.
"""
import os
import json
import re
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/ar_db")
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/ar_db")
LOCAL_DB_PATH = "data/metadata_backup.json"

client: AsyncIOMotorClient = None
db = None
_is_mock = False

async def connect_db():
    global client, db, _is_mock
    try:
        # Try simple connection first (which works on this environment)
        print("Connecting to MongoDB using default settings...")
        client = AsyncIOMotorClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        await client.admin.command('ping')
        db = client.get_default_database()
        if db is None: db = client["shop"]
        _is_mock = False
        print(f"[OK] MongoDB connected: {db.name}")
        try:
            await db["contents"].create_index(
                "originalImageHash", unique=True, sparse=True,
                name="unique_original_target_image"
            )
        except Exception as index_error:
            print(f"[WARN] Could not create duplicate-image index: {index_error}")
        return
    except Exception as e:
        print(f"[WARN] Default MongoDB connection failed: {e}")
        
    try:
        # Fallback with certifi
        import certifi
        ca_file = certifi.where()
        print(f"Retrying with CA bundle: {ca_file}")
        client = AsyncIOMotorClient(
            MONGO_URI,
            tls=True,
            tlsCAFile=ca_file,
            tlsAllowInvalidCertificates=False,
            serverSelectionTimeoutMS=5000,
        )
        await client.admin.command('ping')
        db = client.get_default_database()
        if db is None: db = client["shop"]
        _is_mock = False
        print(f"[OK] MongoDB connected (certifi): {db.name}")
        try:
            await db["contents"].create_index(
                "originalImageHash", unique=True, sparse=True,
                name="unique_original_target_image"
            )
        except Exception as index_error:
            print(f"[WARN] Could not create duplicate-image index: {index_error}")
    except Exception as e:
        print(f"[WARN] MongoDB certifi connection failed: {e}")
        # Retry once with tlsAllowInvalidCertificates=True as fallback
        try:
            print("Retrying with relaxed TLS...")
            client = AsyncIOMotorClient(
                MONGO_URI,
                tls=True,
                tlsAllowInvalidCertificates=True,
                serverSelectionTimeoutMS=5000,
            )
            await client.admin.command('ping')
            db = client.get_default_database()
            if db is None: db = client["shop"]
            _is_mock = False
            print(f"[OK] MongoDB connected (relaxed TLS): {db.name}")
        except Exception as e2:
            print(f"[FAIL] MongoDB fallback mode active: {e2}")
            db = None
            _is_mock = True
            if not os.path.exists(LOCAL_DB_PATH):
                os.makedirs("data", exist_ok=True)
                with open(LOCAL_DB_PATH, "w") as f:
                    json.dump({"contents": [], "attached_contents": []}, f)

async def disconnect_db():
    global client
    if client: client.close()

class MockCollection:
    def __init__(self, key): self.key = key
    def _matches(self, item, query):
        for key, expected in query.items():
            if key == "$or":
                return any(self._matches(item, subquery) for subquery in expected)
            actual = item.get(key)
            if isinstance(expected, dict):
                if "$regex" in expected:
                    flags = re.IGNORECASE if "i" in expected.get("$options", "") else 0
                    if actual is None or not re.match(expected["$regex"], str(actual), flags):
                        return False
                elif "$exists" in expected:
                    if (key in item) != bool(expected["$exists"]):
                        return False
                else:
                    return False
            elif actual != expected:
                return False
        return True
    def _load(self):
        try:
            with open(LOCAL_DB_PATH, "r") as f: return json.load(f).get(self.key, [])
        except: return []
    def _save(self, items):
        data = {"contents": [], "attached_contents": []}
        try:
            with open(LOCAL_DB_PATH, "r") as f: data = json.load(f)
        except: pass
        data[self.key] = items
        with open(LOCAL_DB_PATH, "w") as f: json.dump(data, f, indent=2)

    async def insert_one(self, doc):
        print(f"MOCK DB: Inserting into {self.key}")
        items = self._load()
        if "_id" not in doc:
            doc["_id"] = f"mock_{self.key}_{len(items)}"
        items.append(doc)
        self._save(items)
        return doc
    def find(self, query=None):
        items = self._load()
        if query: items = [i for i in items if self._matches(i, query)]
        class AsyncIter:
            def __init__(self, d): self.d = d; self.i = 0
            def sort(self, k, dir): self.d.sort(key=lambda x: x.get(k, ""), reverse=(dir == -1)); return self
            def __aiter__(self): return self
            async def __anext__(self):
                if self.i >= len(self.d): raise StopAsyncIteration
                val = self.d[self.i]; self.i += 1; return val
        return AsyncIter(items)
    async def find_one(self, query):
        items = self._load()
        for i in items:
            if self._matches(i, query): return i
        return None
    async def delete_one(self, query):
        items = self._load(); new = [i for i in items if not self._matches(i, query)]
        self._save(new); return len(items) != len(new)
    async def delete_many(self, query):
        items = self._load(); new = [i for i in items if not self._matches(i, query)]
        self._save(new); return len(items) - len(new)
    async def count_documents(self, query):
        items = self._load(); return len([i for i in items if self._matches(i, query)])
    async def update_one(self, query, update):
        items = self._load()
        for item in items:
            if self._matches(item, query):
                item.update(update.get("$set", {}))
                self._save(items)
                return True
        return False

def get_images_collection():
    if _is_mock:
        print("Using MOCK images collection")
        return MockCollection("contents")
    print("Using REAL images collection")
    return db["contents"] if db is not None else None

def get_attached_contents_collection():
    if _is_mock:
        print("Using MOCK attached_contents collection")
        return MockCollection("attached_contents")
    print("Using REAL attached_contents collection")
    return db["attached_contents"] if db is not None else None
