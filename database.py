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

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/shop")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DB_PATH = os.path.normpath(os.path.join(BASE_DIR, "data", "metadata_backup.json"))

client: AsyncIOMotorClient = None
db = None
_is_mock = False

async def _create_indexes(db_instance):
    if db_instance is not None:
        try:
            await db_instance["contents"].create_index(
                "originalImageHash", unique=True, sparse=True,
                name="unique_original_target_image"
            )
        except Exception as index_error:
            print(f"[WARN] Could not create duplicate-image index: {index_error}")

async def connect_db():
    global client, db, _is_mock
    import ssl as _ssl

    # Attempt 1: Standard TLS with certifi CA bundle
    try:
        import certifi
        ca_file = certifi.where()
        print(f"Using CA bundle: {ca_file}")
        client = AsyncIOMotorClient(
            MONGO_URI,
            tls=True,
            tlsCAFile=ca_file,
            tlsAllowInvalidCertificates=False,
            serverSelectionTimeoutMS=2000,
            connectTimeoutMS=2000,
            socketTimeoutMS=2000,
        )
        await client.admin.command('ping')
        db = client.get_default_database()
        if db is None: db = client["shop"]
        _is_mock = False
        print(f"[OK] MongoDB connected: {db.name}")
        await _create_indexes(db)
        return
    except Exception as e:
        print(f"[WARN] MongoDB connection attempt 1 failed: {e}")
        if client:
            client.close()

    # Attempt 2: Custom SSL context (fixes TLSV1_ALERT_INTERNAL_ERROR)
    try:
        print("Retrying with custom SSL context...")
        import certifi
        ctx = _ssl.create_default_context(cafile=certifi.where())
        ctx.check_hostname = False
        ctx.verify_mode = _ssl.CERT_NONE
        client = AsyncIOMotorClient(
            MONGO_URI,
            tls=True,
            tlsAllowInvalidCertificates=True,
            tlsAllowInvalidHostnames=True,
            serverSelectionTimeoutMS=2000,
            connectTimeoutMS=2000,
            socketTimeoutMS=2000,
        )
        await client.admin.command('ping')
        db = client.get_default_database()
        if db is None: db = client["shop"]
        _is_mock = False
        print(f"[OK] MongoDB connected (custom SSL): {db.name}")
        await _create_indexes(db)
        return
    except Exception as e2:
        print(f"[WARN] MongoDB connection attempt 2 failed: {e2}")
        if client:
            client.close()

    # Attempt 3: Direct connection string with authSource
    try:
        print("Retrying with direct connection string...")
        # Some Atlas clusters need authSource=admin explicitly
        uri = MONGO_URI
        if "authSource" not in uri:
            sep = "&" if "?" in uri else "?"
            uri = uri + sep + "authSource=admin"
        client = AsyncIOMotorClient(
            uri,
            tls=True,
            tlsAllowInvalidCertificates=True,
            tlsAllowInvalidHostnames=True,
            directConnection=False,
            serverSelectionTimeoutMS=2000,
            connectTimeoutMS=2000,
            socketTimeoutMS=2000,
        )
        await client.admin.command('ping')
        db = client.get_default_database()
        if db is None: db = client["shop"]
        _is_mock = False
        print(f"[OK] MongoDB connected (attempt 3): {db.name}")
        await _create_indexes(db)
        return
    except Exception as e3:
        if client:
            client.close()
        print(f"[FAIL] MongoDB fallback mode active: {e3}")
        db = None
        _is_mock = True
        if not os.path.exists(LOCAL_DB_PATH):
            os.makedirs(os.path.dirname(LOCAL_DB_PATH), exist_ok=True)
            with open(LOCAL_DB_PATH, "w") as f:
                json.dump({"contents": [], "attached_contents": [], "products": [], "orders": [], "website_users": []}, f)

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
            if isinstance(expected, dict) and "$regex" in expected:
                flags = re.IGNORECASE if "i" in expected.get("$options", "") else 0
                if actual is None or not re.match(expected["$regex"], str(actual), flags):
                    return False
            elif actual != expected:
                return False
        return True
    def _load(self):
        try:
            with open(LOCAL_DB_PATH, "r") as f: return json.load(f).get(self.key, [])
        except: return []
    def _save(self, items):
        data = {"contents": [], "attached_contents": [], "products": [], "orders": [], "website_users": []}
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

def get_products_collection():
    if _is_mock:
        print("Using MOCK products collection")
        return MockCollection("products")
    print("Using REAL products collection")
    return db["products"] if db is not None else None

def get_orders_collection():
    if _is_mock:
        print("Using MOCK orders collection")
        return MockCollection("orders")
    print("Using REAL orders collection")
    return db["orders"] if db is not None else None

def get_website_users_collection():
    if _is_mock:
        print("Using MOCK website_users collection")
        return MockCollection("website_users")
    print("Using REAL website_users collection")
    return db["website_users"] if db is not None else None
