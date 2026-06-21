
import os
import pymongo
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

def test_connection(uri, name, **kwargs):
    print(f"Testing {name}...")
    try:
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000, **kwargs)
        client.admin.command('ping')
        print(f"[OK] {name} connected!")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return False

if __name__ == "__main__":
    print(f"URI: {MONGO_URI}")
    
    # Try 1: Standard
    test_connection(MONGO_URI, "Standard")
    
    # Try 2: No TLS (won't work for Atlas but worth a try to see the error)
    test_connection(MONGO_URI, "No TLS", tls=False)
    
    # Try 3: Relaxed TLS
    test_connection(MONGO_URI, "Relaxed TLS", tls=True, tlsAllowInvalidCertificates=True)
    
    # Try 4: With Certifi
    try:
        import certifi
        test_connection(MONGO_URI, "With Certifi", tls=True, tlsCAFile=certifi.where())
    except:
        pass
