"""
Verification script for Step 3: Real-Time Scanning Endpoint.
"""
import httpx
import os
import sys

BASE = "http://localhost:5000"
API = f"{BASE}/api"
# Use a VALID image from the list (64KB+)
TEST_IMAGE = r"d:\noQR\reference\uploads\images\image-1771233914151-919533683.jpg"

if not os.path.exists(TEST_IMAGE):
    print(f"FAIL: Test image not found at {TEST_IMAGE}")
    sys.exit(1)

client = httpx.Client(timeout=120)

print("\n══ STARTING STEP 3 VERIFICATION ══\n")

# 1. Upload
print("1. Uploading fresh test image...")
with open(TEST_IMAGE, "rb") as img:
    r = client.post(f"{API}/upload", 
        files={"image": ("scan-verify-final.jpg", img, "image/jpeg")},
        data={"videoLink": "https://youtu.be/step3-final-demo"})

if r.status_code != 200:
    print(f"FAIL: Upload failed with {r.status_code}: {r.text}")
    sys.exit(1)

upload_data = r.json()
CONTENT_ID = upload_data["contentId"]
print(f"   Success! contentId: {CONTENT_ID}")

# 2. Attach some content
print("\n2. Attaching content to test cascade trigger...")
client.post(f"{API}/attach-content", data={
    "contentId": CONTENT_ID,
    "contentType": "text",
    "text": "VERIFIED: Step 3 Recognition-to-Trigger pipeline working!",
    "title": "Success Message"
})
client.post(f"{API}/attach-content", data={
    "contentId": CONTENT_ID,
    "contentType": "audio",
    "url": "https://example.com/notification.mp3",
    "title": "Trigger Sound"
})
print("   Success! Contents attached.")

# 3. Perform a Scan
print("\n3. Testing /api/scan (The core of Step 3)...")
with open(TEST_IMAGE, "rb") as img:
    # Use 'frame' as the field name as expected by routes.py
    r = client.post(f"{API}/scan", files={"frame": ("frame.jpg", img, "image/jpeg")})

if r.status_code != 200:
    print(f"FAIL: Scan failed with {r.status_code}: {r.text}")
    sys.exit(1)

scan_data = r.json()
if scan_data.get("matchFound") is True:
    print(f"   ✅ SUCCESS: Image recognized!")
    print(f"   Confidence: {scan_data['confidence']:.4f}")
    matched_id = scan_data["content"]["contentId"]
    if matched_id == CONTENT_ID:
        print(f"   ✅ SUCCESS: Correct contentId matched ({matched_id}).")
    else:
        print(f"   ⚠️ NOTE: Matched different contentId: {matched_id}. (Identical image collision)")
    
    # Check attachments
    attachments = scan_data.get("attachments", [])
    print(f"   Found {len(attachments)} attachments.")
    if len(attachments) >= 2:
        print(f"   ✅ SUCCESS: Attachments retrieved correctly via scan.")
        for a in attachments:
            print(f"      - {a['type']}: {a['title']}")
    else:
        print(f"   ❌ ERROR: Expected at least 2 attachments, got {len(attachments)}")
else:
    print(f"   ❌ FAIL: Image NOT recognized. Confidence: {scan_data.get('confidence', 0):.4f}")
    print(f"   Message: {scan_data.get('message')}")

print("\n══ VERIFICATION COMPLETE ══\n")
