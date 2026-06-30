"""
End-to-end verification script for Step 1 & Step 2.
Tests all API endpoints against the running server on port 5000.
"""
import httpx
import os
import sys
import json
import traceback

BASE = "http://localhost:5000"
API = f"{BASE}/api"
TEST_IMAGE = r"d:\noQR\reference\uploads\images\image-1771234977955-642486162.jpg"

if not os.path.exists(TEST_IMAGE):
    print(f"FAIL: Test image not found at {TEST_IMAGE}")
    sys.exit(1)

print(f"Using test image: {TEST_IMAGE} ({os.path.getsize(TEST_IMAGE)} bytes)")

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name} — {detail}")
        failed += 1

client = httpx.Client(timeout=120)

# ═════════════════════════════════════════════════════════════
print("\n══ STEP 1: Image Upload & Fingerprinting ══\n")
# ═════════════════════════════════════════════════════════════

# Test 1: Root endpoint
print("1. Root endpoint")
r = client.get(f"{BASE}/")
test("GET / returns 200", r.status_code == 200)
body = r.json()
test("Response has name", body.get("name") == "AR Image Recognition System")
test("Response has docs", body.get("docs") == "/docs")

# Test 2: Upload (image + video link)
print("\n2. Upload image + video link")
with open(TEST_IMAGE, "rb") as img:
    r = client.post(f"{API}/upload",
        files={"image": ("test-verify.jpg", img, "image/jpeg")},
        data={"videoLink": "https://youtu.be/test-verification"})
test("POST /api/upload returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:300]}")
if r.status_code != 200:
    print(f"   Upload failed! Response: {r.text[:500]}")
    print(f"   ABORTING — cannot continue without a successful upload.")
    sys.exit(1)
upload_data = r.json()
test("Response has message", upload_data.get("message") == "Upload successful")
test("Response has contentId", "contentId" in upload_data and len(upload_data["contentId"]) > 10)
test("Response has videoUrl", "videoUrl" in upload_data)
CONTENT_ID = upload_data.get("contentId", "")
print(f"   → contentId: {CONTENT_ID}")

# Test 3: List all contents
print("\n3. List contents")
r = client.get(f"{API}/contents")
test("GET /api/contents returns 200", r.status_code == 200)
contents = r.json()
test("Response is a list", isinstance(contents, list))
test("Contains our upload", any(c.get("contentId") == CONTENT_ID for c in contents))

# Test 4: Get single content
print("\n4. Get single content")
r = client.get(f"{API}/content/{CONTENT_ID}")
test("GET /api/content/{id} returns 200", r.status_code == 200)
doc = r.json()
test("Has correct contentId", doc.get("contentId") == CONTENT_ID)
test("Has imagePath", "imagePath" in doc and doc["imagePath"].startswith("/uploads/images/"))
test("Has videoPath", "videoPath" in doc)
test("Has metadata.keypointsCount=2048", doc.get("metadata", {}).get("keypointsCount") == 2048, f"got {doc.get('metadata', {}).get('keypointsCount')}")

# Test 5: Get video URL
print("\n5. Get video URL")
r = client.get(f"{API}/video/{CONTENT_ID}")
test("GET /api/video/{id} returns 200", r.status_code == 200)
test("Response has videoUrl", "videoUrl" in r.json())

# Test 6: 404 for non-existent content
print("\n6. Error handling")
r = client.get(f"{API}/content/nonexistent-id-12345")
test("GET /api/content/bad-id returns 404", r.status_code == 404)

# ═════════════════════════════════════════════════════════════
print("\n\n══ STEP 2: Attach Multimedia Content ══\n")
# ═════════════════════════════════════════════════════════════

# Test 7: Attach text content
print("7. Attach text content")
r = client.post(f"{API}/attach-content", data={
    "contentId": CONTENT_ID,
    "contentType": "text",
    "text": "This is a verification test message.",
    "title": "Test Text"
})
test("POST /api/attach-content (text) returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:300]}")
att_data = r.json()
test("Response has message", att_data.get("message") == "Content attached successfully")
test("Attachment has correct type", att_data.get("attachment", {}).get("type") == "text")
test("Attachment order is 1", att_data.get("attachment", {}).get("order") == 1, f"got {att_data.get('attachment', {}).get('order')}")
TEXT_ATT_ID = att_data.get("attachment", {}).get("attachmentId", "")

# Test 8: Attach video URL
print("\n8. Attach video URL")
r = client.post(f"{API}/attach-content", data={
    "contentId": CONTENT_ID,
    "contentType": "video",
    "url": "https://youtu.be/sample-video",
    "title": "Test Video"
})
test("POST /api/attach-content (video URL) returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:300]}")
att2 = r.json()
test("Video attachment order is 2", att2.get("attachment", {}).get("order") == 2, f"got {att2.get('attachment', {}).get('order')}")
VIDEO_ATT_ID = att2.get("attachment", {}).get("attachmentId", "")

# Test 9: Attach image URL
print("\n9. Attach image URL")
r = client.post(f"{API}/attach-content", data={
    "contentId": CONTENT_ID,
    "contentType": "image",
    "url": "https://example.com/photo.jpg",
    "title": "Test Image"
})
test("POST /api/attach-content (image URL) returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:300]}")

# Test 10: Attach audio URL
print("\n10. Attach audio URL")
r = client.post(f"{API}/attach-content", data={
    "contentId": CONTENT_ID,
    "contentType": "audio",
    "url": "https://example.com/audio.mp3",
    "title": "Test Audio"
})
test("POST /api/attach-content (audio URL) returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:300]}")

# Test 11: Retrieve all attachments
print("\n11. Retrieve attachments")
r = client.get(f"{API}/attached-contents/{CONTENT_ID}")
test("GET /api/attached-contents returns 200", r.status_code == 200)
attachments = r.json()
test("Response is a list", isinstance(attachments, list))
test("Contains 4 attachments", len(attachments) == 4, f"got {len(attachments)}")
types_found = {a["type"] for a in attachments}
test("Has text type", "text" in types_found)
test("Has video type", "video" in types_found)
test("Has audio type", "audio" in types_found)
test("Has image type", "image" in types_found)
test("Attachments are ordered", all(attachments[i]["order"] <= attachments[i+1]["order"] for i in range(len(attachments)-1)))

# Test 12: Validation — invalid content type
print("\n12. Validation tests")
r = client.post(f"{API}/attach-content", data={
    "contentId": CONTENT_ID,
    "contentType": "invalid_type",
    "url": "https://example.com"
})
test("Invalid content type returns 400", r.status_code == 400, f"got {r.status_code}")

# Validation — missing text for text type
r = client.post(f"{API}/attach-content", data={
    "contentId": CONTENT_ID,
    "contentType": "text",
    "text": ""
})
test("Empty text for text type returns 400", r.status_code == 400, f"got {r.status_code}")

# Validation — nonexistent parent image
r = client.post(f"{API}/attach-content", data={
    "contentId": "nonexistent-id-99999",
    "contentType": "text",
    "text": "hello"
})
test("Nonexistent parent returns 404", r.status_code == 404, f"got {r.status_code}")

# Test 13: Delete single attachment
print("\n13. Delete single attachment")
r = client.delete(f"{API}/attached-content/{TEXT_ATT_ID}")
test("DELETE /api/attached-content returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")

# Verify it's gone
r = client.get(f"{API}/attached-contents/{CONTENT_ID}")
remaining = r.json()
test("Attachment count decreased to 3", len(remaining) == 3, f"got {len(remaining)}")

# Test 14: Cascade delete (parent image → attachments)
print("\n14. Cascade delete")
r = client.delete(f"{API}/content/{CONTENT_ID}")
test("DELETE /api/content/{id} returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")

# Verify attachments are gone
r = client.get(f"{API}/attached-contents/{CONTENT_ID}")
test("GET attachments after cascade returns 200", r.status_code == 200)
test("No attachments remain", len(r.json()) == 0, f"got {len(r.json())}")

# Verify content is gone
r = client.get(f"{API}/content/{CONTENT_ID}")
test("Deleted content returns 404", r.status_code == 404)

# ═════════════════════════════════════════════════════════════
print(f"\n{'═' * 50}")
print(f"   RESULTS: {passed} passed, {failed} failed")
print(f"{'═' * 50}")

sys.exit(0 if failed == 0 else 1)
