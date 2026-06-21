import httpx
import os
import random
from PIL import Image

url = "http://127.0.0.1:5000/api/upload"
image_path = "backend/temp_unique_test.jpg"

# Generate a random unique image so that sha256 and dhash are unique
w, h = 300, 300
img = Image.new("RGB", (w, h))
pixels = img.load()
for i in range(w):
    for j in range(h):
        pixels[i, j] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
img.save(image_path)

print(f"Uploading new unique image {image_path} to {url}...")
try:
    with open(image_path, "rb") as f:
        files = {"image": ("temp_unique_test.jpg", f, "image/jpeg")}
        data = {
            "type": "text",
            "text": "Unique Target Test Content",
            "user_email": "chiragmutha31@gmail.com"
        }
        resp = httpx.post(url, files=files, data=data, timeout=30.0)
        print("Status Code:", resp.status_code)
        try:
            print("Response:", resp.json())
        except Exception:
            print("Response text:", resp.text)
finally:
    if os.path.exists(image_path):
        os.remove(image_path)
