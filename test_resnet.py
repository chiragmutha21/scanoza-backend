import numpy as np
import cv2
from embeddings import extract_embedding
from routes import apply_stealth_noise
import uuid
import shutil
import os

img_path = "uploads/dummy.jpg"
if not os.path.exists(img_path):
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.imwrite(img_path, img)

# Test 1
shutil.copy(img_path, "test1.jpg")
apply_stealth_noise("test1.jpg", str(uuid.uuid4()), intensity=18.0)
emb1 = extract_embedding("test1.jpg")

# Test 2
shutil.copy(img_path, "test2.jpg")
apply_stealth_noise("test2.jpg", str(uuid.uuid4()), intensity=18.0)
emb2 = extract_embedding("test2.jpg")

# Test Original
emb_orig = extract_embedding(img_path)

print("Sim Orig vs 1:", np.dot(emb_orig, emb1))
print("Sim Orig vs 2:", np.dot(emb_orig, emb2))
print("Sim 1 vs 2:", np.dot(emb1, emb2))
