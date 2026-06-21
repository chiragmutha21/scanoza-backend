import os
import uuid
import numpy as np
from PIL import Image
from fingerprint import apply_forced_uniqueness
import torch
from embeddings import extract_embedding
import cv2

def test_uniqueness():
    # Setup
    test_image_path = "test_base.jpg"
    # Create a dummy image if not exists
    if not os.path.exists(test_image_path):
        dummy_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(dummy_img).save(test_image_path)
    
    id1 = str(uuid.uuid4())
    id2 = str(uuid.uuid4())
    
    print(f"Testing uniqueness for ID1: {id1} and ID2: {id2}")
    
    # Apply fingerprinting to two copies of the same image
    img1 = apply_forced_uniqueness(test_image_path, id1, techniques=['A', 'B'])
    img2 = apply_forced_uniqueness(test_image_path, id2, techniques=['A', 'B'])
    
    path1 = "test_f1.png"
    path2 = "test_f2.png"
    
    img1.save(path1)
    img2.save(path2)
    
    # 1. Pixel-level difference
    diff = np.abs(np.array(img1).astype(float) - np.array(img2).astype(float))
    mean_diff = np.mean(diff)
    print(f"Mean pixel difference: {mean_diff:.4f}")
    
    # 2. Embedding distance
    emb1 = extract_embedding(path1)
    emb2 = extract_embedding(path2)
    
    # Cosine similarity (dot product of L2 normalized vectors)
    similarity = np.dot(emb1, emb2)
    distance = 1.0 - similarity
    
    print(f"Embedding Cosine Similarity: {similarity:.4f}")
    print(f"Embedding Cosine Distance: {distance:.4f}")
    
    # 3. ORB Keypoint difference
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, _ = orb.detectAndCompute(cv2.imread(path1, 0), None)
    kp2, _ = orb.detectAndCompute(cv2.imread(path2, 0), None)
    
    print(f"ID1 Keypoints: {len(kp1)}")
    print(f"ID2 Keypoints: {len(kp2)}")
    
    # Clean up
    for p in [path1, path2]:
        if os.path.exists(p): os.remove(p)
    if os.path.exists(test_image_path): os.remove(test_image_path)

    assert distance > 0.0001, "Embeddings should be slightly different"
    print("Uniqueness test passed!")

if __name__ == "__main__":
    test_uniqueness()
