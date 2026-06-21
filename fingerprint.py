import cv2
import numpy as np
import hashlib
from PIL import Image
import os

def apply_micro_geometric_warp(img: Image.Image, content_id: str) -> Image.Image:
    """
    Technique A: Micro-Geometric Warping (Most Reliable)
    Deterministically warps the image slightly based on content_id.
    """
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    
    # Generate deterministic seed from content_id
    seed = int(hashlib.md5(content_id.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
    
    # Define 4 source points (corners)
    src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    
    # Define 4 destination points with tiny deterministic jitter. Keep this
    # below the visual threshold; it only helps embeddings separate duplicates.
    jitter_scale = 0.0025
    dst_pts = src_pts + rng.uniform(-jitter_scale * min(w, h), jitter_scale * min(w, h), (4, 2)).astype(np.float32)
    
    # Get Perspective Transform Matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply warp
    warped = cv2.warpPerspective(img_cv, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

def inject_adversarial_keypoints(img: Image.Image, content_id: str) -> Image.Image:
    """
    Technique B: Keypoint Injection via Adversarial Noise
    Adds a subtle checkerboard pattern in low-feature areas.
    """
    img_np = np.array(img).astype(np.float32)
    h, w, c = img_np.shape
    
    seed = int(hashlib.md5(content_id.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
    
    # Create a 32x32 noise patch
    patch_size = 32
    noise_patch = rng.choice([-1, 1], size=(patch_size, patch_size, 3)).astype(np.float32)
    
    # Tile it over the image
    noise_full = np.tile(noise_patch, (h // patch_size + 1, w // patch_size + 1, 1))[:h, :w, :]
    
    # Very subtle texture variation for duplicate separation.
    intensity = 1.25
    
    # Apply noise
    altered_np = np.clip(img_np + (noise_full * intensity), 0, 255).astype(np.uint8)
    
    return Image.fromarray(altered_np)

def apply_dct_frequency_shift(img: Image.Image, content_id: str) -> Image.Image:
    """
    Technique C: Frequency Domain Modulation (DCT Alteration)
    Subtly shifts mid-range frequencies.
    """
    try:
        from scipy.fftpack import dct, idct
    except ImportError:
        return img # Fallback if scipy not installed
        
    img_np = np.array(img).astype(np.float32)
    h, w, c = img_np.shape
    
    seed = int(hashlib.md5(content_id.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
    
    # Process in 8x8 blocks for each channel
    altered_np = np.copy(img_np)
    for ch in range(c):
        for i in range(0, h - 7, 8):
            for j in range(0, w - 7, 8):
                block = img_np[i:i+8, j:j+8, ch]
                
                # Forward 2D DCT
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # Alter mid-range frequencies (e.g., [3:6, 3:6])
                shift = rng.uniform(-0.75, 0.75)
                dct_block[3:6, 3:6] += shift
                
                # Inverse 2D DCT
                idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
                altered_np[i:i+8, j:j+8, ch] = idct_block
                
    return Image.fromarray(np.clip(altered_np, 0, 255).astype(np.uint8))

def apply_forced_uniqueness(image_path: str, content_id: str, techniques=['A', 'B']) -> Image.Image:
    """
    Master function to apply selected uniqueness techniques.
    """
    img = Image.open(image_path).convert("RGB")
    
    if 'A' in techniques:
        img = apply_micro_geometric_warp(img, content_id)
        print(f"[FINGERPRINT] Applied Micro-Geometric Warp (A)")
        
    if 'B' in techniques:
        img = inject_adversarial_keypoints(img, content_id)
        print(f"[FINGERPRINT] Applied Adversarial Keypoints (B)")
        
    if 'C' in techniques:
        img = apply_dct_frequency_shift(img, content_id)
        print(f"[FINGERPRINT] Applied DCT Frequency Modulation (C)")
        
    return img
