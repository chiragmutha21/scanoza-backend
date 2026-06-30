"""
API routes for the Image Recognition Content Trigger System.

Step 1 (Image Upload & Fingerprinting):
  POST   /api/upload          — Upload image + video/link
  GET    /api/contents         — List all content
  GET    /api/video/{id}       — Get video URL
  GET    /api/content/{id}     — Get full content details
  DELETE /api/content/{id}     — Delete content

Step 2 (Attach Multimedia Content):
  POST   /api/attach-content            — Attach content to an image
  GET    /api/attached-contents/{id}    — Get all attachments for an image
  DELETE /api/attached-content/{id}     — Delete a specific attachment
"""
import os
import uuid
import shutil
import re
import secrets
from datetime import datetime, timezone
import numpy as np
import hashlib
import cv2
import httpx
import tempfile

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Body
from typing import Optional, List

from database import get_images_collection, get_attached_contents_collection, get_products_collection, get_orders_collection, get_website_users_collection
from models import (
    ARContentResponse, UploadResponse, VideoLookupResponse, ErrorResponse,
    AttachedContentResponse, AttachContentRequest, ALLOWED_CONTENT_TYPES,
    ScanResponse
)
import json
from embeddings import extract_embedding, extract_robust_embeddings, extract_augmented_embeddings, EMBEDDING_DIM
import faiss_index
from dotenv import load_dotenv
from stegano import lsb
from fingerprint import apply_forced_uniqueness
from invisible_watermark import (
    DEFAULT_REPETITIONS,
    DEFAULT_STRENGTH,
    WATERMARK_VERSION,
    embed_watermark,
    expected_watermark_score,
    extract_watermark,
)

load_dotenv()

from PIL import Image as PILImage, ImageDraw, ImageFont
import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

def _upload_to_cloudinary(file_path: str, resource_type: str = "auto", folder: str = None) -> str:
    try:
        kwargs = {"resource_type": resource_type}
        if folder:
            kwargs["folder"] = folder
        response = cloudinary.uploader.upload(file_path, **kwargs)
        return response.get("secure_url")
    except Exception as e:
        print(f"Cloudinary upload failed: {e}")
        return None

def _delete_from_cloudinary(url: str):
    """Parses a Cloudinary URL and deletes the asset using the admin API / uploader.destroy."""
    if not url or "res.cloudinary.com" not in url:
        return
    try:
        # Extract public_id: https://res.cloudinary.com/[cloud]/[type]/upload/v[version]/[public_id].[ext]
        parts = url.split("/")
        # The public_id is usually the last part before extension
        last_part = parts[-1].split(".")[0]
        # If there are folders, we might need a more complex split, 
        # but for simple uploads, the ID is the filename.
        # Cloudinary also accepts the full path if folders are present.
        
        # A safer way to find the public_id is everything after /upload/v[number]/
        import re
        match = re.search(r'/upload/(?:v\d+/)?(.+)\.[a-z0-9]+$', url)
        if match:
            public_id = match.group(1)
            # Determine resource type (image, video, raw)
            res_type = "video" if "/video/" in url else "image"
            cloudinary.uploader.destroy(public_id, resource_type=res_type)
            print(f"Deleted from Cloudinary: {public_id}")
    except Exception as e:
        print(f"Cloudinary deletion failed: {e}")

# --- Recognition Thresholds & Config ---
SIMILARITY_THRESHOLD = 0.20  # Exposed in /debug/status for quick diagnostics
CONSENSUS_THRESHOLD = 0.18   # Legacy constant retained for backward compatibility
MIN_SCORE_GAP = 0.05         # Legacy constant retained for backward compatibility

# Unified scan decision config (single source of truth)
AI_MATCH_THRESHOLD = 0.22         # Final weighted confidence needed for candidate acceptance
AI_GAP_THRESHOLD = 0.03           # Top-1 minus Top-2 minimum margin (highly relaxed for unique database items)
AI_MIN_SCAN_THRESHOLD = 0.15      # Minimum raw AI score to consider any match
ORB_THRESHOLD = 6                 # Minimum ORB verification score
LEGACY_AI_STRICT_THRESHOLD = 0.60 # Old non-watermarked targets need stronger visual proof
LEGACY_ORB_THRESHOLD = 25         # Strong ORB gate for legacy targets to prevent false positives
VOTE_WEIGHT = 0.30                # Weight for vote consistency
MAX_SCORE_WEIGHT = 0.50           # Weight for max score
AVG_SCORE_WEIGHT = 0.20           # Weight for average score
MIN_VOTE_RATIO = 0.15             # Candidate must win at least this share of crops
HIGH_CONFIDENCE_RELAX = 0.35      # Very high AI score can bypass ORB entirely
ORB_RATIO_TEST = 0.75             # Lowe ratio test threshold
ORB_RANSAC_REPROJ = 5.0           # Homography reprojection threshold
ENFORCE_WATERMARK = False         # Keep false for cross-device reliability (Android/iPhone)
router = APIRouter(prefix="/api")

DUPLICATE_GROUP_REJECT_MESSAGE = (
    "This image has multiple Scanoza memories. Please scan the Scanoza-generated print clearly."
)


async def _duplicate_group_count(collection, doc: dict) -> int:
    """Return how many content records share this source image family."""
    if not doc:
        return 0
    source_group_id = doc.get("sourceGroupId") or doc.get("originalImageHash")
    if not source_group_id:
        return 1
    return await collection.count_documents({
        "$or": [
            {"sourceGroupId": source_group_id},
            {"originalImageHash": source_group_id},
        ]
    })


async def _is_duplicate_group(collection, doc: dict) -> bool:
    return await _duplicate_group_count(collection, doc) > 1


def _reject_duplicate_group(best_score: float) -> ScanResponse:
    return ScanResponse(
        matchFound=False,
        confidence=float(best_score),
        matchPercentage=int(max(0.0, min(best_score, 1.0)) * 100),
        message=DUPLICATE_GROUP_REJECT_MESSAGE,
    )

# ── FAISS index (singleton) ────────────────────────────────────────────────
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index.bin")
FAISS_MAPPING_PATH = os.getenv("FAISS_MAPPING_PATH", "data/id_mapping.json")
faiss_index = faiss_index.FaissIndex(FAISS_INDEX_PATH, FAISS_MAPPING_PATH)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")


def _ensure_dirs():
    """Ensure upload directories exist."""
    for sub in ["images", "videos", "audio", "pdfs", "temp_scans", "failed_scans"]:
        os.makedirs(os.path.join(UPLOAD_DIR, sub), exist_ok=True)
    os.makedirs("data", exist_ok=True)


_ensure_dirs()


@router.get("/debug/status")
async def debug_status():
    """Debug endpoint: shows FAISS index and DB stats to diagnose issues."""
    collection = get_images_collection()
    db_count = 0
    db_items = []
    if collection is not None:
        cursor = collection.find()
        async for doc in cursor:
            db_count += 1
            db_items.append({
                "contentId": doc.get("contentId", "?"),
                "name": doc.get("originalImageName", "?"),
                "imagePath": doc.get("imagePath", "?"),
                "inFaiss": doc.get("contentId", "") in faiss_index.id_to_idx,
            })

    return {
        "faissTotal": faiss_index.total,
        "faissIds": list(faiss_index.id_to_idx.keys()),
        "dbCount": db_count,
        "dbItems": db_items,
        "threshold": SIMILARITY_THRESHOLD,
    }


async def _save_upload(file: UploadFile, subfolder: str) -> tuple[str, str]:
    """
    Save an uploaded file to disk.

    Returns:
        (relative_path, filename) e.g. ("/uploads/images/abc.jpg", "abc.jpg")
    """
    ext = os.path.splitext(file.filename or "file")[1]
    unique_name = f"{file.filename.split('.')[0]}-{int(datetime.now().timestamp())}-{uuid.uuid4().hex[:8]}{ext}"
    dest_dir = os.path.join(UPLOAD_DIR, subfolder)
    dest_path = os.path.join(dest_dir, unique_name)

    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    relative_path = f"/{UPLOAD_DIR}/{subfolder}/{unique_name}"
    return relative_path, unique_name


def convert_to_rgb_with_white_bg(img: PILImage.Image) -> PILImage.Image:
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        alpha = img.convert('RGBA').split()[-1]
        bg = PILImage.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=alpha)
        return bg.convert('RGB')
    return img.convert('RGB')

def _add_watermark(image_path: str):
    """Adds a 'SCANOZA TARGET' watermark to the image for user feedback."""
    try:
        abs_path = os.path.join(os.getcwd(), image_path.lstrip("/"))
        with PILImage.open(abs_path) as img:
            # Convert to RGBA for transparency
            img = img.convert("RGBA")
            txt = PILImage.new("RGBA", img.size, (255, 255, 255, 0))
            
            # Simple text watermark
            from PIL import ImageDraw
            draw = ImageDraw.Draw(txt)
            
            # Try to use a font, fallback to default
            try:
                # Assuming some common path or default
                font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()

            width, height = img.size
            text = "SCANOZA AR TARGET"
            # Draw at bottom right
            draw.text((width - 150, height - 30), text, fill=(255, 255, 255, 128), font=font)
            
            watermarked = PILImage.alpha_composite(img, txt)
            watermarked = convert_to_rgb_with_white_bg(watermarked) # Convert back to JPEG compatible
            watermarked.save(abs_path, "JPEG", quality=95)
            print(f"Watermark added to {abs_path}")
    except Exception as e:
        print(f"Failed to add watermark: {e}")

def apply_stealth_noise(image_path: str, seed_str: str, intensity: float = 12.0):
    """
    Applies a low-frequency, deterministic noise pattern to the image to alter its deep learning embedding.
    The noise survives real-world camera scanning by being low frequency (smooth).
    """
    try:
        img = PILImage.open(image_path)
        img = convert_to_rgb_with_white_bg(img)
        img_np = np.array(img, dtype=np.float32)
        height, width, _ = img_np.shape

        hash_val = int(hashlib.md5(seed_str.encode('utf-8')).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(hash_val)

        noise_dim = 32
        noise_small = rng.uniform(-1, 1, (noise_dim, noise_dim, 3)).astype(np.float32)
        noise_img = PILImage.fromarray(((noise_small + 1) * 127.5).astype(np.uint8))
        noise_scaled_img = noise_img.resize((width, height), PILImage.BICUBIC)
        
        noise_scaled = (np.array(noise_scaled_img, dtype=np.float32) / 127.5) - 1.0
        noise_scaled *= intensity

        watermarked_np = np.clip(img_np + noise_scaled, 0, 255).astype(np.uint8)

        watermarked_img = PILImage.fromarray(watermarked_np)
        watermarked_img.save(image_path, "JPEG", quality=95)
        print(f"Stealth noise applied to {image_path} with seed {seed_str}")
    except Exception as e:
        print(f"Failed to apply stealth noise: {e}")

def calculate_file_hash(file_path: str) -> str:
    """Calculate a stable SHA-256 hash for exact duplicate tracking."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()

def calculate_image_dhash(file_path: str, hash_size: int = 16) -> str:
    """Calculate a perceptual dHash for visually duplicate target detection."""
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image")
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return f"{value:0{(hash_size * hash_size) // 4}x}"

def hamming_distance_hex(left: str, right: str) -> int:
    """Return bit distance between two equal-length hex hashes."""
    if not left or not right or len(left) != len(right):
        return 10**9
    return (int(left, 16) ^ int(right, 16)).bit_count()

async def find_duplicate_target(collection, original_hash: str, image_dhash: str):
    """Find an exact or visually near-identical target already stored in MongoDB."""
    exact_doc = await collection.find_one({"originalImageHash": original_hash})
    if exact_doc:
        return exact_doc, "exact_hash", 1.0

    # 16x16 dHash has 256 bits; <= 10 is very strict for same/near-same images.
    cursor = collection.find({"originalImageDHash": {"$regex": r"^[0-9a-f]+$"}})
    async for doc in cursor:
        existing_hash = doc.get("originalImageDHash")
        distance = hamming_distance_hex(image_dhash, existing_hash)
        if distance <= 10:
            score = 1.0 - (distance / 256)
            return doc, "perceptual_hash", score

    return None, "", 0.0

def remove_file_if_exists(path: str):
    if path and os.path.exists(path):
        os.remove(path)

def check_image_quality(image_path: str) -> dict:
    """
    Checks if the image has enough features for reliable recognition.
    Returns a score and a recommendation.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return {"score": 0, "ok": False, "msg": "Invalid image"}
        
        orb = cv2.ORB_create(nfeatures=1000)
        kp = orb.detect(img, None)
        count = len(kp)
        
        # Heuristics for Scanoza
        if count < 150:
            return {"score": count, "ok": False, "msg": "Image too plain/blurry. Add more details."}
        elif count < 400:
            return {"score": count, "ok": True, "msg": "Moderate quality. May be harder to scan."}
        else:
            return {"score": count, "ok": True, "msg": "High quality target."}
    except:
        return {"score": 0, "ok": True, "msg": "Quality check failed"}

def check_watermark_presence(frame_path: str) -> bool:
    """
    Checks if the 'SCANOZA' watermark area has features.
    In a real-world camera shot, the original image on a screen 
    won't have the high-contrast 'SCANOZA AR TARGET' text in the corner.
    """
    try:
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return False
        h, w = img.shape
        
        # Focus on bottom-right corner (where watermark is added)
        # Watermark is roughly at (w-150, h-30)
        roi = img[max(0, h-80):h, max(0, w-200):w]
        
        # Use a high-sensitivity ORB to find the sharp edges of the text
        orb = cv2.ORB_create(nfeatures=100, edgeThreshold=10)
        kp = orb.detect(roi, None)
        
        # If there are very few features in the watermark zone, 
        # it's likely the 'clean' original image.
        print(f"  [WATERMARK] Detected {len(kp)} features in corner.")
        return len(kp) >= 12 # Threshold for watermark text presence
    except:
        return True # Default to true to avoid false rejections on errors

# ── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/upload")
async def upload_status():
    """Health check for upload endpoint."""
    return {"message": "Upload endpoint is active. Use POST to upload files."}


@router.post("/upload")
async def upload_content(
    request: Request,
    image: UploadFile = File(...),
    # Support multiple types
    type: str = Form("video"), # video, audio, image, text, pdf
    title: Optional[str] = Form(""),
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    user_email: Optional[str] = Form(None),
):
    print(f"POST /api/upload received: image={image.filename}, type={type}")
    normalized_user_email = user_email.strip().lower() if user_email else None

    # Validate image type
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed for target image.")

    image_path, image_filename = await _save_upload(image, "images")
    abs_image_path = os.path.join(os.getcwd(), image_path.lstrip("/"))
    original_abs_path = abs_image_path
    original_hash = calculate_file_hash(original_abs_path)

    # QUALITY CHECK (Point 9)
    quality = check_image_quality(abs_image_path)
    if not quality["ok"]:
        if os.path.exists(abs_image_path): os.remove(abs_image_path)
        raise HTTPException(status_code=400, detail=quality["msg"])
    print(f"Image Quality Check: {quality['msg']} ({quality['score']} features)")
    original_dhash = calculate_image_dhash(original_abs_path)

    # Generate content ID early so we can use it as a seed
    content_id = str(uuid.uuid4())
    watermark_id = secrets.token_hex(8)

    collection = get_images_collection()
    if collection is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection is currently unavailable. Please check your MongoDB Altas whitelist/connection."
        )

    # Detect duplicate source images before any Cloudinary/FAISS/DB writes.
    # Same target images should not become separate memories.
    possible_duplicate = False
    existing_duplicate_id = None
    duplicate_score = 0.0
    duplicate_exact_hash = False
    try:
        duplicate_doc, duplicate_method, duplicate_score = await find_duplicate_target(
            collection,
            original_hash,
            original_dhash,
        )
        if duplicate_doc:
            possible_duplicate = True
            duplicate_exact_hash = duplicate_method == "exact_hash"
            existing_duplicate_id = duplicate_doc.get("contentId")
            print(
                f"[DUPLICATE] Existing target image rejected: {existing_duplicate_id} "
                f"method={duplicate_method} score={duplicate_score:.4f}"
            )
        elif faiss_index.total > 0:
            original_embedding = extract_embedding(original_abs_path)
            duplicate_results = faiss_index.search(original_embedding, k=3)
            if duplicate_results:
                existing_duplicate_id, duplicate_score = duplicate_results[0]
                if duplicate_score >= 0.94:
                    duplicate_doc = await collection.find_one({"contentId": existing_duplicate_id})
                    if duplicate_doc:
                        possible_duplicate = True
                        print(
                            f"[DUPLICATE] Visual source match: {existing_duplicate_id} "
                            f"score={duplicate_score:.4f}"
                        )
                    else:
                        print(
                            f"[DUPLICATE] Ignoring stale FAISS match: {existing_duplicate_id} "
                            f"score={duplicate_score:.4f}"
                        )
    except Exception as e:
        print(f"[DUPLICATE] Duplicate check failed: {e}")

    if possible_duplicate:
        remove_file_if_exists(original_abs_path)
        raise HTTPException(
            status_code=409,
            detail={
                "message": "This image already exists in the database. Please upload a different target image.",
                "duplicateOfContentId": existing_duplicate_id,
                "duplicateScore": float(duplicate_score),
            },
        )

    # Keep the uploaded target exactly as the user provided it. Do not create a
    # generated/watermarked copy, because that can alter transparent backgrounds,
    # dimensions, and visible pixels.
    watermark_metrics = None
    print(f"[TARGET] Preserving original uploaded image for {content_id}: {image_path}")

    # Save attached content
    final_url = url or ""
    final_text = text or ""
    
    if file and file.filename:
        # Determine subfolder based on type
        sub = "videos" if type == "video" else ("audio" if type == "audio" else ("pdfs" if type == "pdf" else "images"))
        saved_path, _ = await _save_upload(file, sub)
        abs_saved_path = os.path.join(os.getcwd(), saved_path.lstrip("/"))
        cloud_file_url = _upload_to_cloudinary(abs_saved_path, resource_type="auto", folder="uploaded")
        final_url = cloud_file_url if cloud_file_url else saved_path
    elif type != "text" and not url:
         raise HTTPException(status_code=400, detail=f"A file or URL is required for type '{type}'")

    # Extract deep learning embeddings (Augmented for robustness)
    try:
        # Now indexing multiple variations for the same content_id
        augmented_embeddings = extract_augmented_embeddings(abs_image_path)
        for emb in augmented_embeddings:
            faiss_index.add(emb, content_id)
    except Exception as e:
        # Clean up saved file on failure
        if os.path.exists(abs_image_path):
            os.remove(abs_image_path)
        raise HTTPException(status_code=500, detail=f"Failed to extract augmented embeddings: {str(e)}")

    # Save metadata to MongoDB
    doc = {
        "contentId": content_id,
        "userEmail": normalized_user_email,
        "watermarkId": None,
        "watermarkVersion": None,
        "watermarkStrength": None,
        "watermarkRepetitions": None,
        "originalImageName": image.filename or "unknown",
        "imagePath": image_path,
        "originalImageHash": original_hash,
        "originalImageDHash": original_dhash,
        "sourceGroupId": original_hash,
        "isDuplicateSource": possible_duplicate,
        "duplicateOfContentId": existing_duplicate_id if possible_duplicate else None,
        "duplicateScore": float(duplicate_score),
        "duplicateExactHash": duplicate_exact_hash,
        # Default/Legacy mapping for front-end
        "videoPath": final_url, 
        "videoType": "link" if url else "file",
        # New multi-type support
        "type": type,
        "title": title,
        "text": final_text,
        "url": final_url,
        "descriptorPath": "",
        "fingerprintTechniques": [],
        "qualityMetrics": {
            "psnr": watermark_metrics.psnr if watermark_metrics else None,
            "ssim": watermark_metrics.ssim if watermark_metrics else None,
        },
        "metadata": {
            "keypointsCount": EMBEDDING_DIM,
            "fileSize": file.size if file and file.size else 0,
        },
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    print(f"Attempting to insert into DB: {doc['contentId']}")
    await collection.insert_one(doc)
    print("Insert finished.")

    # Build response URLs
    base_url = str(request.base_url).rstrip("/")
    video_url = final_url if final_url.startswith("http") else f"{base_url}{final_url}"
    full_image_url = image_path if image_path.startswith("http") else f"{base_url}{image_path}"
    message = "Upload successful. Your original target image was preserved without visual changes."

    return {
        "message": message,
        "contentId": content_id,
        "videoUrl": video_url,
        "imageUrl": full_image_url,
        "descriptorUrl": "",
        "isDuplicateSource": possible_duplicate,
        "duplicateOfContentId": existing_duplicate_id if possible_duplicate else None,
        "duplicateScore": float(duplicate_score),
    }


@router.get("/contents")
async def get_all_contents(email: Optional[str] = None):
    """List only the logged-in user's content, sorted newest first."""
    try:
        collection = get_images_collection()
        if collection is None:
            print("[CONTENTS] DB unavailable; returning empty list")
            return []
        normalized_email = email.strip().lower() if email else None
        if not normalized_email:
            return []

        cursor = collection.find({
            "userEmail": {
                "$regex": f"^{re.escape(normalized_email)}$",
                "$options": "i",
            }
        }).sort("createdAt", -1)
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc.get("_id", ""))
            results.append(doc)
        return results
    except Exception as e:
        print(f"[CONTENTS] Failed to load content list: {e}")
        return []


@router.post("/register-user")
async def register_user(payload: dict = Body(...)):
    """Accept frontend user sync calls so dashboard loading stays clean."""
    email = payload.get("email")
    user_id = payload.get("userId") or email
    if not email:
        raise HTTPException(status_code=400, detail="email is required")
    return {"message": "User synced", "email": email, "userId": user_id}


@router.get("/video/{content_id}")
async def get_video(content_id: str, request: Request):
    """Get video URL for a specific content item."""
    collection = get_images_collection()
    doc = await collection.find_one({"contentId": content_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Content not found")

    base_url = str(request.base_url).rstrip("/")
    video_path = doc.get("videoPath", "")
    video_url = video_path if video_path.startswith("http") else f"{base_url}{video_path}"

    return {"videoUrl": video_url}


@router.get("/content/{content_id}")
async def get_content(content_id: str):
    """Get full details for a specific content item."""
    collection = get_images_collection()
    doc = await collection.find_one({"contentId": content_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Content not found")

    doc["_id"] = str(doc.get("_id", ""))
    return doc


@router.put("/content/{content_id}")
async def update_content(
    content_id: str,
    type: str = Form("video"),
    title: Optional[str] = Form(""),
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
):
    """Update existing content metadata and linked media."""
    collection = get_images_collection()
    doc = await collection.find_one({"contentId": content_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Content not found")

    final_url = url or ""
    final_text = text or ""

    if file and file.filename:
        sub = "videos" if type == "video" else ("audio" if type == "audio" else ("pdfs" if type == "pdf" else "images"))
        saved_path, _ = await _save_upload(file, sub)
        abs_saved_path = os.path.join(os.getcwd(), saved_path.lstrip("/"))
        cloud_file_url = _upload_to_cloudinary(abs_saved_path, resource_type="auto", folder="uploaded")
        final_url = cloud_file_url if cloud_file_url else saved_path
    
    update_data = {
        "type": type,
        "title": title or "",
        "text": final_text,
        "url": final_url,
        "videoPath": final_url,
        "videoType": "link" if final_url.startswith("http") else "file",
    }
    
    await collection.update_one({"contentId": content_id}, {"$set": update_data})
    
    return {"message": "Content updated successfully"}


@router.delete("/content/{content_id}")
async def delete_content(content_id: str):
    """Delete a content item from FAISS and MongoDB, and remove files. Cascade-deletes attached contents."""
    collection = get_images_collection()
    doc = await collection.find_one({"contentId": content_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Content not found")

    # Remove files from disk or Cloudinary
    for path_key in ["imagePath", "videoPath"]:
        rel_path = doc.get(path_key, "")
        if rel_path:
            if rel_path.startswith("http"):
                _delete_from_cloudinary(rel_path)
            else:
                abs_path = os.path.join(os.getcwd(), rel_path.lstrip("/"))
                if os.path.exists(abs_path):
                    os.remove(abs_path)

    # Cascade-delete all attached contents (Step 2)
    attached_col = get_attached_contents_collection()
    attached_cursor = attached_col.find({"contentId": content_id})
    async for att in attached_cursor:
        att_url = att.get("url", "")
        if att_url:
            if att_url.startswith("http"):
                _delete_from_cloudinary(att_url)
            else:
                att_abs = os.path.join(os.getcwd(), att_url.lstrip("/"))
                if os.path.exists(att_abs):
                    os.remove(att_abs)
    await attached_col.delete_many({"contentId": content_id})

    # Remove from FAISS
    faiss_index.remove(content_id)

    # Remove from MongoDB
    await collection.delete_one({"contentId": content_id})

    return {"message": "Content deleted successfully"}


@router.get("/fingerprint/preview/{content_id}")
async def get_fingerprint_preview(content_id: str):
    """Diagnostic endpoint to see the difference between original and unique fingerprint."""
    collection = get_images_collection()
    doc = await collection.find_one({"contentId": content_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Content not found")

    image_path = doc.get("imagePath", "")
    if not image_path:
        raise HTTPException(status_code=404, detail="Image path not found")
    
    # In a real scenario, we might want to compare with the absolute original if we kept it.
    # For now, we just return the current one.
    base_url = "http://localhost:8000" # fallback
    full_url = image_path if image_path.startswith("http") else f"{image_path}"
    
    return {
        "contentId": content_id,
        "imageUrl": full_url,
        "techniques": doc.get("fingerprintTechniques", []),
        "message": "This is the invisibly altered version ready for printing."
    }


@router.get("/search")
async def search_similar(content_id: str, k: int = 5):
    """
    Search for images similar to an existing content item.
    (Bonus endpoint for future use in Step 3.)
    """
    collection = get_images_collection()
    doc = await collection.find_one({"contentId": content_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Content not found")

    abs_image_path = os.path.join(os.getcwd(), doc["imagePath"].lstrip("/"))
    embedding = extract_embedding(abs_image_path)
    results = faiss_index.search(embedding, k=k)

    return {
        "query": content_id,
        "results": [{"contentId": cid, "score": score} for cid, score in results],
    }


# ══════════════════════════════════════════════════════════════════════════
# Step 2: Attach Multimedia Content
# ══════════════════════════════════════════════════════════════════════════

_CONTENT_TYPE_SUBFOLDER = {
    "video": "videos",
    "audio": "audio",
    "image": "images",
    "pdf": "pdfs",
}


@router.post("/attach-content")
async def attach_content(
    request: Request,
    contentId: str = Form(...),
    contentType: str = Form(...),
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    title: Optional[str] = Form(""),
):
    """
    Attach a piece of content (video/audio/image/text/pdf) to an existing image.
    Supports file upload, URL, or text depending on content type.
    """
    # Validate content type
    if contentType not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type '{contentType}'. Allowed: {ALLOWED_CONTENT_TYPES}",
        )

    # Validate that parent image exists
    images_col = get_images_collection()
    parent = await images_col.find_one({"contentId": contentId})
    if not parent:
        raise HTTPException(status_code=404, detail="Image not found. Upload the image first.")

    # Determine the content value (file, url, or text)
    final_url = None
    final_text = None

    if contentType == "text":
        # Text content: require the text field
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text field is required for text content.")
        final_text = text.strip()
    else:
        # Media content: need either a file or a URL
        has_file = file is not None and file.filename
        has_url = url is not None and url.strip() != ""

        if not has_file and not has_url:
            raise HTTPException(
                status_code=400,
                detail=f"Either a file or a URL is required for {contentType} content.",
            )

        if has_file:
            subfolder = _CONTENT_TYPE_SUBFOLDER.get(contentType, "uploads")
            saved_path, _ = await _save_upload(file, subfolder)
            final_url = saved_path
        else:
            final_url = url.strip()

    # Calculate order (next in sequence)
    attached_col = get_attached_contents_collection()
    existing_count = await attached_col.count_documents({"contentId": contentId})

    # Create attachment document
    attachment_id = str(uuid.uuid4())
    doc = {
        "attachmentId": attachment_id,
        "contentId": contentId,
        "type": contentType,
        "url": final_url,
        "text": final_text,
        "title": title or "",
        "order": existing_count + 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    await attached_col.insert_one(doc)

    return {
        "message": "Content attached successfully",
        "attachment": {
            "attachmentId": attachment_id,
            "contentId": contentId,
            "type": contentType,
            "url": final_url,
            "text": final_text,
            "title": title or "",
            "order": existing_count + 1,
        },
    }


@router.get("/attached-contents/{content_id}")
async def get_attached_contents(content_id: str):
    """Get all attached contents for an image, sorted by order."""
    try:
        attached_col = get_attached_contents_collection()
        if attached_col is None:
            return []
        cursor = attached_col.find({"contentId": content_id}).sort("order", 1)
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc.get("_id", ""))
            results.append(doc)
        return results
    except Exception as e:
        print(f"[ATTACHMENTS] Failed to load attachments for {content_id}: {e}")
        return []


@router.delete("/attached-content/{attachment_id}")
async def delete_attached_content(attachment_id: str):
    """Delete a specific attached content item and its file."""
    attached_col = get_attached_contents_collection()
    doc = await attached_col.find_one({"attachmentId": attachment_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Attachment not found")

    # Remove file from disk or Cloudinary
    file_url = doc.get("url", "")
    if file_url:
        if file_url.startswith("http"):
            _delete_from_cloudinary(file_url)
        else:
            abs_path = os.path.join(os.getcwd(), file_url.lstrip("/"))
            if os.path.exists(abs_path):
                os.remove(abs_path)

    await attached_col.delete_one({"attachmentId": attachment_id})
    return {"message": "Attachment deleted successfully"}


@router.post("/scan", response_model=ScanResponse)
async def scan_frame(
    frame: UploadFile = File(...),
):
    """
    Scan a camera frame for a match in the FAISS index.
    Returns matched content and its attachments.
    """
    # Save frame temporarily
    temp_dir = os.path.join(UPLOAD_DIR, "temp_scans")
    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = f"scan_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}.jpg"
    temp_path = os.path.join(temp_dir, temp_filename)
    is_match = False
    best_score = 0.0

    try:
        content = await frame.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        is_watermarked = check_watermark_presence(temp_path)

        # 1. DCT/QIM invisible watermark fast-track. This is the exact identity
        # layer that differentiates visually identical uploaded targets.
        dct_matched_doc = None
        try:
            watermark_result = extract_watermark(temp_path)
            if watermark_result.watermark_id:
                print(
                    f"[DCT WATERMARK] Decoded {watermark_result.watermark_id} "
                    f"confidence={watermark_result.confidence:.3f} "
                    f"agreement={watermark_result.bit_agreement:.3f}"
                )
                img_col = get_images_collection()
                dct_matched_doc = await img_col.find_one({"watermarkId": watermark_result.watermark_id})
                if not dct_matched_doc:
                    print(f"[DCT WATERMARK] Decoded ID not found in DB: {watermark_result.watermark_id}")
            else:
                print(
                    f"[DCT WATERMARK] No valid payload "
                    f"confidence={watermark_result.confidence:.3f} "
                    f"crc={watermark_result.valid_crc}"
                )
        except Exception as e:
            print(f"[DCT WATERMARK] Extraction failed: {e}")

        # 2. Legacy LSB compatibility check. New uploads do not depend on this.
        stego_matched_doc = None
        if not dct_matched_doc:
            try:
                from stegano import lsb
                revealed = lsb.reveal(temp_path)
                if revealed and len(revealed) == 36 and '-' in revealed:
                    print(f"[STEGO] Steganography payload detected: {revealed}")
                    img_col = get_images_collection()
                    stego_matched_doc = await img_col.find_one({"contentId": revealed})
            except Exception:
                pass

        content_id = None
        score = 0.0
        doc = None

        if dct_matched_doc:
            content_id = dct_matched_doc["contentId"]
            score = max(0.90, float(watermark_result.confidence))
            best_score = score
            is_match = True
            print(f"Scan result: DCT watermark exact match for {content_id}")
            doc = dct_matched_doc
        elif stego_matched_doc:
            content_id = stego_matched_doc["contentId"]
            score = 1.0 # Perfect matching
            best_score = score
            is_match = True
            print(f"Scan result: Steganography perfect match for {content_id}")
            doc = stego_matched_doc
        else:
            # 3. Visual Matching Fallback (Robust Multi-Crop ResNet + FAISS)
            print("Watermark extraction failed, attempting robust visual matching...")
            try:
                # Use robust multi-crop/enhanced embeddings for scanning
                query_embeddings = extract_robust_embeddings(temp_path)
                
                # Track both max score AND vote count per candidate
                candidate_max_score = {}  # id -> max_score
                candidate_vote_count = {}  # id -> number of crops that matched this id
                candidate_score_sum = {}   # id -> sum of all scores (for averaging)
                
                total_queries = len(query_embeddings)
                
                for i, emb in enumerate(query_embeddings):
                    results = faiss_index.search(emb, k=3)
                    if results:
                        top_match_id = results[0][0]
                        candidate_vote_count[top_match_id] = candidate_vote_count.get(top_match_id, 0) + 1
                    seen_ids = set()
                    for match_id, match_score in results:
                        if match_id in seen_ids:
                            continue
                        seen_ids.add(match_id)
                        # Track max score
                        if match_score > candidate_max_score.get(match_id, 0):
                            candidate_max_score[match_id] = match_score
                        # Accumulate scores
                        candidate_score_sum[match_id] = candidate_score_sum.get(match_id, 0) + match_score
                
                if candidate_max_score:
                    # Identity-First Analysis (High Sensitivity, Zero Mismatch)
                    sorted_candidates = sorted(candidate_max_score.items(), key=lambda x: -x[1])
                    best_id, best_score = sorted_candidates[0]

                    watermark_assisted_pass = False
                    watermark_scores = {}
                    best_candidate_doc = None
                    try:
                        img_col = get_images_collection()
                        for candidate_id, _candidate_score in sorted_candidates[:5]:
                            candidate_doc = await img_col.find_one({"contentId": candidate_id})
                            candidate_wm = candidate_doc.get("watermarkId") if candidate_doc else None
                            if candidate_wm:
                                watermark_scores[candidate_id] = expected_watermark_score(temp_path, candidate_wm)
                        if watermark_scores:
                            ranked_wm = sorted(watermark_scores.items(), key=lambda x: -x[1])
                            wm_best_id, wm_best_score = ranked_wm[0]
                            wm_second_score = ranked_wm[1][1] if len(ranked_wm) > 1 else 0.0
                            # Assisted watermark is only a tie-breaker. With a single
                            # watermarked candidate, do not treat "gap vs zero" as proof.
                            enough_wm_evidence = (
                                (len(ranked_wm) > 1 and wm_best_score >= 0.40 and (wm_best_score - wm_second_score) >= 0.08)
                                or wm_best_score >= 0.62
                            )
                            if enough_wm_evidence:
                                best_id = wm_best_id
                                best_score = candidate_max_score.get(best_id, best_score)
                                watermark_assisted_pass = True
                                print(
                                    f"  [DCT ASSIST] Candidate watermark selected {best_id[:12]} "
                                    f"score={wm_best_score:.3f} gap={(wm_best_score - wm_second_score):.3f}"
                                )
                            else:
                                print(f"  [DCT ASSIST] Weak/ambiguous pilot scores: {watermark_scores}")
                        best_candidate_doc = await img_col.find_one({"contentId": best_id})
                    except Exception as e:
                        print(f"  [DCT ASSIST] Scoring failed: {e}")
                    
                    if best_candidate_doc is None:
                        img_col = get_images_collection()
                        best_candidate_doc = await img_col.find_one({"contentId": best_id})
                    best_has_dct_watermark = bool(best_candidate_doc and best_candidate_doc.get("watermarkId"))
                    best_is_duplicate_group = await _is_duplicate_group(img_col, best_candidate_doc)

                    second_best_score = sorted_candidates[1][1] if len(sorted_candidates) > 1 else 0.0
                    score_gap = best_score - second_best_score
                    single_target_mode = len(candidate_max_score) == 1 and faiss_index.total == 1
                    
                    # ── Double Layer Verification (AI + OpenCV) ──────────────────
                    is_match = False
                    
                    # 1. Decision Factor: weighted score from max + average + vote ratio
                    vote_ratio = candidate_vote_count.get(best_id, 0) / total_queries
                    avg_score = candidate_score_sum.get(best_id, 0.0) / max(total_queries, 1)
                    weighted_score = (
                        (best_score * MAX_SCORE_WEIGHT)
                        + (avg_score * AVG_SCORE_WEIGHT)
                        + (vote_ratio * VOTE_WEIGHT)
                    )
                    has_required_votes = vote_ratio >= MIN_VOTE_RATIO
                    has_required_gap = watermark_assisted_pass or single_target_mode or (score_gap >= AI_GAP_THRESHOLD) or (best_score >= LEGACY_AI_STRICT_THRESHOLD)
                    ai_gate_passed = (
                        best_score >= AI_MIN_SCAN_THRESHOLD
                        and weighted_score >= AI_MATCH_THRESHOLD
                        and has_required_gap
                        and has_required_votes
                    )
                    
                    # 2. Decision logic using Config Constants
                    if best_score >= AI_MIN_SCAN_THRESHOLD:
                        print(f"  [VERIFY] Score: {best_score:.2f}. Checking details...")
                        
                        async def get_detailed_similarity(target_id, frame_path):
                            """
                            Loads both images, aligns them using ORB keypoint homography warping,
                            and calculates:
                            1. ORB inliers count
                            2. Resized Normalized Cross-Correlation (entire image comparison)
                            Returns a dict: {"orb": inliers_count, "correlation": correlation_coefficient}
                            """
                            try:
                                # Get original image path from DB
                                img_col = get_images_collection()
                                doc = await img_col.find_one({"contentId": target_id})
                                if not doc: return {"orb": 0, "correlation": 0.0}
                                
                                image_path = doc['imagePath']
                                img1_bgr = None

                                # Case 1: Cloudinary URL or HTTP URL
                                if image_path.startswith("http"):
                                    async with httpx.AsyncClient() as client:
                                        resp = await client.get(image_path)
                                        if resp.status_code == 200:
                                            target_img_data = np.frombuffer(resp.content, np.uint8)
                                            img1_bgr = cv2.imdecode(target_img_data, cv2.IMREAD_COLOR)
                                        else:
                                            print(f"[VERIFY] Failed to download {image_path}: {resp.status_code}")
                                            return {"orb": 0, "correlation": 0.0}
                                # Case 2: Local File Path
                                else:
                                    rel_path = image_path
                                    if "uploads/" in rel_path:
                                        rel_path = rel_path[rel_path.find("uploads/"):]
                                    elif "uploads\\" in rel_path:
                                        rel_path = rel_path[rel_path.find("uploads\\"):]

                                    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                                    abs_local_path = os.path.normpath(
                                        os.path.join(BASE_DIR, rel_path.lstrip("/\\"))
                                    )
                                    if not os.path.exists(abs_local_path):
                                        print(f"[VERIFY] Local file not found: {abs_local_path}")
                                        return {"orb": 0, "correlation": 0.0}
                                    img1_bgr = cv2.imread(abs_local_path, cv2.IMREAD_COLOR)
                                
                                if img1_bgr is None: return {"orb": 0, "correlation": 0.0}

                                # Read the current scan frame in color
                                img2_bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                                if img2_bgr is None: return {"orb": 0, "correlation": 0.0}
                                
                                # Convert to grayscale for feature matching
                                img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
                                img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
                                
                                # Use higher features for better target alignment in cluttered scans
                                orb = cv2.ORB_create(nfeatures=1500)
                                kp1, des1 = orb.detectAndCompute(img1_gray, None)
                                kp2, des2 = orb.detectAndCompute(img2_gray, None)
                                
                                inliers = 0
                                correlation = 0.0
                                if des1 is not None and des2 is not None:
                                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                                    matches = bf.knnMatch(des1, des2, k=2)
                                    
                                    good_matches = []
                                    for m, n in matches:
                                        if m.distance < ORB_RATIO_TEST * n.distance:
                                            good_matches.append(m)
                                    
                                    if len(good_matches) > 10:
                                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ORB_RANSAC_REPROJ)
                                        if M is not None:
                                            inliers = int(np.sum(mask))
                                            
                                            # Warp current frame using inverse homography to align with template coordinates
                                            h, w, c = img1_bgr.shape
                                            warped_query = cv2.warpPerspective(img2_bgr, np.linalg.inv(M), (w, h))
                                            
                                            # Compare the entire aligned images
                                            t_resized = cv2.resize(img1_bgr, (300, 300))
                                            wq_resized = cv2.resize(warped_query, (300, 300))
                                            
                                            res = cv2.matchTemplate(wq_resized, t_resized, cv2.TM_CCOEFF_NORMED)
                                            correlation = float(res[0][0])
                                        else:
                                            inliers = len(good_matches)
                                    else:
                                        inliers = len(good_matches)
                                        
                                return {"orb": inliers, "correlation": correlation}
                            except Exception as e:
                                print(f"[DETAILED SIM ERROR] {e}")
                                return {"orb": 0, "correlation": 0.0}

                        # Compute detailed similarity for Top 1 and Top 2
                        sim1 = await get_detailed_similarity(best_id, temp_path)
                        orb_score1 = sim1["orb"]
                        correlation1 = sim1.get("correlation", 0.0)
                        
                        rival_id = sorted_candidates[1][0] if len(sorted_candidates) > 1 else None
                        sim2 = await get_detailed_similarity(rival_id, temp_path) if rival_id else {"orb": 0, "correlation": 0.0}
                        orb_score2 = sim2["orb"]
                        correlation2 = sim2.get("correlation", 0.0)
                        
                        print(f"  [OPENCV ALIGNED] Scores -> Best ({best_id[:8]}): ORB={orb_score1}, Correlation={correlation1:.3f} | Rival ({rival_id[:8] if rival_id else 'None'}): ORB={orb_score2}, Correlation={correlation2:.3f}")
                        
                        # Disambiguate based on template cross-correlation and keypoint matches.
                        should_swap = False
                        if rival_id:
                            # If the rival has a significantly better correlation and has at least 15 inliers, swap.
                            if correlation2 > correlation1 + 0.08 and orb_score2 >= 15:
                                print(f"  [CORRELATION SWAP] Rival has significantly better template correlation ({correlation2:.3f} > {correlation1:.3f})")
                                should_swap = True
                                    
                        if should_swap:
                            best_id, rival_id = rival_id, best_id
                            orb_score1, orb_score2 = orb_score2, orb_score1
                            correlation1, correlation2 = correlation2, correlation1
                            best_score, second_best_score = second_best_score, best_score
                            score_gap = best_score - second_best_score
                            ai_gate_passed = True
                            # Reload candidate doc for swapped ID
                            best_candidate_doc = await img_col.find_one({"contentId": best_id})
                            best_has_dct_watermark = bool(best_candidate_doc and best_candidate_doc.get("watermarkId"))
                            best_is_duplicate_group = await _is_duplicate_group(img_col, best_candidate_doc)
                            
                        # Check for ambiguity (same logo / template conflict)
                        is_ambiguous = False
                        if rival_id:
                            # If both templates have high similarity and are very close, it means they share the same logo/template
                            # and we cannot safely distinguish them. In this case, we skip/ignore it.
                            if correlation1 >= 0.55 and correlation2 >= 0.55:
                                if abs(correlation1 - correlation2) < 0.08 and abs(orb_score1 - orb_score2) < 10:
                                    is_ambiguous = True
                                    print(f"  [AMBIGUOUS] Same logo/template detected with close scores ({correlation1:.3f} vs {correlation2:.3f}). Skipping match")
                        # Accept match logic
                        high_confidence_visual_match = (best_score >= 0.60 and orb_score1 >= 25)
                        accept_match = False
                        if is_ambiguous:
                            print(f"  [REJECTED] Ambiguous match (same logo conflict). Skipping.")
                        elif high_confidence_visual_match:
                            # If we have an extremely strong visual match (high AI score + high ORB geometry matches),
                            # we accept it to prevent false negatives from print/scan distortion or unwatermarked uploads.
                            accept_match = True
                        elif best_has_dct_watermark:
                            # For watermarked targets, never show output on visual
                            # similarity alone. Blind DCT exact match already returned
                            # above; this fallback needs assisted DCT plus ORB geometry.
                            accept_match = (
                                ai_gate_passed
                                and watermark_assisted_pass
                                and orb_score1 > orb_score2
                                and orb_score1 >= ORB_THRESHOLD
                            )
                        else:
                            # Legacy targets do not have DCT IDs. Keep them usable, but require
                            # much stronger visual proof so random camera frames do not auto-open output.
                            accept_match = (
                                ai_gate_passed
                                and best_score >= LEGACY_AI_STRICT_THRESHOLD
                                and orb_score1 > orb_score2
                                and orb_score1 >= LEGACY_ORB_THRESHOLD
                            )

                        if accept_match:
                            if ENFORCE_WATERMARK and not is_watermarked:
                                return ScanResponse(
                                    matchFound=False, 
                                    confidence=float(best_score), 
                                    matchPercentage=int(best_score*100),
                                    message="Please use watermarked image"
                                )
                            if not is_watermarked:
                                print("  [WATERMARK] Not detected, but allowed (ENFORCE_WATERMARK=False).")
                            is_match = True
                            print(f"  [VERIFIED] Identity confirmed: {best_id[:12]}")
                        else:
                            print(f"  [REJECTED] Verification failed.")
                            print(f"  [REJECTED] Too much ambiguity even for OpenCV.")
                    
                    # Final Logging for Scan (Requested)
                    print(f"\n[SCAN LOG] {'--- ACCEPTED ---' if is_match else '--- REJECTED ---'}")
                    print(f"  Top AI Score: {best_score:.4f}")
                    print(f"  Avg AI Score: {avg_score:.4f}")
                    print(f"  Weighted Score: {weighted_score:.4f}")
                    print(f"  Second AI Score: {second_best_score:.4f}")
                    print(f"  Score Gap: {score_gap:.4f}")
                    print(f"  Single Target Mode: {single_target_mode}")
                    print(f"  Vote Count: {candidate_vote_count.get(best_id, 0)}/{total_queries}")
                    print(f"  Vote Ratio: {vote_ratio:.4f} (min {MIN_VOTE_RATIO:.2f})")
                    print(f"  AI Gate Passed: {ai_gate_passed}")
                    print(f"  Candidate Has DCT Watermark: {best_has_dct_watermark}")
                    print(f"  Candidate Is Duplicate Group: {best_is_duplicate_group}")
                    if watermark_scores:
                        print(f"  DCT Assist Scores: {watermark_scores}")
                        print(f"  DCT Assist Passed: {watermark_assisted_pass}")
                    if 'orb_score1' in locals():
                        print(f"  ORB Score: {orb_score1}")
                    print(f"  Matched ID: {best_id if is_match else 'None'}")
                    print(f"---------------------------------\n")

                    if is_match:
                        content_id = best_id
                        score = best_score
                        img_col = get_images_collection()
                        if img_col is None:
                            return ScanResponse(matchFound=False, confidence=0, matchPercentage=0, message="Database connection unavailable")
                        doc = await img_col.find_one({"contentId": content_id})
                        if doc:
                            print(f"  [FINAL MATCH] {content_id}")
                        else:
                            return ScanResponse(matchFound=False, confidence=0, matchPercentage=0, message="image not in database")
                    else:
                        # Improved feedback for near-misses and new users
                        detected_pct = int(best_score * 100)
                        
                        return ScanResponse(
                            matchFound=False, 
                            confidence=float(best_score), 
                            matchPercentage=detected_pct,
                            message="image not in database"
                        )
                else:
                    return ScanResponse(matchFound=False, confidence=0, matchPercentage=0, message="image not in database")
            except Exception as e:
                print(f"Visual matching error: {e}")
                import traceback; traceback.print_exc()
                return ScanResponse(matchFound=False, confidence=0, matchPercentage=0, message="image not in database")

        if not doc:
            return ScanResponse(matchFound=False, confidence=0, matchPercentage=0, message="image not in database")

        doc["_id"] = str(doc.get("_id", ""))
        match_content = ARContentResponse(**doc)

        # Fetch attachments
        attached_col = get_attached_contents_collection()
        cursor = attached_col.find({"contentId": content_id}).sort("order", 1)
        attachments = []
        async for att_doc in cursor:
            att_doc["_id"] = str(att_doc.get("_id", ""))
            attachments.append(AttachedContentResponse(**att_doc))

        return ScanResponse(
            matchFound=True,
            confidence=float(score),
            matchPercentage=int(score * 100),
            content=match_content,
            attachments=attachments,
            message="Match found!"
        )

    except Exception as e:
        print(f"Scan fatal error: {e}")
        return ScanResponse(matchFound=False, confidence=0, matchPercentage=0, message="Internal error")
    finally:
        # Cleanup or Save for Debugging (Point 2)
        if os.path.exists(temp_path):
            if 'is_match' in locals() and not is_match and best_score > 0.05:
                failed_dir = os.path.join(UPLOAD_DIR, "failed_scans")
                failed_path = os.path.join(failed_dir, os.path.basename(temp_path))
                shutil.move(temp_path, failed_path)
                print(f"  [DEBUG] Failed scan saved to {failed_path}")
            else:
                os.remove(temp_path)


# ══════════════════════════════════════════════════════════════════════════
# Products API
# ══════════════════════════════════════════════════════════════════════════

DEFAULT_PRODUCTS = [
    {
        "productId": "1",
        "title": "Distressed White Farmhouse Frame",
        "category": "Birthday",
        "price": "₹299",
        "originalPrice": "₹699",
        "image": "https://images.unsplash.com/photo-1513519245088-0e12902e5a38?auto=format&fit=crop&q=80&w=800",
        "description": "Premium handcrafted distressed white farmhouse frame. Give your warm memories a charming rustic look.",
        "createdAt": datetime.now(timezone.utc).isoformat()
    },
    {
        "productId": "2",
        "title": "Rustic Dark Wood Frame",
        "category": "Anniversary",
        "price": "₹200",
        "originalPrice": "₹399",
        "image": "https://images.unsplash.com/photo-1607604276583-eef5d076aa5f?auto=format&fit=crop&q=80&w=800",
        "description": "Classic dark wood profile with deep grain texture. Perfect for anniversary memories and premium portraits.",
        "createdAt": datetime.now(timezone.utc).isoformat()
    },
    {
        "productId": "3",
        "title": "Classic Gold Ornate Frame",
        "category": "Family",
        "price": "₹250",
        "originalPrice": "₹450",
        "image": "https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?auto=format&fit=crop&q=80&w=800",
        "description": "Elegant gold baroque frame with beautiful ornate carvings. A luxurious addition to any living space.",
        "createdAt": datetime.now(timezone.utc).isoformat()
    },
    {
        "productId": "4",
        "title": "Petite Memories Mini Frame",
        "category": "Return Gift",
        "price": "₹499",
        "originalPrice": "₹799",
        "image": "https://images.unsplash.com/photo-1544457070-4cd773b4d71e?auto=format&fit=crop&q=80&w=800",
        "description": "Small table frames, ideal for returning gifts and event giveaways.",
        "createdAt": datetime.now(timezone.utc).isoformat()
    },
    {
        "productId": "5",
        "title": "Personalized Square Keychain",
        "category": "Keychains",
        "price": "₹199",
        "image": "https://images.unsplash.com/photo-1574634534894-89d7576c8259?auto=format&fit=crop&q=80&w=800",
        "description": "Carry your loved ones wherever you go with our Personalized Square Keychain, made from high-quality stainless steel.",
        "createdAt": datetime.now(timezone.utc).isoformat()
    }
]


@router.get("/products")
async def get_products():
    """Retrieve all products. If empty, automatically seed defaults."""
    col = get_products_collection()
    if col is None:
        return DEFAULT_PRODUCTS

    count = await col.count_documents({})
    if count == 0:
        for p in DEFAULT_PRODUCTS:
            await col.insert_one(p.copy())
    
    cursor = col.find({}).sort("createdAt", -1)
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc.get("_id", ""))
        results.append(doc)
    return results


@router.post("/products")
async def create_product(
    title: str = Form(...),
    price: str = Form(...),
    category: str = Form(...),
    description: str = Form(...),
    size: str = Form(...),
    originalPrice: Optional[str] = Form(None),
    sizes: Optional[str] = Form(None),  # JSON string
    image: Optional[UploadFile] = File(None),
    imageUrl: Optional[str] = Form(None),
    imageFiles: Optional[List[UploadFile]] = File(None),
    imageUrls: Optional[str] = Form(None)  # comma separated list
):
    """Create a new product in database."""
    col = get_products_collection()
    if col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    all_images = []

    # Process first/main image if provided
    main_image = imageUrl or ""
    if image and image.filename:
        saved_path, _ = await _save_upload(image, "images")
        abs_saved_path = os.path.join(os.getcwd(), saved_path.lstrip("/"))
        cloud_url = _upload_to_cloudinary(abs_saved_path, resource_type="image", folder="products")
        main_image = cloud_url if cloud_url else saved_path

    if main_image:
        all_images.append(main_image)

    # Process multiple image files
    if imageFiles:
        for img_file in imageFiles:
            if img_file.filename:
                saved_path, _ = await _save_upload(img_file, "images")
                abs_saved_path = os.path.join(os.getcwd(), saved_path.lstrip("/"))
                cloud_url = _upload_to_cloudinary(abs_saved_path, resource_type="image", folder="products")
                url = cloud_url if cloud_url else saved_path
                all_images.append(url)

    # Process comma separated image URLs
    if imageUrls:
        for url in imageUrls.split(","):
            url = url.strip()
            if url:
                all_images.append(url)

    # Clean up empty strings or duplicates
    all_images = list(dict.fromkeys([x for x in all_images if x]))

    if not all_images:
        raise HTTPException(status_code=400, detail="At least one product image is required")

    # Fallback if main image wasn't set specifically
    if not main_image:
        main_image = all_images[0]

    # Parse sizes
    parsed_sizes = []
    if sizes:
        try:
            parsed_sizes = json.loads(sizes)
        except Exception as e:
            print(f"Error parsing sizes: {e}")

    product_doc = {
        "productId": str(uuid.uuid4()),
        "title": title,
        "price": price if price.startswith("₹") else f"₹{price}",
        "originalPrice": originalPrice if not originalPrice or originalPrice.startswith("₹") else f"₹{originalPrice}",
        "category": category,
        "size": size,
        "description": description,
        "image": main_image,
        "images": all_images,
        "sizes": parsed_sizes,
        "createdAt": datetime.now(timezone.utc).isoformat()
    }
    await col.insert_one(product_doc)
    product_doc["_id"] = str(product_doc.get("_id", ""))
    return {"message": "Product created successfully", "product": product_doc}


@router.delete("/products/{product_id}")
async def delete_product(product_id: str):
    """Delete a product by ID."""
    col = get_products_collection()
    if col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    doc = await col.find_one({"productId": product_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Product not found")

    image_url = doc.get("image", "")
    if image_url:
        if image_url.startswith("http"):
            _delete_from_cloudinary(image_url)
        else:
            abs_path = os.path.join(os.getcwd(), image_url.lstrip("/"))
            if os.path.exists(abs_path):
                os.remove(abs_path)

    # Also clean up all other images in the list
    other_images = doc.get("images", [])
    for img_url in other_images:
        if img_url != image_url:
            if img_url.startswith("http"):
                _delete_from_cloudinary(img_url)
            else:
                abs_path = os.path.join(os.getcwd(), img_url.lstrip("/"))
                if os.path.exists(abs_path):
                    os.remove(abs_path)

    await col.delete_one({"productId": product_id})
    return {"message": "Product deleted successfully"}


@router.put("/products/{product_id}")
async def update_product(
    product_id: str,
    title: Optional[str] = Form(None),
    price: Optional[str] = Form(None),
    originalPrice: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    size: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    sizes: Optional[str] = Form(None),  # JSON string
    image: Optional[UploadFile] = File(None),
    imageUrl: Optional[str] = Form(None),
    imageFiles: Optional[List[UploadFile]] = File(None),
    imageUrls: Optional[str] = Form(None)
):
    """Update an existing product by ID."""
    col = get_products_collection()
    if col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    doc = await col.find_one({"productId": product_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Product not found")

    update_data = {}
    if title is not None:
        update_data["title"] = title
    if price is not None:
        update_data["price"] = price if price.startswith("₹") else f"₹{price}"
    if originalPrice is not None:
        update_data["originalPrice"] = originalPrice if not originalPrice or originalPrice.startswith("₹") else f"₹{originalPrice}"
    if category is not None:
        update_data["category"] = category
    if size is not None:
        update_data["size"] = size
    if description is not None:
        update_data["description"] = description
    if sizes is not None:
        try:
            update_data["sizes"] = json.loads(sizes)
        except Exception as e:
            print(f"Error parsing sizes: {e}")

    # Handle image updates
    # We will build the new images list if imageUrls, imageFiles, image, or imageUrl is provided.
    # Note: frontend sends remaining urls in imageUrls, and new files in imageFiles.
    has_image_update = (imageUrls is not None) or (imageFiles is not None) or (image is not None) or (imageUrl is not None)
    
    if has_image_update:
        final_images = []
        
        # 1. Add remaining existing URLs (if imageUrls parameter was provided)
        if imageUrls is not None:
            for url in imageUrls.split(","):
                url = url.strip()
                if url:
                    final_images.append(url)
        else:
            # If imageUrls is not provided, default to current images
            final_images.extend(doc.get("images", []))
            if not final_images and doc.get("image"):
                final_images.append(doc["image"])

        # 2. Process new main image upload if any
        if image and image.filename:
            saved_path, _ = await _save_upload(image, "images")
            abs_saved_path = os.path.join(os.getcwd(), saved_path.lstrip("/"))
            cloud_url = _upload_to_cloudinary(abs_saved_path, resource_type="image", folder="products")
            main_img_url = cloud_url if cloud_url else saved_path
            final_images.append(main_img_url)
        elif imageUrl is not None and imageUrl.strip():
            if imageUrl not in final_images:
                final_images.append(imageUrl)

        # 3. Process new multiple image files if any
        if imageFiles:
            for img_file in imageFiles:
                if img_file.filename:
                    saved_path, _ = await _save_upload(img_file, "images")
                    abs_saved_path = os.path.join(os.getcwd(), saved_path.lstrip("/"))
                    cloud_url = _upload_to_cloudinary(abs_saved_path, resource_type="image", folder="products")
                    url = cloud_url if cloud_url else saved_path
                    final_images.append(url)

        # Clean list
        final_images = list(dict.fromkeys([x for x in final_images if x]))
        
        if not final_images:
            raise HTTPException(status_code=400, detail="At least one product image is required")
            
        update_data["images"] = final_images
        update_data["image"] = final_images[0]

    if update_data:
        await col.update_one({"productId": product_id}, {"$set": update_data})

    updated_doc = await col.find_one({"productId": product_id})
    updated_doc["_id"] = str(updated_doc.get("_id", ""))
    return {"message": "Product updated successfully", "product": updated_doc}


# ══════════════════════════════════════════════════════════════════════════
@router.get("/orders/lookup")
async def lookup_orders(query: str):
    """Look up orders by customer email or contact number."""
    col = get_orders_collection()
    if col is None:
        return []

    clean_query = query.strip()
    if not clean_query:
        return []

    cursor = col.find({
        "$or": [
            {"customerDetails.contact": clean_query},
            {"customerDetails.email": {"$regex": f"^{re.escape(clean_query)}$", "$options": "i"}}
        ]
    }).sort("createdAt", -1)

    results = []
    async for doc in cursor:
        doc["_id"] = str(doc.get("_id", ""))
        results.append(doc)
    return results


@router.get("/orders")
async def get_orders():
    """Retrieve all placed orders."""
    col = get_orders_collection()
    if col is None:
        return []

    cursor = col.find({}).sort("createdAt", -1)
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc.get("_id", ""))
        results.append(doc)
    return results


@router.post("/orders")
async def create_order(
    customerName: str = Form(...),
    customerContact: str = Form(...),
    customerAddress: str = Form(...),
    customerPincode: str = Form(...),
    customerEmail: Optional[str] = Form(""),
    items: str = Form(...),  # JSON string
    total: str = Form(...),
    arType: str = Form("text"),
    arText: Optional[str] = Form(None),
    arLink: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    arFile: Optional[UploadFile] = File(None)
):
    """Place a new order with optional uploaded photo and AR content."""
    col = get_orders_collection()
    if col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    # Save customer frame photo
    frame_photo_url = ""
    frame_photo_hash = ""
    frame_photo_dhash = ""
    if image and image.filename:
        saved_path, _ = await _save_upload(image, "images")
        abs_saved_path = os.path.join(os.getcwd(), saved_path.lstrip("/"))
        try:
            frame_photo_hash = calculate_file_hash(abs_saved_path)
            frame_photo_dhash = calculate_image_dhash(abs_saved_path)
        except Exception as eh:
            print(f"Error calculating hashes: {eh}")
        cloud_url = _upload_to_cloudinary(abs_saved_path, resource_type="image", folder="orders")
        frame_photo_url = cloud_url if cloud_url else saved_path

    # Save AR attachment file
    ar_file_url = ""
    if arFile and arFile.filename:
        subfolder = _CONTENT_TYPE_SUBFOLDER.get(arType, "images")
        saved_path, _ = await _save_upload(arFile, subfolder)
        abs_saved_path = os.path.join(os.getcwd(), saved_path.lstrip("/"))
        cloud_url = _upload_to_cloudinary(abs_saved_path, resource_type="auto", folder="orders")
        ar_file_url = cloud_url if cloud_url else saved_path

    try:
        parsed_items = json.loads(items)
    except:
        parsed_items = [{"title": "Custom Frame Story Piece", "price": total, "quantity": 1}]

    order_doc = {
        "orderId": str(uuid.uuid4()),
        "customerDetails": {
            "name": customerName,
            "contact": customerContact,
            "address": customerAddress,
            "pincode": customerPincode,
            "email": customerEmail
        },
        "items": parsed_items,
        "total": total,
        "arType": arType,
        "arText": arText,
        "arLink": arLink,
        "framePhoto": frame_photo_url,
        "framePhotoHash": frame_photo_hash,
        "framePhotoDHash": frame_photo_dhash,
        "arFile": ar_file_url,
        "status": "Pending",
        "createdAt": datetime.now(timezone.utc).isoformat()
    }
    await col.insert_one(order_doc)
    return {"message": "Order placed successfully", "orderId": order_doc["orderId"]}


@router.post("/orders/check-duplicate-frame")
async def check_duplicate_frame(
    frame: UploadFile = File(...),
):
    """
    Check if the uploaded customer frame photo already exists in any placed order.
    Uses both SHA-256 (exact) and perceptual dHash (visual similarity).
    """
    col = get_orders_collection()
    if col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    # Save to temp location to compute hash
    temp_dir = os.path.join(UPLOAD_DIR, "temp_scans")
    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = f"check_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}.jpg"
    temp_path = os.path.join(temp_dir, temp_filename)

    try:
        content = await frame.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Compute hashes
        uploaded_hash = calculate_file_hash(temp_path)
        uploaded_dhash = calculate_image_dhash(temp_path)

        # 1. Check for exact SHA-256 match
        exact_doc = await col.find_one({"framePhotoHash": uploaded_hash})
        if exact_doc:
            print(f"[DUPLICATE FRAME] Exact match found with order {exact_doc.get('orderId')}")
            return {"matchFound": True, "message": "This image already exists in our database. Please upload a different image."}

        # 2. Check for perceptual dHash match
        cursor = col.find({"framePhotoDHash": {"$regex": r"^[0-9a-f]+$"}})
        async for doc in cursor:
            existing_dhash = doc.get("framePhotoDHash")
            distance = hamming_distance_hex(uploaded_dhash, existing_dhash)
            if distance <= 10:
                print(f"[DUPLICATE FRAME] Perceptual match found with order {doc.get('orderId')} (distance={distance})")
                return {"matchFound": True, "message": "This image already exists in our database. Please upload a different image."}

        return {"matchFound": False}

    except Exception as e:
        print(f"Error checking duplicate frame: {e}")
        return {"matchFound": False}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.delete("/orders/{order_id}")
async def delete_order(order_id: str):
    """Delete/Cancel an order by ID."""
    col = get_orders_collection()
    if col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    doc = await col.find_one({"orderId": order_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Order not found")

    # Cleanup files
    for url in [doc.get("framePhoto"), doc.get("arFile")]:
        if url:
            if url.startswith("http"):
                _delete_from_cloudinary(url)
            else:
                abs_path = os.path.join(os.getcwd(), url.lstrip("/"))
                if os.path.exists(abs_path):
                    os.remove(abs_path)

    await col.delete_one({"orderId": order_id})
    return {"message": "Order deleted successfully"}


@router.put("/orders/{order_id}")
async def update_order(
    order_id: str,
    status: Optional[str] = Form(None),
    customerName: Optional[str] = Form(None),
    customerContact: Optional[str] = Form(None),
    customerAddress: Optional[str] = Form(None),
    customerPincode: Optional[str] = Form(None)
):
    """Update an existing order (status or customer details) by ID."""
    col = get_orders_collection()
    if col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    doc = await col.find_one({"orderId": order_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Order not found")

    update_data = {}
    if status is not None:
        update_data["status"] = status

    customer_details = doc.get("customerDetails", {})
    if customerName is not None:
        customer_details["name"] = customerName
    if customerContact is not None:
        customer_details["contact"] = customerContact
    if customerAddress is not None:
        customer_details["address"] = customerAddress
    if customerPincode is not None:
        customer_details["pincode"] = customerPincode

    update_data["customerDetails"] = customer_details

    await col.update_one({"orderId": order_id}, {"$set": update_data})

    updated_doc = await col.find_one({"orderId": order_id})
    updated_doc["_id"] = str(updated_doc.get("_id", ""))
    return {"message": "Order updated successfully", "order": updated_doc}


import base64
import json
import random
import string

def decode_google_jwt(credential: str):
    try:
        parts = credential.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1]
        padded = payload_b64 + "=" * (4 - len(payload_b64) % 4)
        decoded_bytes = base64.urlsafe_b64decode(padded)
        payload = json.loads(decoded_bytes.decode("utf-8"))
        return payload
    except Exception as e:
        print(f"Error decoding JWT: {e}")
        return None

@router.post("/auth/google")
async def google_auth(payload: dict = Body(...)):
    credential = payload.get("credential")
    if not credential:
        raise HTTPException(status_code=400, detail="Credential is required")
        
    user_info = decode_google_jwt(credential)
    if not user_info:
        raise HTTPException(status_code=400, detail="Invalid token")
        
    email = user_info.get("email")
    name = user_info.get("name", "")
    picture = user_info.get("picture", "")
    
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in token")
        
    users_col = get_website_users_collection()
    if users_col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
        
    user = await users_col.find_one({"email": email})
    if not user:
        while True:
            rand_suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
            user_id = f"USR-{rand_suffix}"
            existing = await users_col.find_one({"userId": user_id})
            if not existing:
                break
                
        user = {
            "userId": user_id,
            "email": email,
            "name": name,
            "picture": picture,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "details": {}
        }
        await users_col.insert_one(user)
    else:
        await users_col.update_one(
            {"email": email},
            {"$set": {"name": name, "picture": picture}}
        )
        user = await users_col.find_one({"email": email})
        
    user["_id"] = str(user["_id"])
    return user

@router.put("/users/profile")
async def update_user_profile(payload: dict = Body(...)):
    email = payload.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    
    users_col = get_website_users_collection()
    if users_col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
        
    user = await users_col.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    current_details = user.get("details", {}) or {}
    details = {
        "name": payload.get("name", current_details.get("name", "")),
        "contact": payload.get("contact", current_details.get("contact", "")),
        "address": payload.get("address", current_details.get("address", "")),
        "pincode": payload.get("pincode", current_details.get("pincode", ""))
    }
    
    await users_col.update_one(
        {"email": email},
        {"$set": {"details": details}}
    )
    
    updated_user = await users_col.find_one({"email": email})
    updated_user["_id"] = str(updated_user["_id"])
    return updated_user
