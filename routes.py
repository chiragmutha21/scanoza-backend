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
import re
import uuid
import shutil
from datetime import datetime, timezone
import numpy as np
import hashlib
import cv2
import httpx
import tempfile

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Body
from typing import Optional, List

from database import get_images_collection, get_attached_contents_collection
from models import (
    ARContentResponse, UploadResponse, VideoLookupResponse, ErrorResponse,
    AttachedContentResponse, AttachContentRequest, ALLOWED_CONTENT_TYPES,
    ScanResponse
)
from embeddings import extract_embedding, extract_robust_embeddings, extract_augmented_embeddings, EMBEDDING_DIM
import faiss_index
from dotenv import load_dotenv
from stegano import lsb
from fingerprint import apply_forced_uniqueness

load_dotenv()

from PIL import Image as PILImage, ImageDraw, ImageFont
import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

def _upload_to_cloudinary(file_path: str, resource_type: str = "auto") -> str:
    try:
        response = cloudinary.uploader.upload(file_path, resource_type=resource_type)
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
VOTE_WEIGHT = 0.30                # Weight for vote consistency
MAX_SCORE_WEIGHT = 0.50           # Weight for max score
AVG_SCORE_WEIGHT = 0.20           # Weight for average score
MIN_VOTE_RATIO = 0.15             # Candidate must win at least this share of crops
HIGH_CONFIDENCE_RELAX = 0.35      # Very high AI score can bypass ORB entirely
ORB_RATIO_TEST = 0.75             # Lowe ratio test threshold
ORB_RANSAC_REPROJ = 5.0           # Homography reprojection threshold
ENFORCE_WATERMARK = False         # Keep false for cross-device reliability (Android/iPhone)
router = APIRouter(prefix="/api")

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


def _normalize_email(email: Optional[str]) -> Optional[str]:
    if not email:
        return None
    return email.strip().lower()


def _remove_file_if_exists(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Failed to remove file {file_path}: {e}")


def _calculate_image_dhash(image_path: str, hash_size: int = 8) -> str:
    try:
        with PILImage.open(image_path) as img:
            # Resize to (hash_size + 1, hash_size), grayscale
            img = img.convert('L').resize((hash_size + 1, hash_size), PILImage.Resampling.LANCZOS)
            pixels = list(img.getdata())
            
            # Compute difference between adjacent pixels
            diff = []
            for row in range(hash_size):
                for col in range(hash_size):
                    pixel_left = pixels[row * (hash_size + 1) + col]
                    pixel_right = pixels[row * (hash_size + 1) + col + 1]
                    diff.append(pixel_left > pixel_right)
            
            # Convert binary array to hex string
            decimal_value = 0
            hex_string = []
            for index, value in enumerate(diff):
                if value:
                    decimal_value += 2**(index % 8)
                if (index % 8) == 7:
                    hex_string.append(hex(decimal_value)[2:].zfill(2))
                    decimal_value = 0
            return "".join(hex_string)
    except Exception as e:
        print(f"Error calculating dhash: {e}")
        return ""


async def _find_duplicate_target(collection, image_hash: str, image_dhash: str):
    if not image_hash and not image_dhash:
        return None
    
    query = {"$or": []}
    if image_hash:
        query["$or"].append({"originalImageHash": image_hash})
    if image_dhash:
        query["$or"].append({"originalImageDHash": image_dhash})
        
    try:
        doc = await collection.find_one(query)
        return doc
    except Exception as e:
        print(f"Error finding duplicate target: {e}")
        return None


async def _is_actual_duplicate(img_path_1: str, img_path_2_or_url: str) -> bool:
    """
    Performs a strict pixel-level and ORB similarity check between two images.
    Returns True only if they are virtually identical (same image).
    Allows visually similar templates (like IN and OUT) to be uploaded.
    """
    try:
        # Load image 1 (local path)
        img1 = cv2.imread(img_path_1, cv2.IMREAD_GRAYSCALE)
        if img1 is None:
            return False
            
        # Load image 2 (local path or URL)
        if img_path_2_or_url.startswith("http"):
            async with httpx.AsyncClient() as client:
                resp = await client.get(img_path_2_or_url)
                if resp.status_code != 200:
                    return False
                img2_data = np.frombuffer(resp.content, np.uint8)
                img2 = cv2.imdecode(img2_data, cv2.IMREAD_GRAYSCALE)
        else:
            local_path = img_path_2_or_url.replace('/uploads/', 'uploads/').lstrip("/")
            # Remove leading slash or backslash
            local_path = re.sub(r'^[/\\]+', '', local_path)
            abs_local_path = os.path.join(os.getcwd(), local_path)
            if not os.path.exists(abs_local_path):
                return False
            img2 = cv2.imread(abs_local_path, cv2.IMREAD_GRAYSCALE)
            
        if img2 is None:
            return False
            
        # Resize to same dimensions for comparison
        h1, w1 = img1.shape
        img2 = cv2.resize(img2, (w1, h1))
        
        # Calculate Mean Squared Error (MSE)
        err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
        err /= float(img1.shape[0] * img1.shape[1])
        
        # If the pixel error is extremely low, they are identical
        if err < 50.0:
            return True
            
        # Also run ORB to check keypoint correspondence
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return False
            
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        match_ratio = len(matches) / min(len(kp1), len(kp2))
        
        # If almost all keypoints match exactly and MSE is low, they are duplicates
        if err < 1500.0 and match_ratio > 0.90:
            return True
            
        return False
    except Exception as e:
        print(f"Error checking actual duplicate: {e}")
        return False


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
            watermarked = watermarked.convert("RGB") # Convert back to JPEG compatible
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
        img = PILImage.open(image_path).convert("RGB")
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
    videoLink: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    user_email: Optional[str] = Form(None),
):
    print(f"POST /api/upload received: image={image.filename}, type={type}")

    owner_email = _normalize_email(user_email)
    if not owner_email or "@" not in owner_email:
        raise HTTPException(
            status_code=401,
            detail="Please sign in with Google before uploading content."
        )

    # Validate image type
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed for target image.")

    image_path, image_filename = await _save_upload(image, "images")
    abs_image_path = os.path.join(os.getcwd(), image_path.lstrip("/"))

    collection = get_images_collection()
    if collection is None:
        _remove_file_if_exists(abs_image_path)
        raise HTTPException(
            status_code=503,
            detail="Database connection is currently unavailable. Please try again."
        )

    try:
        with open(abs_image_path, "rb") as original_file:
            original_image_hash = hashlib.sha256(original_file.read()).hexdigest()
        original_image_dhash = _calculate_image_dhash(abs_image_path)
        duplicate = await _find_duplicate_target(
            collection, original_image_hash, original_image_dhash
        )
    except HTTPException:
        raise
    except Exception as exc:
        _remove_file_if_exists(abs_image_path)
        raise HTTPException(status_code=400, detail=f"Invalid target image: {exc}")

    is_update = False
    if duplicate:
        # Perform deep pixel and ORB validation to verify if this is an actual duplicate
        dup_image_path = duplicate.get("imagePath", "")
        is_actual_dup = await _is_actual_duplicate(abs_image_path, dup_image_path)
        
        if is_actual_dup:
            dup_content_id = duplicate.get("contentId")
            is_missing = False
            if dup_image_path and not dup_image_path.startswith("http"):
                abs_dup_path = os.path.join(os.getcwd(), dup_image_path.lstrip("/\\"))
                if not os.path.exists(abs_dup_path):
                    is_missing = True
            
            if dup_content_id not in faiss_index.id_to_idx:
                is_missing = True

            if is_missing:
                print(f"[RE-INDEX] Target {dup_content_id} is missing its file or FAISS index. Updating/Re-indexing.")
                content_id = dup_content_id
                is_update = True
            else:
                _remove_file_if_exists(abs_image_path)
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "This image already exists in the database. Please upload a different target image.",
                        "duplicateOfContentId": duplicate.get("contentId"),
                    },
                )
        else:
            # Not an actual duplicate, just a similar template! Allow it as a new target image.
            print(f"[UPLOAD] Visually similar template detected but allowed (distinct pixel layout).")
            duplicate = None

    # QUALITY CHECK (Point 9)
    quality = check_image_quality(abs_image_path)
    if not quality["ok"]:
        if os.path.exists(abs_image_path): os.remove(abs_image_path)
        raise HTTPException(status_code=400, detail=quality["msg"])
    print(f"Image Quality Check: {quality['msg']} ({quality['score']} features)")

    if not is_update:
        # Generate content ID early so we can use it as a seed
        content_id = str(uuid.uuid4())

    # ── FORCED UNIQUENESS: Apply invisible alteration before printing ──
    try:
        forced_img = apply_forced_uniqueness(abs_image_path, content_id, techniques=['A', 'B'])
        forced_img.save(abs_image_path, "JPEG", quality=95)
        print(f"[FINGERPRINT] Forced uniqueness applied for {content_id}")
    except Exception as e:
        print(f"[FINGERPRINT ERROR] {e}")

    # STEGANOGRAPHY ONLY: Removing visible noise and text watermarks as per user request.
    # Image will look 100% original to the human eye.

    # Apply Steganography (LSB) to hide `content_id`
    # Must save as PNG to preserve LSB!
    png_path = os.path.splitext(abs_image_path)[0] + ".png"
    try:
        secret_img = lsb.hide(abs_image_path, content_id)
        secret_img.save(png_path)
        if abs_image_path != png_path:
            os.remove(abs_image_path)
        abs_image_path = png_path
        image_path = f"/{UPLOAD_DIR}/images/{os.path.basename(png_path)}"
        print(f"Invisible Steganography successfully embedded content_id {content_id}")
    except Exception as e:
        print(f"Steganography failed: {e}")

    # Cloudinary Upload for target image (Restoring as per user request)
    # Using the PNG path to preserve the invisible watermark
    cloud_image_url = _upload_to_cloudinary(abs_image_path, resource_type="image")
    if not cloud_image_url:
        raise HTTPException(status_code=500, detail="Cloudinary upload failed.")
    
    local_image_path = image_path
    image_path = cloud_image_url
    print(f"Watermarked image successfully uploaded to Cloudinary: {cloud_image_url}")

    # Save attached content
    final_url = url or videoLink or ""
    final_text = text or ""
    
    abs_saved_path = None
    if file and file.filename:
        # Determine subfolder based on type
        sub = "videos" if type == "video" else ("audio" if type == "audio" else ("pdfs" if type == "pdf" else "images"))
        saved_path, _ = await _save_upload(file, sub)
        abs_saved_path = os.path.join(os.getcwd(), saved_path.lstrip("/"))
        cloud_file_url = _upload_to_cloudinary(abs_saved_path, resource_type="auto")
        final_url = cloud_file_url if cloud_file_url else saved_path
    elif type != "text" and not final_url:
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
    if is_update:
        update_data = {
            "imagePath": image_path,
            "localImagePath": local_image_path,
            "originalImageHash": original_image_hash,
            "originalImageDHash": original_image_dhash,
            "userEmail": owner_email,
            "originalImageName": image.filename or "unknown",
            "videoPath": final_url, 
            "videoType": "link" if (url or videoLink) else "file",
            "type": type,
            "title": title,
            "text": final_text,
            "url": final_url,
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        await collection.update_one({"contentId": content_id}, {"$set": update_data})
        print(f"Updated duplicate document {content_id} with new image & embeddings.")
    else:
        doc = {
            "contentId": content_id,
            "userEmail": owner_email,
            "originalImageHash": original_image_hash,
            "originalImageDHash": original_image_dhash,
            "originalImageName": image.filename or "unknown",
            "imagePath": image_path,
            "localImagePath": local_image_path,
            # Default/Legacy mapping for front-end
            "videoPath": final_url, 
            "videoType": "link" if (url or videoLink) else "file",
            # New multi-type support
            "type": type,
            "title": title,
            "text": final_text,
            "url": final_url,
            "descriptorPath": "",
            "fingerprintTechniques": ["A", "B"],
            "metadata": {
                "keypointsCount": EMBEDDING_DIM,
                "fileSize": file.size if file and file.size else 0,
            },
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        print(f"Attempting to insert into DB: {doc['contentId']}")
        await collection.insert_one(doc)
        print("Insert finished.")

    # Delete local temporary files after successful DB write
    if os.path.exists(abs_image_path):
        os.remove(abs_image_path)
    if abs_saved_path and os.path.exists(abs_saved_path) and final_url.startswith("http"):
        os.remove(abs_saved_path)

    # Build response URLs
    base_url = str(request.base_url).rstrip("/")
    video_url = final_url if final_url.startswith("http") else f"{base_url}{final_url}"
    full_image_url = image_path if image_path.startswith("http") else f"{base_url}{image_path}"

    return {
        "message": "Upload successful",
        "contentId": content_id,
        "videoUrl": video_url,
        "imageUrl": full_image_url,
        "descriptorUrl": "",
    }


@router.get("/contents")
async def get_all_contents(email: Optional[str] = None):
    """List only content owned by the supplied logged-in email."""
    collection = get_images_collection()
    if collection is None:
        raise HTTPException(status_code=503, detail="Database connection is currently unavailable.")

    owner_email = _normalize_email(email)
    if not owner_email or "@" not in owner_email:
        raise HTTPException(status_code=401, detail="Google sign-in is required.")

    cursor = collection.find({"userEmail": owner_email}).sort("createdAt", -1)
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc.get("_id", ""))
        results.append(doc)
    return results


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
        cloud_file_url = _upload_to_cloudinary(abs_saved_path, resource_type="auto")
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
    attached_col = get_attached_contents_collection()
    cursor = attached_col.find({"contentId": content_id}).sort("order", 1)
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc.get("_id", ""))
        results.append(doc)
    return results


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

    try:
        content = await frame.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # 1. Steganography Fast-Track Check (LSB is fragile, but worth a shot)
        stego_matched_doc = None
        try:
            from stegano import lsb
            revealed = lsb.reveal(temp_path)
            if revealed and len(revealed) == 36 and '-' in revealed:
                print(f"[STEGO] Steganography payload detected: {revealed}")
                img_col = get_images_collection()
                stego_matched_doc = await img_col.find_one({"contentId": revealed})
        except Exception as e:
            pass

        content_id = None
        score = 0.0

        if stego_matched_doc:
            content_id = stego_matched_doc["contentId"]
            score = 1.0 # Perfect matching
            print(f"Scan result: Steganography perfect match for {content_id}")
            doc = stego_matched_doc
        else:
            # 2. Visual Matching Fallback (Robust Multi-Crop ResNet + FAISS)
            print("LSB failed, attempting robust visual matching...")
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
                    for match_id, match_score in results:
                        # Track max score
                        if match_score > candidate_max_score.get(match_id, 0):
                            candidate_max_score[match_id] = match_score
                        # Track vote count (each crop gets 1 vote for its top match)
                        if results and results[0][0] == match_id:  # Top-1 match
                            candidate_vote_count[match_id] = candidate_vote_count.get(match_id, 0) + 1
                        # Accumulate scores
                        candidate_score_sum[match_id] = candidate_score_sum.get(match_id, 0) + match_score
                
                if candidate_max_score:
                    # Identity-First Analysis (High Sensitivity, Zero Mismatch)
                    sorted_candidates = sorted(candidate_max_score.items(), key=lambda x: -x[1])
                    best_id, best_score = sorted_candidates[0]
                    
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
                    has_required_gap = single_target_mode or (score_gap >= AI_GAP_THRESHOLD)
                    ai_gate_passed = (
                        best_score >= AI_MIN_SCAN_THRESHOLD
                        and weighted_score >= AI_MATCH_THRESHOLD
                        and has_required_gap
                        and has_required_votes
                    )
                    
                    # 2. Decision logic using Config Constants
                    if best_score >= AI_MIN_SCAN_THRESHOLD:
                        print(f"  [VERIFY] Score: {best_score:.2f}. Checking details...")
                        
                        async def get_orb_score(target_id, frame_path):
                            """
                            Performs pixel-level verification using ORB.
                            Handles both local file paths and Cloudinary URLs.
                            """
                            try:
                                # Get original image path from DB
                                img_col = get_images_collection()
                                doc = await img_col.find_one({"contentId": target_id})
                                if not doc: return 0
                                
                                image_path = doc['imagePath']
                                target_img_data = None

                                # Case 1: Cloudinary URL or HTTP URL
                                if image_path.startswith("http"):
                                    async with httpx.AsyncClient() as client:
                                        resp = await client.get(image_path)
                                        if resp.status_code == 200:
                                            target_img_data = np.frombuffer(resp.content, np.uint8)
                                            img1 = cv2.imdecode(target_img_data, cv2.IMREAD_GRAYSCALE)
                                        else:
                                            print(f"[ORB] Failed to download {image_path}: {resp.status_code}")
                                            return 0
                                # Case 2: Local File Path
                                else:
                                    target_path = image_path.replace('/uploads/', 'uploads/').lstrip("/")
                                    if not os.path.exists(target_path):
                                        print(f"[ORB] Local file not found: {target_path}")
                                        return 0
                                    img1 = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
                                
                                if img1 is None: return 0

                                # Read the current scan frame
                                img2 = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                                if img2 is None: return 0
                                
                                # Use ORB (Fast & Efficient)
                                orb = cv2.ORB_create(nfeatures=500)
                                kp1, des1 = orb.detectAndCompute(img1, None)
                                kp2, des2 = orb.detectAndCompute(img2, None)
                                
                                if des1 is None or des2 is None: return 0
                                
                                # Brute-Force Matcher with Ratio Test or Homography
                                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                                matches = bf.knnMatch(des1, des2, k=2)
                                
                                # Apply Lowe's ratio test
                                good_matches = []
                                for m, n in matches:
                                    if m.distance < ORB_RATIO_TEST * n.distance:
                                        good_matches.append(m)
                                
                                # Homography verification (RANSAC)
                                if len(good_matches) > 10:
                                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ORB_RANSAC_REPROJ)
                                    if M is not None:
                                        return int(np.sum(mask))
                                
                                return len(good_matches)
                            except Exception as e:
                                print(f"[ORB ERROR] {e}")
                                return 0

                        # Check Top 1 and Top 2
                        orb_score1 = await get_orb_score(best_id, temp_path)
                        
                        rival_id = sorted_candidates[1][0] if len(sorted_candidates) > 1 else None
                        orb_score2 = await get_orb_score(rival_id, temp_path) if rival_id else 0
                        
                        print(f"  [OPENCV] Scores -> Best: {orb_score1} matches | Rival: {orb_score2} matches")
                        
                        # Swap if the rival has a better pixel-level geometric keypoint match.
                        # This disambiguates extremely similar templates (like IN and OUT targets).
                        if rival_id and orb_score2 > orb_score1:
                            print(f"  [ORB SWAP] Rival has more keypoint matches ({orb_score2} > {orb_score1}). Swapping best target to {rival_id[:12]}")
                            best_id, rival_id = rival_id, best_id
                            orb_score1, orb_score2 = orb_score2, orb_score1
                            best_score, second_best_score = second_best_score, best_score
                            # Re-evaluate gap and gate
                            score_gap = best_score - second_best_score
                            ai_gate_passed = True
                        
                        # 3. Watermark Check
                        is_watermarked = check_watermark_presence(temp_path)
                        
                        # Accept if AI gate passed AND (high confidence or ORB verified)
                        # OR if it's an extremely high confidence AI match (best_score >= 0.50) bypassing the gate
                        if (best_score >= 0.50) or (
                            ai_gate_passed and (
                                best_score >= HIGH_CONFIDENCE_RELAX
                                or (orb_score1 > orb_score2 and orb_score1 >= ORB_THRESHOLD)
                            )
                        ):
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
                    if 'orb_score1' in locals():
                        print(f"  ORB Score: {orb_score1}")
                    print(f"  Matched ID: {best_id if is_match else 'None'}")
                    print(f"---------------------------------\n")

                    if is_match:
                        content_id = best_id
                        score = best_score
                        img_col = get_images_collection()
                        doc = await img_col.find_one({"contentId": content_id})
                        if doc:
                            print(f"  [FINAL MATCH] {content_id}")
                        else:
                            return ScanResponse(matchFound=False, confidence=0, matchPercentage=0, message="No DB data")
                    else:
                        # Improved feedback for near-misses and new users
                        detected_pct = int(best_score * 100)
                        required_pct = int(AI_MATCH_THRESHOLD * 100)
                        
                        if best_score < AI_MIN_SCAN_THRESHOLD:
                            # User requested: No percent, just "Image not uploaded"
                            return ScanResponse(
                                matchFound=False, 
                                confidence=0, 
                                matchPercentage=0,
                                message="Image not uploaded"
                            )
                        else:
                            # AI found a strong candidate, but one or more verification gates failed.
                            gate_reasons = []
                            if not has_required_gap:
                                gate_reasons.append(f"low gap (<{AI_GAP_THRESHOLD:.2f})")
                            if not has_required_votes:
                                gate_reasons.append(f"low vote ratio (<{MIN_VOTE_RATIO:.2f})")
                            if 'orb_score1' in locals() and orb_score1 < ORB_THRESHOLD:
                                gate_reasons.append(f"ORB<{ORB_THRESHOLD}")

                            if gate_reasons:
                                reason_text = ", ".join(gate_reasons)
                                msg = f"AI match {detected_pct}% but verification failed: {reason_text}."
                            else:
                                msg = f"AI match {detected_pct}% but final confidence did not pass (~{required_pct}% weighted threshold)."
                            
                        return ScanResponse(
                            matchFound=False, 
                            confidence=float(best_score), 
                            matchPercentage=detected_pct,
                            message=msg
                        )
                else:
                    return ScanResponse(matchFound=False, confidence=0, matchPercentage=0, message="No match found.")
            except Exception as e:
                print(f"Visual matching error: {e}")
                import traceback; traceback.print_exc()
                return ScanResponse(matchFound=False, confidence=0, matchPercentage=0, message="Scanning error.")

        if not doc:
            return ScanResponse(matchFound=False, confidence=0, matchPercentage=0, message="Match found but content was deleted.")

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
