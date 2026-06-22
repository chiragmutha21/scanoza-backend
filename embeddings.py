"""
Deep learning embedding extraction using ResNet-50.

Extracts a 2048-dimensional feature vector from an image using a pretrained
ResNet-50 with the final classification layer removed. Vectors are L2-normalized
so that inner product equals cosine similarity.

Enhanced with multi-crop and augmented extraction for robust real-world
scanning (physical objects like keychains, printed images, etc.)
"""
import numpy as np
import cv2
import os
import requests
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageEnhance, ImageFilter

# ── Model Setup (singleton, loaded once) ───────────────────────────────────

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet-50 pretrained on ImageNet, remove the classification head
_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
_model = torch.nn.Sequential(*list(_model.children())[:-1])  # Remove fc layer → outputs (batch, 2048, 1, 1)
_model.eval()
_model.to(_device)

# Standard ImageNet preprocessing
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

EMBEDDING_DIM = 2048

print(f"Embedding model loaded on {_device} (ResNet-50, {EMBEDDING_DIM}-d)")


def load_image_from_path_or_url(image_path: str, grayscale: bool = False) -> np.ndarray:
    """Load an OpenCV image from a local path or HTTP/Cloudinary URL."""
    if not image_path:
        raise ValueError("Empty image path or URL")

    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    if image_path.lower().startswith(("http://", "https://")):
        response = requests.get(image_path, timeout=20.0)
        response.raise_for_status()
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), mode)
        if image is None:
            raise ValueError(f"Failed to decode remote image: {image_path}")
        return image

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.normpath(
        os.path.join(
            BASE_DIR,
            image_path.lstrip("/\\")
        )
    )
    image = cv2.imread(local_path, mode)
    if image is None:
        raise FileNotFoundError(f"Could not load local image: {local_path}")
    return image


def _opencv_to_pil(image: np.ndarray) -> Image.Image:
    """Convert a BGR OpenCV image to an RGB PIL image without temp files."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _extract_single(img: Image.Image) -> np.ndarray:
    """Extract embedding from a single PIL Image (already RGB)."""
    tensor = _preprocess(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        features = _model(tensor)
    embedding = features.squeeze().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def extract_embedding(image_path: str) -> np.ndarray:
    """
    Extract a 2048-d L2-normalized embedding from an image file.
    Uses the original image only (no augmentation) for clean indexing.

    Args:
        image_path: Path to the image file or URL.

    Returns:
        np.ndarray of shape (EMBEDDING_DIM,) — L2-normalized feature vector.
    """
    img = _opencv_to_pil(load_image_from_path_or_url(image_path))
    return _extract_single(img)


def extract_robust_embeddings(image_path: str) -> list[np.ndarray]:
    """
    SUPER-SONIC extraction. ~4 key embeddings for 1-2s results.
    """
    img = _opencv_to_pil(load_image_from_path_or_url(image_path))
    # Resize to ResNet native size range immediately
    img.thumbnail((448, 448))
    w, h = img.size
    embeddings = []

    # 1. Original
    embeddings.append(_extract_single(img))

    # 2. Key Multi-scale (2 best crops)
    cx, cy = w // 2, h // 2
    for ratio in [0.65, 0.9]:
        cw, ch = int(w * ratio), int(h * ratio)
        embeddings.append(_extract_single(img.crop((cx - cw // 2, cy - ch // 2, cx + cw // 2, cy + ch // 2))))

    # 3. High Visibility (Contrast + Sharpness)
    enhanced = ImageEnhance.Contrast(ImageEnhance.Sharpness(img).enhance(1.4)).enhance(1.2)
    embeddings.append(_extract_single(enhanced))

    # 4. Center-Focused Crop (for distant shots)
    cw, ch = int(w * 0.5), int(h * 0.5)
    embeddings.append(_extract_single(img.crop((cx - cw // 2, cy - ch // 2, cx + cw // 2, cy + ch // 2))))

    print(f"Generated {len(embeddings)} super-sonic embeddings")
    return embeddings


def extract_augmented_embeddings(image_path: str) -> list[np.ndarray]:
    """
    Extract multiple augmented embeddings for indexing.
    This creates variations (blur, brightness, contrast, desaturation, crops)
    to make the FAISS index robust against real-world scanning conditions.
    """
    img = _opencv_to_pil(load_image_from_path_or_url(image_path))
    w, h = img.size
    embeddings = []

    # 1. Original
    embeddings.append(_extract_single(img))

    # 2. Slight Blur (simulates out-of-focus camera)
    embeddings.append(_extract_single(img.filter(ImageFilter.GaussianBlur(radius=1.5))))

    # 3. Brightness variations
    embeddings.append(_extract_single(ImageEnhance.Brightness(img).enhance(0.8)))
    embeddings.append(_extract_single(ImageEnhance.Brightness(img).enhance(1.2)))

    # 4. Contrast variations
    embeddings.append(_extract_single(ImageEnhance.Contrast(img).enhance(0.8)))
    embeddings.append(_extract_single(ImageEnhance.Contrast(img).enhance(1.3)))

    # 5. Mild Crop (simulates framing variations)
    cx, cy = w // 2, h // 2
    for ratio in [0.85]:
        cw, ch = int(w * ratio), int(h * ratio)
        embeddings.append(_extract_single(img.crop((cx - cw // 2, cy - ch // 2, cx + cw // 2, cy + ch // 2))))

    # 6. Print-like desaturation
    embeddings.append(_extract_single(ImageEnhance.Color(img).enhance(0.6)))

    # 7. Slight Rotation (simulates camera angle)
    for angle in [-5, 5]:
        rotated = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))
        embeddings.append(_extract_single(rotated))

    # 8. Print Simulation (Low contrast + Grain + Slight Warmth)
    # This simulates how paper absorbs ink and how scanners/cameras see it
    print_sim = ImageEnhance.Contrast(img).enhance(0.7)
    print_sim = ImageEnhance.Color(print_sim).enhance(0.8)
    # Add very fine grain (noise) using filter
    print_sim = print_sim.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    embeddings.append(_extract_single(print_sim))

    print(f"Generated {len(embeddings)} augmented embeddings for indexing")
    return embeddings
