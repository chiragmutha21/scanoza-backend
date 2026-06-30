"""
Blind DCT/QIM watermarking for print-and-scan target identity.

This module embeds a short opaque watermark_id, not user content. The app stores
watermark_id -> contentId in MongoDB and uses the decoded ID to fetch the real
message. The watermark is spread across mid-frequency DCT coefficients in the
luminance channel so the visible image stays effectively unchanged.
"""
from __future__ import annotations

import binascii
import hashlib
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


WATERMARK_VERSION = "dct-qim-v1"
PAYLOAD_HEX_LEN = 16  # 64-bit opaque ID
CRC_BITS = 16
PAYLOAD_BITS = PAYLOAD_HEX_LEN * 4
TOTAL_BITS = PAYLOAD_BITS + CRC_BITS
BLOCK_SIZE = 8
DEFAULT_STRENGTH = 10.0
DEFAULT_REPETITIONS = 28
CANONICAL_MAX_DIM = 768
PILOT_BLOCKS = 1400
SECRET = os.getenv("WATERMARK_SECRET", "scanoza-local-watermark-secret-v1")
COEFF_PAIR = ((3, 2), (2, 3))
PILOT_COEFF_PAIR = ((3, 3), (4, 1))


@dataclass
class WatermarkResult:
    watermark_id: str | None
    confidence: float
    valid_crc: bool
    bit_agreement: float
    message: str = ""


@dataclass
class QualityMetrics:
    psnr: float
    ssim: float


def _bytes_to_bits(data: bytes) -> list[int]:
    bits: list[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def _bits_to_bytes(bits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i + 8]:
            byte = (byte << 1) | int(bit)
        out.append(byte)
    return bytes(out)


def encode_watermark_payload(watermark_id: str) -> list[int]:
    if len(watermark_id) != PAYLOAD_HEX_LEN:
        raise ValueError(f"watermark_id must be {PAYLOAD_HEX_LEN} hex chars")
    payload = bytes.fromhex(watermark_id)
    crc = binascii.crc_hqx(payload, 0xFFFF).to_bytes(2, "big")
    return _bytes_to_bits(payload + crc)


def decode_watermark_payload(bits: list[int]) -> tuple[str | None, bool]:
    if len(bits) != TOTAL_BITS:
        return None, False
    raw = _bits_to_bytes(bits)
    payload, crc = raw[:-2], raw[-2:]
    expected_crc = binascii.crc_hqx(payload, 0xFFFF).to_bytes(2, "big")
    if crc != expected_crc:
        return None, False
    return payload.hex(), True


def _block_positions(height: int, width: int, repetitions: int, seed_tag: str = "global") -> list[list[tuple[int, int]]]:
    blocks_y = height // BLOCK_SIZE
    blocks_x = width // BLOCK_SIZE
    margin = 2
    positions = [
        (by, bx)
        for by in range(margin, max(margin, blocks_y - margin))
        for bx in range(margin, max(margin, blocks_x - margin))
    ]
    required = TOTAL_BITS * repetitions
    if len(positions) < TOTAL_BITS:
        raise ValueError("Image is too small for watermark payload")

    digest = hashlib.sha256(f"{SECRET}:{WATERMARK_VERSION}:{seed_tag}:{height}:{width}".encode()).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    rng.shuffle(positions)

    if len(positions) < required:
        repetitions = max(1, len(positions) // TOTAL_BITS)
        required = TOTAL_BITS * repetitions

    selected = positions[:required]
    return [
        selected[i * repetitions:(i + 1) * repetitions]
        for i in range(TOTAL_BITS)
    ]


def _pilot_positions(height: int, width: int, seed_tag: str, count: int = PILOT_BLOCKS) -> list[tuple[int, int]]:
    blocks_y = height // BLOCK_SIZE
    blocks_x = width // BLOCK_SIZE
    margin = 2
    positions = [
        (by, bx)
        for by in range(margin, max(margin, blocks_y - margin))
        for bx in range(margin, max(margin, blocks_x - margin))
    ]
    digest = hashlib.sha256(f"{SECRET}:{WATERMARK_VERSION}:pilot:{seed_tag}:{height}:{width}".encode()).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    rng.shuffle(positions)
    return positions[:min(count, len(positions))]


def convert_to_rgb_with_white_bg(img: Image.Image) -> Image.Image:
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        alpha = img.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=alpha)
        return bg.convert('RGB')
    return img.convert('RGB')

def _prepare_luma(path: str, target_size: int | None = None) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    from embeddings import load_image_from_path_or_url, _opencv_to_pil
    opencv_img = load_image_from_path_or_url(path)
    img = _opencv_to_pil(opencv_img)
    original_size = img.size
    if target_size:
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    rgb = np.array(img)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    return rgb, ycrcb, original_size


def _quality_metrics(original_rgb: np.ndarray, watermarked_rgb: np.ndarray) -> QualityMetrics:
    a = original_rgb.astype(np.float32)
    b = watermarked_rgb.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    psnr = 99.0 if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))

    gray_a = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(watermarked_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(gray_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gray_b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(gray_a * gray_a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(gray_b * gray_b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(gray_a * gray_b, (11, 11), 1.5) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    )
    return QualityMetrics(psnr=psnr, ssim=float(np.mean(ssim_map)))


def embed_watermark(
    input_path: str,
    output_path: str,
    watermark_id: str,
    strength: float = DEFAULT_STRENGTH,
    repetitions: int = DEFAULT_REPETITIONS,
) -> QualityMetrics:
    bits = encode_watermark_payload(watermark_id)
    original_rgb, ycrcb, _ = _prepare_luma(input_path, target_size=CANONICAL_MAX_DIM)
    height, width = ycrcb.shape[:2]
    height -= height % BLOCK_SIZE
    width -= width % BLOCK_SIZE
    ycrcb = ycrcb[:height, :width, :]
    original_rgb = original_rgb[:height, :width, :]

    y = ycrcb[:, :, 0].copy()
    bit_positions = _block_positions(height, width, repetitions)
    c_a, c_b = COEFF_PAIR

    def apply_bits(positions_by_bit: list[list[tuple[int, int]]], layer_strength: float):
        for bit, positions in zip(bits, positions_by_bit):
            for by, bx in positions:
                y0 = by * BLOCK_SIZE
                x0 = bx * BLOCK_SIZE
                block = y[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE]
                dct_block = cv2.dct(block - 128.0)

                a = float(dct_block[c_a])
                b = float(dct_block[c_b])
                diff = a - b
                target_sign = 1.0 if bit == 1 else -1.0
                if diff * target_sign < layer_strength:
                    adjustment = (layer_strength - diff * target_sign) / 2.0
                    dct_block[c_a] = a + target_sign * adjustment
                    dct_block[c_b] = b - target_sign * adjustment

                y[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE] = cv2.idct(dct_block) + 128.0

    apply_bits(bit_positions, strength)
    # Candidate-specific pilot for assisted verification after ORB/AI shortlist.
    pilot_strength = strength * 0.75
    p_a, p_b = PILOT_COEFF_PAIR
    for by, bx in _pilot_positions(height, width, watermark_id):
        y0 = by * BLOCK_SIZE
        x0 = bx * BLOCK_SIZE
        block = y[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE]
        dct_block = cv2.dct(block - 128.0)
        a = float(dct_block[p_a])
        b = float(dct_block[p_b])
        diff = a - b
        if diff < pilot_strength:
            adjustment = (pilot_strength - diff) / 2.0
            dct_block[p_a] = a + adjustment
            dct_block[p_b] = b - adjustment
        y[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE] = cv2.idct(dct_block) + 128.0

    ycrcb[:, :, 0] = np.clip(y, 0, 255)
    watermarked_rgb = cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    Image.fromarray(watermarked_rgb).save(output_path, "PNG", optimize=True)
    return _quality_metrics(original_rgb, watermarked_rgb)


def _decode_from_array(rgb: np.ndarray, repetitions: int = DEFAULT_REPETITIONS) -> WatermarkResult:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    height, width = ycrcb.shape[:2]
    height -= height % BLOCK_SIZE
    width -= width % BLOCK_SIZE
    if height < 64 or width < 64:
        return WatermarkResult(None, 0.0, False, 0.0, "image too small")

    y = ycrcb[:height, :width, 0]
    c_a, c_b = COEFF_PAIR
    try:
        bit_positions = _block_positions(height, width, repetitions)
    except ValueError as exc:
        return WatermarkResult(None, 0.0, False, 0.0, str(exc))

    decoded_bits: list[int] = []
    agreements: list[float] = []
    margins: list[float] = []

    for positions in bit_positions:
        votes = 0
        abs_margin = 0.0
        for by, bx in positions:
            y0 = by * BLOCK_SIZE
            x0 = bx * BLOCK_SIZE
            block = y[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE]
            dct_block = cv2.dct(block - 128.0)
            diff = float(dct_block[c_a] - dct_block[c_b])
            votes += 1 if diff >= 0 else -1
            abs_margin += abs(diff)

        bit = 1 if votes >= 0 else 0
        decoded_bits.append(bit)
        agreements.append(abs(votes) / max(len(positions), 1))
        margins.append(abs_margin / max(len(positions), 1))

    watermark_id, valid_crc = decode_watermark_payload(decoded_bits)
    bit_agreement = float(np.mean(agreements)) if agreements else 0.0
    margin_score = min(1.0, float(np.mean(margins)) / max(DEFAULT_STRENGTH, 1.0)) if margins else 0.0
    confidence = (bit_agreement * 0.75) + (margin_score * 0.25)
    if not valid_crc:
        watermark_id = None
    return WatermarkResult(watermark_id, confidence, valid_crc, bit_agreement)


def _read_bit_votes(
    rgb: np.ndarray,
    repetitions: int = DEFAULT_REPETITIONS,
    seed_tag: str = "global",
) -> tuple[list[int], list[float], list[float]]:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    height, width = ycrcb.shape[:2]
    height -= height % BLOCK_SIZE
    width -= width % BLOCK_SIZE
    if height < 64 or width < 64:
        return [], []

    y = ycrcb[:height, :width, 0]
    c_a, c_b = COEFF_PAIR
    try:
        bit_positions = _block_positions(height, width, repetitions, seed_tag=seed_tag)
    except ValueError:
        return [], []

    decoded_bits: list[int] = []
    agreements: list[float] = []
    signed_votes: list[float] = []
    for positions in bit_positions:
        votes = 0
        for by, bx in positions:
            y0 = by * BLOCK_SIZE
            x0 = bx * BLOCK_SIZE
            block = y[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE]
            dct_block = cv2.dct(block - 128.0)
            diff = float(dct_block[c_a] - dct_block[c_b])
            votes += 1 if diff >= 0 else -1
        decoded_bits.append(1 if votes >= 0 else 0)
        normalized_vote = votes / max(len(positions), 1)
        agreements.append(abs(normalized_vote))
        signed_votes.append(normalized_vote)
    return decoded_bits, agreements, signed_votes


def extract_watermark(path: str, repetitions: int = DEFAULT_REPETITIONS) -> WatermarkResult:
    from embeddings import load_image_from_path_or_url, _opencv_to_pil
    opencv_img = load_image_from_path_or_url(path)
    img = _opencv_to_pil(opencv_img)
    rgb = np.array(img)
    candidates: list[WatermarkResult] = []

    # Try a small set of normalizations. Real print-scan alignment is handled
    # mainly by ORB/AI fallback; these variants cover common resize/rotation.
    variants = [rgb]
    if max(rgb.shape[:2]) != CANONICAL_MAX_DIM:
        scale = CANONICAL_MAX_DIM / max(rgb.shape[:2])
        variants.append(cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
    for angle in (-2, 2):
        h, w = rgb.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        variants.append(cv2.warpAffine(rgb, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE))

    for variant in variants:
        max_dim = max(variant.shape[:2])
        if max_dim > 1400:
            scale = 1400.0 / max_dim
            variant = cv2.resize(variant, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        candidates.append(_decode_from_array(variant, repetitions))

    valid = [c for c in candidates if c.valid_crc and c.watermark_id]
    if valid:
        return max(valid, key=lambda c: c.confidence)
    return max(candidates, key=lambda c: c.confidence) if candidates else WatermarkResult(None, 0.0, False, 0.0)


def expected_watermark_score(path: str, watermark_id: str, repetitions: int = DEFAULT_REPETITIONS) -> float:
    """Candidate-assisted score for a known watermark ID.

    This is used after ORB/AI has shortlisted likely targets. It is more useful
    under print-scan damage than blind CRC decode because it asks: "does this
    scan look more like candidate A's watermark or candidate B's watermark?"
    """
    from embeddings import load_image_from_path_or_url, _opencv_to_pil
    opencv_img = load_image_from_path_or_url(path)
    img = _opencv_to_pil(opencv_img)
    rgb = np.array(img)
    variants = [rgb]
    if max(rgb.shape[:2]) != CANONICAL_MAX_DIM:
        scale = CANONICAL_MAX_DIM / max(rgb.shape[:2])
        variants.append(cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))

    best = 0.0
    for variant in variants:
        height, width = variant.shape[:2]
        height -= height % BLOCK_SIZE
        width -= width % BLOCK_SIZE
        if height < 64 or width < 64:
            continue
        ycrcb = cv2.cvtColor(variant[:height, :width, :], cv2.COLOR_RGB2YCrCb).astype(np.float32)
        y = ycrcb[:, :, 0]
        c_a, c_b = PILOT_COEFF_PAIR
        votes = []
        for by, bx in _pilot_positions(height, width, watermark_id):
            y0 = by * BLOCK_SIZE
            x0 = bx * BLOCK_SIZE
            block = y[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE]
            dct_block = cv2.dct(block - 128.0)
            votes.append(float(dct_block[c_a] - dct_block[c_b]))
        if not votes:
            continue
        mean_vote = float(np.mean(votes))
        positive_ratio = float(np.mean([1.0 if v > 0 else 0.0 for v in votes]))
        confidence = (positive_ratio * 0.75) + (min(1.0, max(0.0, mean_vote / DEFAULT_STRENGTH)) * 0.25)
        best = max(best, confidence)
    return best
