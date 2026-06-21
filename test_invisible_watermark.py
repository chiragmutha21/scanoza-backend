import os
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw

from invisible_watermark import embed_watermark, expected_watermark_score, extract_watermark


class InvisibleWatermarkTests(unittest.TestCase):
    def _make_source(self, path: str):
        img = Image.new("RGB", (768, 768), "white")
        draw = ImageDraw.Draw(img)
        for i in range(0, 768, 24):
            color = (30 + (i % 160), 80, 180)
            draw.line((0, i, 767, 767 - i), fill=color, width=3)
            draw.rectangle((i // 2, i // 3, i // 2 + 80, i // 3 + 60), outline=(20, 20, 20), width=2)
        draw.text((80, 350), "SCANOZA WATERMARK TEST", fill=(10, 10, 10))
        img.save(path, "PNG")

    def test_same_image_can_hold_different_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.png")
            out_a = os.path.join(tmp, "a.png")
            out_b = os.path.join(tmp, "b.png")
            self._make_source(src)

            embed_watermark(src, out_a, "0123456789abcdef")
            embed_watermark(src, out_b, "fedcba9876543210")

            self.assertEqual(extract_watermark(out_a).watermark_id, "0123456789abcdef")
            self.assertEqual(extract_watermark(out_b).watermark_id, "fedcba9876543210")

    def test_survives_common_digital_distortions(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.png")
            out = os.path.join(tmp, "wm.png")
            distorted = os.path.join(tmp, "distorted.jpg")
            self._make_source(src)

            embed_watermark(src, out, "1111222233334444")
            img = cv2.imread(out)
            img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
            img = cv2.convertScaleAbs(img, alpha=1.08, beta=8)
            noise = np.random.default_rng(7).normal(0, 1.5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(distorted, img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

            result = extract_watermark(distorted)
            if result.watermark_id is not None:
                self.assertEqual(result.watermark_id, "1111222233334444")
            correct = expected_watermark_score(distorted, "1111222233334444")
            wrong = expected_watermark_score(distorted, "9999888877776666")
            self.assertGreater(correct, wrong)
            self.assertGreater(correct, 0.24)


if __name__ == "__main__":
    unittest.main()
