from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from .config import IMAGE_EXTENSIONS


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def iter_bucket_images(bucket_dir: Path) -> list[Path]:
    files = [p for p in bucket_dir.rglob("*") if p.is_file() and is_image_file(p)]
    files.sort()
    return files


def paired_txt_path(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def build_stable_output_stem(image_path: Path, bucket_dir: Path) -> str:
    rel = image_path.relative_to(bucket_dir)
    parent = "__".join(rel.parts[:-1])
    stem = image_path.stem
    if parent:
        return f"{parent}__{stem}"
    return stem


def copy_text_pair(src_image: Path, dst_image: Path) -> None:
    src_txt = paired_txt_path(src_image)
    if src_txt.exists():
        dst_txt = dst_image.with_suffix(".txt")
        shutil.copy2(src_txt, dst_txt)


def copy_image_and_text(src_image: Path, dst_image: Path) -> None:
    ensure_dir(dst_image.parent)
    shutil.copy2(src_image, dst_image)
    copy_text_pair(src_image, dst_image)


def read_image_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError("empty image file")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode image")
    return image


def write_image_bgr(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    ext = path.suffix.lower()
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"failed to encode image with extension {ext}")
    encoded.tofile(str(path))


def image_resolution_str(image: np.ndarray) -> str:
    h, w = image.shape[:2]
    return f"{w}x{h}"


def short_side(image: np.ndarray) -> int:
    h, w = image.shape[:2]
    return min(h, w)


def estimate_small_face_present(image_bgr: np.ndarray, threshold_px: int = 128) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade_file = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_file.exists():
        return False

    face_cascade = cv2.CascadeClassifier(str(cascade_file))
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(24, 24),
    )
    if len(faces) == 0:
        return False

    for (_x, _y, w, h) in faces:
        if max(int(w), int(h)) < threshold_px:
            return True
    return False


def sample_items(items: list[str], k: int, seed: int) -> list[str]:
    if len(items) <= k:
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, k)
