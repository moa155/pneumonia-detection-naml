"""Preprocess RSNA DICOM files to PNG for faster training.

DICOM loading is ~10-50x slower than PNG loading.  This script converts
all training DICOM images to 8-bit PNG files, enabling much faster
data loading during training.

Usage:
    python -m src.preprocess --data-dir data/
    python -m src.preprocess --data-dir data/ --compress 1   # faster, slightly larger
"""

import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pydicom

# Prefer OpenCV (faster PNG I/O) but fall back to PIL
try:
    import cv2
    _USE_CV2 = True
except ImportError:
    from PIL import Image
    _USE_CV2 = False

# Global set by initializer so each worker knows the compression level
_COMPRESS_LEVEL = 1


def _init_worker(compress_level: int):
    global _COMPRESS_LEVEL
    _COMPRESS_LEVEL = compress_level


def convert_one(args):
    dcm_path, out_path = args
    try:
        dcm = pydicom.dcmread(str(dcm_path), force=True)
        pixel_array = dcm.pixel_array.astype(np.float32)
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax - pmin > 0:
            pixel_array = (pixel_array - pmin) / (pmax - pmin)
        pixel_array = (pixel_array * 255).astype(np.uint8)

        if _USE_CV2:
            cv2.imwrite(str(out_path), pixel_array,
                        [cv2.IMWRITE_PNG_COMPRESSION, _COMPRESS_LEVEL])
        else:
            Image.fromarray(pixel_array).save(
                str(out_path), compress_level=_COMPRESS_LEVEL)
        return True
    except Exception as e:
        print(f"Error converting {dcm_path}: {e}")
        return False


def preprocess(data_dir: str, compress_level: int = 1):
    data_dir = Path(data_dir)
    dcm_dir = data_dir / "stage_2_train_images"
    png_dir = data_dir / "stage_2_train_images_png"
    png_dir.mkdir(exist_ok=True)

    dcm_files = sorted(dcm_dir.glob("*.dcm"))
    print(f"Found {len(dcm_files)} DICOM files")
    print(f"Backend: {'OpenCV' if _USE_CV2 else 'PIL'}, compression level: {compress_level}")

    # Skip already converted
    tasks = []
    for dcm_path in dcm_files:
        out_path = png_dir / (dcm_path.stem + ".png")
        if not out_path.exists():
            tasks.append((dcm_path, out_path))

    if not tasks:
        print("All files already converted.")
        return

    print(f"Converting {len(tasks)} files to PNG (skipping {len(dcm_files) - len(tasks)} already done)...")
    n_workers = max(1, cpu_count() - 1)

    import sys
    with Pool(n_workers, initializer=_init_worker, initargs=(compress_level,)) as pool:
        results = []
        done = 0
        log_every = max(1, len(tasks) // 20)  # Print ~20 progress lines
        for result in pool.imap_unordered(convert_one, tasks, chunksize=32):
            results.append(result)
            done += 1
            if done % log_every == 0 or done == len(tasks):
                print(f"  DICOM -> PNG: {done}/{len(tasks)} ({100*done//len(tasks)}%)", flush=True)

    success = sum(results)
    print(f"Done: {success}/{len(tasks)} converted successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DICOM to PNG")
    parser.add_argument("--data-dir", default="data", help="Path to RSNA data directory")
    parser.add_argument("--compress", type=int, default=1, choices=range(0, 10),
                        help="PNG compression level: 0=none (fastest), 1=fast (default), 9=max")
    args = parser.parse_args()
    preprocess(args.data_dir, compress_level=args.compress)
