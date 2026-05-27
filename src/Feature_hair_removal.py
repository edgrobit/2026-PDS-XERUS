"""
hair_removal.py — Hair detection and removal
=============================================
Removes hair from skin lesion images using morphological filtering.

Pipeline per image:
  1. CLAHE enhancement on green channel for better hair visibility
  2. Sobel (thick hairs) + Laplacian (thin hairs) edge detection, averaged
  3. Adaptive thresholding → binary hair mask
  4. Morphological closing to reduce artifacts
  5. Inpainting to fill hair pixels with surrounding skin texture

Kernel size is adjusted dynamically based on hair coverage:
  coverage < 0.005        → skip (no meaningful hair)
  0.005 <= coverage <= 0.035 → kernel size 15
  coverage > 0.035        → kernel size 25

Skips images listed in data/masks_without_images.csv (no corresponding mask).

Usage:
    python hair_removal.py
    from src.hair_removal import run_dataset_pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import os
import glob
from utils import load_masks_blocklist


# ── Algorithm functions ───────────────────────────────────────────────────────

def get_valid_area_mask(img_gray):
    """
    Creates a mask of the actual skin/lesion area, ignoring black
    vignette corners often found in dermoscopy images.
    """
    _, mask = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
    return mask


def calculate_hair_coverage(img_bgr):
    """
    Estimates hair coverage using CLAHE, Sobel, and Laplacian edge detection.
    Returns coverage ratio (0-1).
    """
    base_channel = img_bgr[:, :, 1]   # green channel — best contrast for skin
    img_gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    valid_mask   = get_valid_area_mask(img_gray)

    # Boost visibility with CLAHE
    clahe         = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_base = clahe.apply(base_channel)

    # Sobel — highlights thick hairs (strong directional edges)
    sobel_x    = cv2.Sobel(enhanced_base, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y    = cv2.Sobel(enhanced_base, cv2.CV_64F, 0, 1, ksize=3)
    sobel_norm = cv2.normalize(cv2.magnitude(sobel_x, sobel_y),
                               None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Laplacian — highlights thin hair-like structures
    lap_norm = cv2.normalize(np.absolute(cv2.Laplacian(enhanced_base, cv2.CV_64F)),
                             None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    averaged_edges = cv2.addWeighted(sobel_norm, 0.5, lap_norm, 0.5, 0)

    blurred = cv2.GaussianBlur(averaged_edges, (3, 3), 0)
    _, binary_hair_mask = cv2.threshold(blurred, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hair_pixels  = cv2.countNonZero(cv2.bitwise_and(binary_hair_mask, valid_mask))
    valid_pixels = cv2.countNonZero(valid_mask)

    return 0.0 if valid_pixels == 0 else hair_pixels / valid_pixels


def process_image(img_bgr):
    """
    Remove hair from a single image.
    Returns (processed_image, coverage, was_processed).
    """
    coverage = calculate_hair_coverage(img_bgr)

    if coverage < 0.005:
        return img_bgr, coverage, False

    k_size = 15 if coverage <= 0.035 else 25
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_size, k_size))

    base_channel = img_bgr[:, :, 1]

    # Combine BlackHat (dark hair) and TopHat (light hair) for mixed hair types
    blackhat     = cv2.morphologyEx(base_channel, cv2.MORPH_BLACKHAT, kernel)
    tophat       = cv2.morphologyEx(base_channel, cv2.MORPH_TOPHAT,   kernel)
    combined     = cv2.max(blackhat, tophat)

    # Adaptive threshold handles non-uniform lighting better than global
    block_size   = (k_size * 2) + 1
    removal_mask = cv2.adaptiveThreshold(
        combined, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, -5
    )

    # Closing to reduce leftover artifacts
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    removal_mask   = cv2.morphologyEx(removal_mask, cv2.MORPH_CLOSE, closing_kernel)

    cleaned = cv2.inpaint(img_bgr, removal_mask, inpaintRadius=3,
                          flags=cv2.INPAINT_TELEA)

    return cleaned, coverage, True


# ── Batch pipeline ────────────────────────────────────────────────────────────

def run_dataset_pipeline(input_dir, output_dir):
    """
    Process all PNG images in input_dir and save results to output_dir.
    Images in the blocklist (no corresponding mask) are skipped entirely.
    """
    blocklist    = load_masks_blocklist()
    image_files  = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    if not image_files:
        print(f"No PNG images found in '{input_dir}'. Check the path.")
        return

    os.makedirs(output_dir, exist_ok=True)

    total      = len(image_files)
    processed  = skipped_hair = skipped_block = errors = 0

    print(f"Found {total} images | Blocklist: {len(blocklist)} to skip\n")

    for i, filepath in enumerate(image_files, 1):
        filename = os.path.basename(filepath)
        stem     = os.path.splitext(filename)[0]

        # Skip images with no corresponding mask
        if stem in blocklist:
            skipped_block += 1
            continue

        img = cv2.imread(filepath)
        if img is None:
            print(f"  [ERROR] Could not read: {filename}")
            errors += 1
            continue

        final_img, coverage_val, was_processed = process_image(img)

        if was_processed:
            processed += 1
            status = "Cleaned"
        else:
            skipped_hair += 1
            status = "No hair"

        cv2.imwrite(os.path.join(output_dir, filename), final_img)

        if i % 100 == 0 or i == total:
            print(f"  [{i}/{total}] {filename} | coverage={coverage_val:.4f} | {status}")

    print("\n" + "="*40)
    print("HAIR REMOVAL COMPLETE")
    print(f"  Hair removed:    {processed}")
    print(f"  No hair (kept):  {skipped_hair}")
    print(f"  Blocklisted:     {skipped_block}")
    print(f"  Errors:          {errors}")
    print(f"  Output:          {output_dir}")
    print("="*40)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_dataset_pipeline(
        input_dir  = "data/imgs",
        output_dir = "data/imgs_hairless",
    )
