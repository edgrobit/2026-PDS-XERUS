"""
Feature_penmark_mask.py — Pen mark detection and inpainting
============================================================
Detects blue and black pen marks on skin images using adaptive
thresholding based on skin tone, then inpaints them out.

Two detection modes selected automatically per image:
  - Dark skin  (gray mean < 120): HSV hue + absolute darkness threshold
  - Light skin (gray mean >= 120): adaptive B-R channel shift + darkness

Saves cleaned PNGs (pen marks removed) to data/imgs_clean/.
Images without pen marks are copied as-is so imgs_clean/ is complete.

Usage:
    python Feature_penmark_mask.py   # run standalone
    from src.Feature_penmark_mask import run  # called by main.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
IMG_DIR         = Path("data/imgs")
MASK_DIR        = Path("data/masks")
IMGS_CLEAN_DIR  = Path("data/imgs_clean")   # output: cleaned images
ANNOTATIONS_CSV = Path("data/annotations_combined.csv")

DARK_SKIN_THRESHOLD = 120


# ── Pen detection helpers ─────────────────────────────────────────────────────

def remove_lesion(img, lesion_mask):
    _, lm = cv2.threshold(lesion_mask, 127, 255, cv2.THRESH_BINARY)
    median_color = np.median(img[lm == 0], axis=0).astype(np.uint8)
    clean = img.copy()
    clean[lm == 255] = median_color
    return clean


def skin_gray_mean(img, lesion_mask):
    _, lm = cv2.threshold(lesion_mask, 127, 255, cv2.THRESH_BINARY)
    px = img[lm == 0].astype(float)
    b, g, r = px[:, 0], px[:, 1], px[:, 2]
    return (0.114 * b + 0.587 * g + 0.299 * r).mean()


def directional_dilation(mask):
    """
    Dilate in 4 directions and keep pixels agreed on by at least 2.
    Follows pen stroke trends without bloating uniformly.
    """
    h_kernel = np.ones((1, 15), np.uint8)
    v_kernel = np.ones((15, 1), np.uint8)
    d1       = np.eye(11, dtype=np.uint8)
    d2       = np.fliplr(d1)

    dil_h  = cv2.dilate(mask, h_kernel, iterations=1)
    dil_v  = cv2.dilate(mask, v_kernel, iterations=1)
    dil_d1 = cv2.dilate(mask, d1,       iterations=1)
    dil_d2 = cv2.dilate(mask, d2,       iterations=1)

    vote = (dil_h.astype(int) + dil_v.astype(int) +
            dil_d1.astype(int) + dil_d2.astype(int)) // 255
    return (vote >= 2).astype(np.uint8) * 255


def detect_pen_dark_skin(img, lm):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]

    dark_pen = (gray.astype(int) < 90) & (s < 110)
    s_thresh = np.percentile(s[lm == 0], 65)
    hue_pen  = (h >= 95) & (h <= 175) & (s.astype(int) > s_thresh)

    pen = (dark_pen | hue_pen).astype(np.uint8) * 255
    pen[lm == 255] = 0
    return pen


def detect_pen_light_skin(img, lm):
    b_ch = img[:, :, 0].astype(float)
    g_ch = img[:, :, 1].astype(float)
    r_ch = img[:, :, 2].astype(float)
    gray = 0.114 * b_ch + 0.587 * g_ch + 0.299 * r_ch

    skin  = lm == 0
    sg    = gray[skin]
    gmed  = np.median(sg)
    gstd  = max(np.median(np.abs(sg - gmed)) * 1.4826, 8.0)

    br     = b_ch[skin] - r_ch[skin]
    br_med = np.median(br)
    br_std = max(np.median(np.abs(br - br_med)) * 1.4826, 5.0)

    dark_pen = gray < (gmed - 1.6 * gstd)
    blue_pen = (b_ch - r_ch) > (br_med + 2.0 * br_std)

    pen = (dark_pen | blue_pen).astype(np.uint8) * 255
    pen[lm == 255] = 0
    return pen


def generate_pen_mask(img, lesion_mask):
    """
    Full pipeline: lesion fill → adaptive pen detection → directional dilation.
    Returns binary pen mask (255 = pen, 0 = clean).
    """
    lesion_mask = cv2.resize(lesion_mask, (img.shape[1], img.shape[0]))
    _, lm = cv2.threshold(lesion_mask, 127, 255, cv2.THRESH_BINARY)

    clean_img = remove_lesion(img, lm)
    gm        = skin_gray_mean(img, lm)

    if gm < DARK_SKIN_THRESHOLD:
        pen = detect_pen_dark_skin(clean_img, lm)
    else:
        pen = detect_pen_light_skin(clean_img, lm)

    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    pen = cv2.morphologyEx(pen, cv2.MORPH_OPEN,  k3)
    pen = cv2.morphologyEx(pen, cv2.MORPH_CLOSE, k5)
    pen = directional_dilation(pen)
    pen = cv2.morphologyEx(pen, cv2.MORPH_CLOSE, k5)
    pen[lm == 255] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pen)
    cleaned = np.zeros_like(pen)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 80:
            cleaned[labels == i] = 255

    return cleaned


# ── Main run function ─────────────────────────────────────────────────────────

def run(img_dir=IMG_DIR, mask_dir=MASK_DIR,
        annotations_csv=ANNOTATIONS_CSV, output_dir=IMGS_CLEAN_DIR):
    """
    For images WITH pen marks: inpaint and save to output_dir.
    For images WITHOUT pen marks: copy original to output_dir.
    This ensures output_dir is a complete set of clean images
    that feature_skincolour.py can read from directly.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations to identify pen mark images
    df       = pd.read_csv(annotations_csv)
    pen_cols = [c for c in ["pen_1", "pen_2", "pen_3", "pen_4"] if c in df.columns]
    has_pen  = df[pen_cols].eq(1).sum(axis=1) >= 2
    pen_ids  = set(df[has_pen]["img_id"].str.replace(".png", "", regex=False).tolist())

    print(f"Images with pen marks: {len(pen_ids)}")

    # Load blocklist — masks with no corresponding image
    from utils import load_masks_blocklist
    blocklist = load_masks_blocklist()
    print(f"Blocklist loaded:      {len(blocklist)} images to skip")

    # Get all images
    valid_ext   = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    all_images  = [f for f in sorted(Path(img_dir).iterdir())
                   if f.suffix.lower() in valid_ext]

    print(f"Total images found:    {len(all_images)}")

    processed = skipped = copied = 0

    for img_path in all_images:
        if img_path.stem in blocklist:
            continue
        stem     = img_path.stem
        out_path = output_dir / img_path.name

        # Images without pen marks — copy directly
        if stem not in pen_ids:
            import shutil
            shutil.copy2(str(img_path), str(out_path))
            copied += 1
            continue

        # Images with pen marks — inpaint
        mask_path = Path(mask_dir) / f"{stem}_mask.png"

        if not mask_path.exists():
            print(f"  [MISSING MASK] {stem} — copying original")
            import shutil
            shutil.copy2(str(img_path), str(out_path))
            copied += 1
            continue

        img         = cv2.imread(str(img_path))
        lesion_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or lesion_mask is None:
            print(f"  [UNREADABLE] {stem}")
            skipped += 1
            continue

        try:
            pen_mask = generate_pen_mask(img, lesion_mask)
            result   = cv2.inpaint(img, pen_mask, inpaintRadius=7,
                                   flags=cv2.INPAINT_TELEA)
            cv2.imwrite(str(out_path), result)
            processed += 1
        except Exception as e:
            print(f"  [ERROR] {stem}: {e}")
            skipped += 1

    print(f"\nPen removal done:")
    print(f"  Inpainted: {processed} | Copied as-is: {copied} | Skipped: {skipped}")
    print(f"  Clean images saved to: {output_dir}")


if __name__ == "__main__":
    run()
