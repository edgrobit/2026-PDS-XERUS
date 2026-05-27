"""
feature_skincolour.py — Skin color feature extraction
======================================================
Extracts color features from the skin region of each image.
Reads from data/imgs_clean/ if available (pen marks removed),
falls back to data/imgs/ otherwise.
Returns a dataframe — does not save to disk (main.py builds the master CSV).

Output columns:
    image_name      — filename
    rgb_var_r/g/b   — RGB variance across superpixels
    hsv_var_h/s/v   — HSV variance (hue uses circular variance)
    ita_mean        — Individual Typology Angle (continuous skin tone)
    fst_predicted   — Fitzpatrick Skin Type 1-6, derived from ITA
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from skimage.segmentation import slic
from skimage.color import rgb2hsv, rgb2lab
from scipy.stats import circvar
from utils import (find_project_paths, load_image, load_mask,
                   find_mask_for_image, VALID_EXTENSIONS)

PATHS          = find_project_paths()
IMG_DIR        = PATHS["img_dir"]
IMGS_CLEAN_DIR = PATHS["project_root"] / "data" / "imgs_clean"
MASK_DIR       = PATHS["mask_dir"]

MAX_SIZE   = 512
N_SEGMENTS = 50
ITA_THRESHOLDS = [55.0, 41.0, 28.0, 10.0, -30.0]


def ita_to_fst(ita):
    """Convert ITA angle to Fitzpatrick Skin Type 1-6."""
    for fst, threshold in enumerate(ITA_THRESHOLDS, start=1):
        if ita > threshold:
            return fst
    return 6


def compute_ita(image, segments):
    """
    Compute Individual Typology Angle from superpixels in CIELab space.
    ITA = arctan((L-50)/b) * (180/pi)
    Only skin-brightness superpixels (L 30-90) used to exclude lesion.
    Returns (ita_mean, fst_predicted).
    """
    lab      = rgb2lab(image)
    seg_ids  = np.unique(segments)
    n        = len(seg_ids)
    lut      = np.zeros(segments.max() + 1, dtype=np.intp)
    lut[seg_ids] = np.arange(n)
    flat_idx = lut[segments.ravel()]
    counts   = np.bincount(flat_idx, minlength=n).astype(np.float64)
    lab_flat = lab.reshape(-1, 3)
    lab_sum  = np.zeros((n, 3))
    np.add.at(lab_sum, flat_idx, lab_flat)
    lab_means = lab_sum / counts[:, None]

    L, b = lab_means[:, 0], lab_means[:, 2]
    skin_mask = (L >= 30) & (L <= 90)
    if skin_mask.sum() == 0:
        skin_mask = np.ones(n, dtype=bool)

    b_safe   = np.where(np.abs(b[skin_mask]) < 1e-6, 1e-6, b[skin_mask])
    ita_vals = np.degrees(np.arctan((L[skin_mask] - 50) / b_safe))
    ita_mean = float(np.mean(ita_vals))
    return round(ita_mean, 4), ita_to_fst(ita_mean)


def get_segment_means(image, hsv, segments):
    """Compute mean RGB and HSV per superpixel. Hue uses circular mean."""
    seg_ids  = np.unique(segments)
    n        = len(seg_ids)
    lut      = np.zeros(segments.max() + 1, dtype=np.intp)
    lut[seg_ids] = np.arange(n)
    flat_idx = lut[segments.ravel()]
    counts   = np.bincount(flat_idx, minlength=n).astype(np.float64)

    rgb_flat  = image.reshape(-1, 3)
    rgb_sum   = np.zeros((n, 3))
    np.add.at(rgb_sum, flat_idx, rgb_flat)
    rgb_means = rgb_sum / counts[:, None]

    hsv_flat = hsv.reshape(-1, 3)
    sv_sum   = np.zeros((n, 2))
    np.add.at(sv_sum, flat_idx, hsv_flat[:, 1:])
    sv_means = sv_sum / counts[:, None]

    angles  = hsv_flat[:, 0] * 2 * np.pi
    sin_sum = np.zeros(n)
    cos_sum = np.zeros(n)
    np.add.at(sin_sum, flat_idx, np.sin(angles))
    np.add.at(cos_sum, flat_idx, np.cos(angles))
    h_means = (np.arctan2(sin_sum/counts, cos_sum/counts) % (2*np.pi)) / (2*np.pi)

    return rgb_means, np.column_stack([h_means, sv_means])


def extract_color_features(image):
    hsv      = rgb2hsv(image)
    segments = slic(image, n_segments=N_SEGMENTS, compactness=10, start_label=1)
    rgb_means, hsv_means = get_segment_means(image, hsv, segments)

    rgb_var = tuple(np.var(rgb_means, axis=0)) if len(rgb_means) >= 2 else (0., 0., 0.)
    hsv_var = (
        circvar(hsv_means[:, 0], high=1, low=0),
        np.var(hsv_means[:, 1]),
        np.var(hsv_means[:, 2]),
    ) if len(hsv_means) >= 2 else (0., 0., 0.)

    ita_mean, fst_predicted = compute_ita(image, segments)

    return {
        "rgb_var_r":     rgb_var[0],
        "rgb_var_g":     rgb_var[1],
        "rgb_var_b":     rgb_var[2],
        "hsv_var_h":     hsv_var[0],
        "hsv_var_s":     hsv_var[1],
        "hsv_var_v":     hsv_var[2],
        "ita_mean":      ita_mean,
        "fst_predicted": fst_predicted,
    }


def run():
    # Use imgs_clean/ if it exists and has files, otherwise fall back to imgs/
    if IMGS_CLEAN_DIR.exists() and any(IMGS_CLEAN_DIR.iterdir()):
        source_dir = IMGS_CLEAN_DIR
        print(f"  Using cleaned images: {source_dir}")
    else:
        source_dir = IMG_DIR
        print(f"  Using original images: {source_dir}")

    image_files = [
        f for f in sorted(source_dir.iterdir())
        if f.suffix.lower() in VALID_EXTENSIONS
    ]
    print(f"  Images found: {len(image_files)}")

    rows = []
    for image_path in image_files:
        mask_path = find_mask_for_image(image_path, MASK_DIR)
        if mask_path is None:
            mask_path = find_mask_for_image(IMG_DIR / image_path.name, MASK_DIR)
        if mask_path is None:
            continue

        try:
            image    = load_image(image_path, max_size=MAX_SIZE)
            mask_bin = load_mask(mask_path)
            if mask_bin.sum() < 50:
                continue

            rows.append({
                "image_name": image_path.name,
                **extract_color_features(image),
            })
        except Exception as e:
            print(f"  [ERROR] {image_path.name}: {e}")

    df = pd.DataFrame(rows).sort_values("image_name").reset_index(drop=True)
    print(f"  Extracted {len(df)} rows")
    return df


if __name__ == "__main__":
    print(run().head())
