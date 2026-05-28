import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from math import floor, ceil
from skimage.transform import rotate
from utils import (find_project_paths, load_mask, VALID_EXTENSIONS,load_masks_blocklist)

PATHS= find_project_paths()
MASK_DIR= PATHS["mask_dir"]
N_ROTATIONS = 8


def midpoint(mask):
    row_summed= np.sum(mask, axis=1)
    col_summed= np.sum(mask, axis=0)
    half_row= np.sum(row_summed) / 2
    half_col= np.sum(col_summed) / 2
    row_mid= next(i for i, n in enumerate(np.add.accumulate(row_summed)) if n > half_row)
    col_mid= next(i for i, n in enumerate(np.add.accumulate(col_summed)) if n > half_col)
    return row_mid, col_mid


def trim_to_bounding_box(mask):
    col_sums= np.sum(mask, axis=0)
    row_sums= np.sum(mask, axis=1)
    active_cols= [i for i, v in enumerate(col_sums) if v != 0]
    active_rows= [i for i, v in enumerate(row_sums) if v != 0]
    if not active_rows or not active_cols:
        return None
    return mask[active_rows[0]:active_rows[-1]+1, active_cols[0]:active_cols[-1]+1]


def asymmetry_score(mask):
    row_mid, col_mid= midpoint(mask)
    upper= mask[:ceil(row_mid), :]
    lower= mask[floor(row_mid):, :]
    left= mask[:, :ceil(col_mid)]
    right= mask[:, floor(col_mid):]

    fl= np.flip(lower, axis=0)
    fr= np.flip(right, axis=1)
    min_r = min(upper.shape[0], fl.shape[0])
    min_c = min(left.shape[1],  fr.shape[1])

    hxor= np.logical_xor(upper[:min_r, :], fl[:min_r, :])
    vxor= np.logical_xor(left[:, :min_c], fr[:, :min_c])
    total= np.sum(mask)

    if total == 0:
        return None
    return round(float((np.sum(hxor) + np.sum(vxor)) / (total * 2)), 4)


def feature_asymmetry(mask, n=N_ROTATIONS):
    scores = []
    for i in range(n):
        degrees= 90 * i / n
        rotated= rotate(mask.astype(np.float64), degrees, cval=0.0)
        binarized= (rotated > 0.5).astype(np.uint8)
        trimmed= trim_to_bounding_box(binarized)
        if trimmed is None:
            continue
        score = asymmetry_score(trimmed)
        if score is not None and 0 <= score <= 1:
            scores.append(score)

    if not scores:
        return None, 0
    return round(float(np.mean(scores)), 4), len(scores)


def run():
    blocklist= load_masks_blocklist()
    print(f"  Blocklist loaded: {len(blocklist)} masks to skip")

    mask_files= [
        f for f in sorted(MASK_DIR.iterdir())
        if f.suffix.lower() in VALID_EXTENSIONS and "_mask" in f.name
    ]
    print(f"  Masks found: {len(mask_files)}")

    rows = []
    for mask_path in mask_files:
        stem = mask_path.name.replace("_mask.png", "").replace("_mask", "")
        if stem in blocklist:
            continue

        try:
            mask_bin = load_mask(mask_path)
            if mask_bin.sum() < 50:
                continue

            score, valid_n = feature_asymmetry(mask_bin.astype(np.float64))
            if score is None:
                continue

            rows.append({
                "image_name":      mask_path.name.replace("_mask", ""),
                "asymmetry_score": score,
                "rotations_used":  valid_n,
            })
        except Exception as e:
            print(f"  [ERROR] {mask_path.name}: {e}")

    df = pd.DataFrame(rows).sort_values("image_name").reset_index(drop=True)
    print(f"  Extracted {len(df)} rows")
    return df


if __name__ == "__main__":
    print(run().head())
