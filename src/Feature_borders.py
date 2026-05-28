import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from utils import (find_project_paths, load_mask, VALID_EXTENSIONS,
                   preprocess_mask, get_main_lesion, load_masks_blocklist)

PATHS= find_project_paths()
MASK_DIR= PATHS["mask_dir"]


def extract_border_features(mask_bin):
    """Extract shape and border features from a binary lesion mask."""
    cleaned= preprocess_mask(mask_bin)
    lesion, region = get_main_lesion(cleaned)
    if lesion is None:
        return None

    area= region.area
    perimeter= region.perimeter 
    total_pixels= lesion.shape[0] * lesion.shape[1]

    return {
        "total_pixels": total_pixels,
        "area": area,
        "lesion_percentage": area / total_pixels if total_pixels > 0 else np.nan,
        "compactness": (perimeter ** 2) / (4 * np.pi * area) if area > 0 else np.nan,
    }


def run():
    blocklist= load_masks_blocklist()
    print(f"  Blocklist loaded: {len(blocklist)} masks to skip")

    mask_files= [
        f for f in sorted(MASK_DIR.iterdir())
        if f.suffix.lower() in VALID_EXTENSIONS and "_mask" in f.name]
    print(f"  Masks found: {len(mask_files)}")

    rows= []
    for mask_path in mask_files:
        stem= mask_path.name.replace("_mask.png", "").replace("_mask", "")
        if stem in blocklist:
            continue

        try:
            mask_bin= load_mask(mask_path)
            if mask_bin.sum() < 50:
                continue

            features = extract_border_features(mask_bin)
            if features is None:
                continue

            rows.append({
                "image_name": mask_path.name.replace("_mask", ""),
                **features,
            })
        except Exception as e:
            print(f"  [ERROR] {mask_path.name}: {e}")

    df= pd.DataFrame(rows).sort_values("image_name").reset_index(drop=True)
    print(f"  Extracted {len(df)} rows")
    return df


if __name__ == "__main__":
    print(run().head())
