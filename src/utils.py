from pathlib import Path
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage import img_as_float, transform, morphology, measure

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def find_project_paths():
    try:
        start= Path(__file__).resolve().parent
    except NameError:
        start= Path.cwd().resolve()

    for parent in [start] + list(start.parents):
        img_dir= parent / "data" / "imgs"
        mask_candidates = [
            parent / "data" / "masks" / "masks",
            parent / "data" / "masks",
        ]
        if img_dir.exists():
            for mask_dir in mask_candidates:
                if mask_dir.exists():
                    return {
                        "project_root": parent,
                        "img_dir":      img_dir,
                        "mask_dir":     mask_dir,
                    }

    raise FileNotFoundError(
        "Could not find project folders. Make sure the repo contains "
        "'data/imgs' and either 'data/masks' or 'data/masks/masks'."
    )


def load_masks_blocklist(csv_path="data/masks_without_images.csv"):
    path= Path(csv_path)
    if not path.exists():
        return set()
    df= pd.read_csv(path)
    return set(df.iloc[:, 0].str.replace(".png", "", regex=False).tolist())


def load_image(path, max_size=512):
    img= img_as_float(imread(path))
    if img.ndim == 2:
        img= np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img= img[:, :, :3]
    if max_size is not None:
        h, w= img.shape[:2]
        scale= max_size / max(h, w)
        if scale < 1.0:
            img = transform.resize(img, (int(h * scale), int(w * scale)),
                                   anti_aliasing=True)
    return img


def load_mask(mask_path):
    mask= imread(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.max() > 1:
        mask = mask / 255.0
    return mask > 0.5


def find_mask_for_image(image_path, mask_dir):
    stem= image_path.stem
    for ext in VALID_EXTENSIONS:
        for name in [f"{stem}_mask{ext}", f"{stem}{ext}"]:
            candidate = mask_dir / name
            if candidate.exists():
                return candidate
    return None


def preprocess_mask(mask_bin):
    mask_bin= morphology.remove_small_objects(mask_bin, min_size=100)
    mask_bin= morphology.remove_small_holes(mask_bin, area_threshold=100)
    return mask_bin


def get_main_lesion(mask_bin):
    labels= measure.label(mask_bin)
    regions= measure.regionprops(labels)
    if not regions:
        return None, None
    largest = max(regions, key=lambda r: r.area)
    return labels == largest.label, largest
