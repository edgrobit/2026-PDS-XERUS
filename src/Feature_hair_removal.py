import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import os
import glob
from utils import load_masks_blocklist

def get_valid_area_mask(img_gray):
    _, mask= cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
    return mask


def calculate_hair_coverage(img_bgr):
    base_channel= img_bgr[:, :, 1] 
    img_gray= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    valid_mask= get_valid_area_mask(img_gray)
    clahe= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_base = clahe.apply(base_channel)
    sobel_x= cv2.Sobel(enhanced_base, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y= cv2.Sobel(enhanced_base, cv2.CV_64F, 0, 1, ksize=3)
    sobel_norm= cv2.normalize(cv2.magnitude(sobel_x, sobel_y),None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    lap_norm= cv2.normalize(np.absolute(cv2.Laplacian(enhanced_base, cv2.CV_64F)),None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    averaged_edges= cv2.addWeighted(sobel_norm, 0.5, lap_norm, 0.5, 0)

    blurred= cv2.GaussianBlur(averaged_edges, (3, 3), 0)
    _, binary_hair_mask= cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hair_pixels= cv2.countNonZero(cv2.bitwise_and(binary_hair_mask, valid_mask))
    valid_pixels= cv2.countNonZero(valid_mask)

    return 0.0 if valid_pixels == 0 else hair_pixels / valid_pixels


def process_image(img_bgr):
    coverage = calculate_hair_coverage(img_bgr)

    if coverage < 0.005:
        return img_bgr, coverage, False

    k_size = 15 if coverage <= 0.035 else 25
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_size, k_size))

    base_channel = img_bgr[:, :, 1]

    blackhat= cv2.morphologyEx(base_channel, cv2.MORPH_BLACKHAT, kernel)
    tophat= cv2.morphologyEx(base_channel, cv2.MORPH_TOPHAT,   kernel)
    combined= cv2.max(blackhat, tophat)

    block_size= (k_size * 2) + 1
    removal_mask= cv2.adaptiveThreshold(
        combined, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, -5
    )


    closing_kernel= cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    removal_mask= cv2.morphologyEx(removal_mask, cv2.MORPH_CLOSE, closing_kernel)

    cleaned = cv2.inpaint(img_bgr, removal_mask, inpaintRadius=3,flags=cv2.INPAINT_TELEA)

    return cleaned, coverage, True


def run_dataset_pipeline(input_dir, output_dir):
    blocklist= load_masks_blocklist()
    image_files= sorted(glob.glob(os.path.join(input_dir, "*.png")))

    if not image_files:
        print(f"No PNG images found in '{input_dir}'. Check the path.")
        return

    os.makedirs(output_dir, exist_ok=True)

    total= len(image_files)
    processed= skipped_hair = skipped_block = errors = 0

    print(f"Found {total} images | Blocklist: {len(blocklist)} to skip\n")

    for i, filepath in enumerate(image_files, 1):
        filename= os.path.basename(filepath)
        stem= os.path.splitext(filename)[0]
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
            status= "Cleaned"
        else:
            skipped_hair += 1
            status= "No hair"

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

if __name__ == "__main__":
    run_dataset_pipeline(
        input_dir  = "data/imgs",
        output_dir = "data/imgs_hairless",
    )
