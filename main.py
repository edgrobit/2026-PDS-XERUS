import sys
import argparse
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Feature scripts ───────────────────────────────────────────────────────────
from src.Feature_hair_removal         import run_dataset_pipeline as run_hair_removal
from src.Feature_penmark_mask import run as run_pen_removal
from src.Feature_asymmetry    import run as run_asymmetry
from src.Feature_borders      import run as run_borders
from src.Feature_skincolour   import run as run_skincolour

# ── Model scripts ─────────────────────────────────────────────────────────────
from results.models.model_KNN                 import run as run_knn
from results.models.model_decision_tree       import run as run_dt
from results.models.model_logistic_regression import run as run_lr

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data")
IMGS_DIR        = DATA_DIR / "imgs"
IMGS_HAIRLESS   = DATA_DIR / "imgs_hairless"
IMGS_CLEAN      = DATA_DIR / "imgs_clean"
MASKS_DIR       = DATA_DIR / "masks"
ANNOTATIONS_CSV = DATA_DIR / "annotations_combined.csv"
METADATA_CSV    = DATA_DIR / "metadata.csv"
OUTPUT_CSV      = DATA_DIR / "features.csv"

MALIGNANT = {"BCC", "SCC", "MEL"}


# ── Step 0a: Hair removal ─────────────────────────────────────────────────────

def remove_hair():
    print("\n" + "="*60)
    print("STEP 0a — HAIR REMOVAL")
    print("="*60)
    print(f"Removing hair: {IMGS_DIR} → {IMGS_HAIRLESS}")
    run_hair_removal(
        input_dir  = str(IMGS_DIR),
        output_dir = str(IMGS_HAIRLESS),
    )


# ── Step 0b: Pen mark removal ─────────────────────────────────────────────────

def remove_pen_marks():
    print("\n" + "="*60)
    print("STEP 0b — PEN MARK REMOVAL")
    print("="*60)

    # Read from imgs_hairless/ if it exists, otherwise fall back to imgs/
    if IMGS_HAIRLESS.exists() and any(IMGS_HAIRLESS.iterdir()):
        src = IMGS_HAIRLESS
    else:
        print("  WARNING: imgs_hairless/ not found, reading from imgs/ instead")
        src = IMGS_DIR

    print(f"Removing pen marks: {src} → {IMGS_CLEAN}")
    run_pen_removal(
        img_dir         = src,
        mask_dir        = MASKS_DIR,
        annotations_csv = ANNOTATIONS_CSV,
        output_dir      = IMGS_CLEAN,
    )


# ── Step 1: Feature extraction and build master CSV ───────────────────────────

def extract_features():
    print("\n" + "="*60)
    print("STEP 1 — FEATURE EXTRACTION")
    print("="*60)

    print("\n[1/3] Borders and shape...")
    bord = run_borders()
    bord["stem"] = bord["image_name"].str.replace(".png", "", regex=False)

    print("\n[2/3] Asymmetry...")
    asym = run_asymmetry()
    asym["stem"] = asym["image_name"].str.replace(".png", "", regex=False)

    print("\n[3/3] Skin color (ITA, FST, RGB/HSV variance)...")
    color = run_skincolour()
    color["stem"] = color["image_name"].str.replace(".png", "", regex=False)

    # Merge all features together on stem
    df = bord.merge(
        asym[["stem", "asymmetry_score", "rotations_used"]],
        on="stem", how="left").merge(
        color[["stem", "rgb_var_r", "rgb_var_g", "rgb_var_b",
               "hsv_var_h", "hsv_var_s", "hsv_var_v",
               "ita_mean", "fst_predicted"]],
        on="stem", how="left")

    # Merge metadata for diagnostic labels
    meta = pd.read_csv(METADATA_CSV)
    meta["stem"] = meta["img_id"].str.replace(".png", "", regex=False)
    df = df.merge(
        meta[["stem", "diagnostic", "fitspatrick"]],
        on="stem", how="left")

    df["img_id"] = df["stem"]
    df["label"]  = df["diagnostic"].apply(
        lambda x: 1 if x in MALIGNANT else 0)
    
    df = df.drop(columns=["stem"]).sort_values("img_id").reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nMaster CSV saved → {OUTPUT_CSV}")
    print(f"  Rows:      {len(df)}")
    print(f"  Columns:   {df.columns.tolist()}")
    print(f"  Malignant: {df['label'].sum()} | Benign: {(df['label']==0).sum()}")

    return df


# ── Step 2: Train models ──────────────────────────────────────────────────────

def train_models():
    print("\n" + "="*60)
    print("STEP 2 — MODEL TRAINING")
    print("="*60)

    knn_results = run_knn(features_csv=OUTPUT_CSV)
    dt_results  = run_dt(features_csv=OUTPUT_CSV)
    lr_results  = run_lr(features_csv=OUTPUT_CSV)

    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)
    print(f"  {'Model':<30} {'Accuracy':>10} {'ROC-AUC':>10}")
    print(f"  {'-'*50}")

    best_knn = max(knn_results.values(), key=lambda r: r["roc_auc"])
    best_dt  = max(dt_results.values(),  key=lambda r: r["roc_auc"])
    best_lr  = max(lr_results.values(),  key=lambda r: r["roc_auc"])

    print(f"  {'KNN (best feature set)':<30} "
          f"{best_knn['accuracy']:>10.3f} {best_knn['roc_auc']:>10.3f}")
    print(f"  {'Decision Tree (best)':<30} "
          f"{best_dt['accuracy']:>10.3f} {best_dt['roc_auc']:>10.3f}")
    print(f"  {'Logistic Regression (best)':<30} "
          f"{best_lr['accuracy']:>10.3f} {best_lr['roc_auc']:>10.3f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Skin lesion pipeline")
    parser.add_argument("--models-only",   action="store_true",
                        help="Skip to model training — features.csv must exist")
    parser.add_argument("--features-only", action="store_true",
                        help="Extract features only, skip models")
    parser.add_argument("--skip-hair",     action="store_true",
                        help="Skip hair removal — use existing data/imgs_hairless/")
    parser.add_argument("--skip-pen",      action="store_true",
                        help="Skip hair + pen removal — use existing data/imgs_clean/")
    args = parser.parse_args()

    if not args.models_only:
        if not args.skip_pen:
            if not args.skip_hair:
                remove_hair()
            else:
                print("\nSkipping hair removal (--skip-hair)")
            remove_pen_marks()
        else:
            print("\nSkipping hair and pen removal (--skip-pen)")

        extract_features()

    if not args.features_only:
        if not OUTPUT_CSV.exists():
            print(f"\nERROR: {OUTPUT_CSV} not found.")
            print("Run without --models-only first to generate features.")
            return
        train_models()

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
