"""
main.py — Full pipeline orchestration
======================================
Runs the complete project pipeline in order:

    0. Pen mark removal    → data/imgs_clean/
    1. Feature extraction  → each script appends to data/features.csv
    2. Model training      → results/

Usage:
    python main.py                 # full pipeline
    python main.py --models-only   # skip to model training (features.csv must exist)
    python main.py --features-only # extract features only, skip models
    python main.py --skip-pen      # skip pen removal (use existing data/imgs_clean/)
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Feature scripts ───────────────────────────────────────────────────────────
from src.Feature_penmark_mask import run as run_pen_removal
from src.Feature_asymmetry    import run as run_asymmetry
from src.Feature_borders      import run as run_borders
from src.Feature_skincolour   import run as run_skincolour

# ── Model scripts ─────────────────────────────────────────────────────────────
from results.models.model_KNN                 import run as run_knn
from results.models.model_decision_tree       import run as run_dt
from results.models.model_logistic_regression import run as run_lr

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data")
METADATA_CSV = DATA_DIR / "metadata.csv"
OUTPUT_CSV   = DATA_DIR / "features.csv"   # single master feature file

MALIGNANT = {"BCC", "SCC", "MEL"}


# ── Step 0: Pen mark removal ──────────────────────────────────────────────────

def remove_pen_marks():
    print("\n" + "="*60)
    print("STEP 0 — PEN MARK REMOVAL")
    print("="*60)
    print("Inpainting pen marks → data/imgs_clean/")
    run_pen_removal()


# ── Step 1: Feature extraction and build master CSV ───────────────────────────

def extract_features():
    print("\n" + "="*60)
    print("STEP 1 — FEATURE EXTRACTION")
    print("="*60)

    # --- 1a. Borders and shape (starts the master CSV) ---
    print("\n[1/3] Borders and shape...")
    bord = run_borders()          # returns a dataframe
    bord["stem"] = bord["image_name"].str.replace(".png", "", regex=False)

    # --- 1b. Asymmetry (join on stem) ---
    print("\n[2/3] Asymmetry...")
    asym = run_asymmetry()
    asym["stem"] = asym["image_name"].str.replace(".png", "", regex=False)
    df = bord.merge(
        asym[["stem", "asymmetry_score", "rotations_used"]],
        on="stem", how="left"
    )

    # --- 1c. Skin color (join on stem) ---
    print("\n[3/3] Skin color (ITA, FST, RGB/HSV variance)...")
    color = run_skincolour()
    color["stem"] = color["image_name"].str.replace(".png", "", regex=False)
    df = df.merge(
        color[["stem", "rgb_var_r", "rgb_var_g", "rgb_var_b",
               "hsv_var_h", "hsv_var_s", "hsv_var_v",
               "ita_mean", "fst_predicted"]],
        on="stem", how="left"
    )

    # --- Merge metadata for labels ---
    meta = pd.read_csv(METADATA_CSV)
    meta["stem"] = meta["img_id"].str.replace(".png", "", regex=False)
    df = df.merge(
        meta[["stem", "diagnostic", "fitspatrick"]],
        on="stem", how="left"
    )

    # --- Final tidy up ---
    df["img_id"] = df["stem"]
    df["label"]  = df["diagnostic"].apply(
        lambda x: 1 if x in MALIGNANT else 0
    )
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

    # Summary comparison across all models
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
    parser.add_argument("--skip-pen",      action="store_true",
                        help="Skip pen removal — use existing data/imgs_clean/")
    args = parser.parse_args()

    if not args.models_only:
        if not args.skip_pen:
            remove_pen_marks()
        else:
            print("\nSkipping pen removal (--skip-pen)")

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
