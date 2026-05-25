"""
Decision Tree — Three Feature Sets
==================================

This script runs three Decision Tree models using the same data structure:

1. Border + asymmetry features only
2. Border + asymmetry + color variance features, excluding ITA and Fitzpatrick
3. Border + asymmetry + color variance + ITA + predicted Fitzpatrick

Input:
    data/Separate csv of features/features_all.csv

Target:
    1 = malignant: BCC, SCC, MEL
    0 = benign: ACK, NEV, SEK

Outputs:
    results/models/
    results/predictions/
    results/figures/
    results/reports/

Run:
    python DecisionTree_Three_Feature_Sets.py

or from the project root:
    python src/DecisionTree_Three_Feature_Sets.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)


# ============================================================
# 1. Project paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# If this script is inside src/, project root is one folder above.
if SCRIPT_DIR.name == "src":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

FEATURES_CSV = PROJECT_ROOT / "data" / "Separate csv of features" / "features_all.csv"
ANNOTATIONS_CSV = PROJECT_ROOT / "data" / "annotations_combined.csv"

RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR = RESULTS_DIR / "models"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Features CSV:", FEATURES_CSV)


# ============================================================
# 2. Model configuration
# ============================================================

MALIGNANT = {"BCC", "SCC", "MEL"}
BENIGN = {"ACK", "NEV", "SEK"}
VALID_DIAGNOSTICS = MALIGNANT | BENIGN

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Decision trees can overfit easily, so we test simpler and deeper trees.
MAX_DEPTH_VALUES = [2, 3, 4, 5, 6, 8, 10, None]
MIN_SAMPLES_LEAF_VALUES = [1, 2, 5, 10, 20]
CRITERION_VALUES = ["gini", "entropy"]


# ============================================================
# 3. Feature set configuration
# ============================================================

BORDER_ASYMMETRY_FEATURES = [
    "total_pixels",
    "area",
    "lesion_percentage",
    "perimeter",
    "compactness",
    "border_pixels",
    "asymmetry_score",
]

COLOR_FEATURES_NO_ITA_FST = [
    "rgb_var_r",
    "rgb_var_g",
    "rgb_var_b",
    "hsv_var_h",
    "hsv_var_s",
    "hsv_var_v",
]

ITA_FITZPATRICK_FEATURES = [
    "ita_mean",
    "fst_predicted",
]

FEATURE_SETS = {
    "border_asymmetry": {
        "description": "Only asymmetry and border/shape features",
        "features": BORDER_ASYMMETRY_FEATURES,
    },
    "border_asymmetry_color_no_ita_fst": {
        "description": "Border/asymmetry plus color variance, excluding ITA and Fitzpatrick",
        "features": BORDER_ASYMMETRY_FEATURES + COLOR_FEATURES_NO_ITA_FST,
    },
    "all_features": {
        "description": "Border/asymmetry plus color variance, ITA, and predicted Fitzpatrick",
        "features": BORDER_ASYMMETRY_FEATURES + COLOR_FEATURES_NO_ITA_FST + ITA_FITZPATRICK_FEATURES,
    },
}


# ============================================================
# 4. Helper functions
# ============================================================

def clean_stem(series):
    """
    Creates comparable image stems by removing common image extensions.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".png", "", regex=False)
        .str.replace(".jpg", "", regex=False)
        .str.replace(".jpeg", "", regex=False)
        .str.replace(".tif", "", regex=False)
        .str.replace(".tiff", "", regex=False)
        .str.replace(".bmp", "", regex=False)
    )


def add_diagnostic_if_missing(df):
    """
    Uses diagnostic from features_all.csv if it already exists.

    If diagnostic is missing, the function tries to merge it from:
        data/annotations_combined.csv
    """

    if "diagnostic" in df.columns:
        return df

    print("\n'diagnostic' not found in features_all.csv.")
    print("Trying to merge diagnostic from annotations_combined.csv...")

    if not ANNOTATIONS_CSV.exists():
        raise ValueError(
            "The column 'diagnostic' was not found in features_all.csv, "
            "and annotations_combined.csv was not found either."
        )

    metadata = pd.read_csv(ANNOTATIONS_CSV)

    possible_image_cols = ["img_id", "image_name", "file", "filename"]
    possible_diag_cols = ["diagnostic", "diagnosis", "label"]

    image_col = None
    for col in possible_image_cols:
        if col in metadata.columns:
            image_col = col
            break

    diag_col = None
    for col in possible_diag_cols:
        if col in metadata.columns:
            diag_col = col
            break

    if image_col is None:
        raise ValueError(
            "Could not find an image identifier column in annotations_combined.csv. "
            f"Expected one of: {possible_image_cols}"
        )

    if diag_col is None:
        raise ValueError(
            "Could not find a diagnostic column in annotations_combined.csv. "
            f"Expected one of: {possible_diag_cols}"
        )

    if "image_name" not in df.columns:
        raise ValueError(
            "features_all.csv must contain 'image_name' to merge diagnostics."
        )

    df["stem"] = clean_stem(df["image_name"])
    metadata["stem"] = clean_stem(metadata[image_col])

    metadata_small = metadata[["stem", diag_col]].copy()
    metadata_small = metadata_small.rename(columns={diag_col: "diagnostic"})

    df = df.merge(metadata_small, on="stem", how="left")

    return df


def clean_diagnostics(df):
    """
    Cleans diagnostic labels and creates the binary target.

    label:
        1 = malignant
        0 = benign
    """

    df = df.copy()

    df["diagnostic"] = df["diagnostic"].astype(str).str.strip().str.upper()

    df = df[df["diagnostic"].isin(VALID_DIAGNOSTICS)].copy()

    if df.empty:
        raise ValueError(
            "No valid diagnostic rows remained after filtering. "
            "Expected diagnostics: BCC, SCC, MEL, ACK, NEV, SEK."
        )

    df["label"] = df["diagnostic"].apply(lambda x: 1 if x in MALIGNANT else 0)

    return df


def get_available_features(df, requested_features, feature_set_name):
    """
    Keeps only requested features that exist in the dataframe and are numeric.
    """

    available = [
        col for col in requested_features
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    missing = [
        col for col in requested_features
        if col not in df.columns
    ]

    non_numeric = [
        col for col in requested_features
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    ]

    if missing:
        print(f"\n[{feature_set_name}] Missing columns skipped:")
        print(missing)

    if non_numeric:
        print(f"\n[{feature_set_name}] Non-numeric columns skipped:")
        print(non_numeric)

    if not available:
        raise ValueError(
            f"No usable numeric features found for feature set: {feature_set_name}"
        )

    return available


def get_cv_object(y_train):
    """
    Creates a stratified CV object while avoiding too many folds for small classes.
    """

    min_class_count = pd.Series(y_train).value_counts().min()
    cv_folds = min(5, int(min_class_count))

    if cv_folds < 2:
        raise ValueError("Not enough samples per class for cross-validation.")

    return StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def make_decision_tree(max_depth, min_samples_leaf, criterion):
    """
    Creates a Decision Tree classifier.
    """

    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def normalize_max_depth(value):
    """
    Converts pandas NaN/None values into Python None for max_depth.
    """

    if pd.isna(value):
        return None

    return int(value)


def run_one_decision_tree_model(df, feature_set_name, feature_set_info):
    """
    Runs the full Decision Tree workflow for one feature set:
    - select features
    - train/test split
    - hyperparameter search
    - train final model
    - evaluate
    - save outputs
    """

    print("\n" + "=" * 80)
    print(f"Running feature set: {feature_set_name}")
    print(feature_set_info["description"])
    print("=" * 80)

    requested_features = feature_set_info["features"]
    feature_cols = get_available_features(df, requested_features, feature_set_name)

    print("\nFeatures used:")
    for feature in feature_cols:
        print(f"- {feature}")

    df_model = df[["image_name", "diagnostic", "label"] + feature_cols].copy()

    # Replace infinite values with missing values.
    df_model = df_model.replace([np.inf, -np.inf], np.nan)

    # Drop rows with missing values in the selected features.
    df_model = df_model.dropna(subset=feature_cols + ["label"])

    print(f"\nRows after dropping missing values: {len(df_model)}")
    print(f"Malignant rows: {int(df_model['label'].sum())}")
    print(f"Benign rows:    {int((df_model['label'] == 0).sum())}")

    if df_model["label"].nunique() < 2:
        raise ValueError(
            f"The feature set '{feature_set_name}' has only one class after cleaning."
        )

    X = df_model[feature_cols]
    y = df_model["label"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df_model.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")

    cv = get_cv_object(y_train)

    # ------------------------------------------------------------
    # Hyperparameter search
    # ------------------------------------------------------------

    print("\nSearching for best Decision Tree parameters...")

    grid_results = []

    for criterion in CRITERION_VALUES:
        for max_depth in MAX_DEPTH_VALUES:
            for min_samples_leaf in MIN_SAMPLES_LEAF_VALUES:
                model_candidate = make_decision_tree(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                )

                auc_scores = cross_val_score(
                    model_candidate,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="roc_auc",
                )

                acc_scores = cross_val_score(
                    model_candidate,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="accuracy",
                )

                grid_results.append({
                    "feature_set": feature_set_name,
                    "criterion": criterion,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "cv_auc_mean": auc_scores.mean(),
                    "cv_auc_std": auc_scores.std(),
                    "cv_accuracy_mean": acc_scores.mean(),
                    "cv_accuracy_std": acc_scores.std(),
                })

                print(
                    f"criterion={criterion:<7} | "
                    f"max_depth={str(max_depth):<4} | "
                    f"min_leaf={min_samples_leaf:<2} | "
                    f"CV ROC-AUC={auc_scores.mean():.3f} ± {auc_scores.std():.3f} | "
                    f"CV Accuracy={acc_scores.mean():.3f} ± {acc_scores.std():.3f}"
                )

    grid_results_df = pd.DataFrame(grid_results)

    # Choose best parameter set by ROC-AUC.
    best_row = grid_results_df.sort_values("cv_auc_mean", ascending=False).iloc[0]

    best_criterion = best_row["criterion"]
    best_max_depth = normalize_max_depth(best_row["max_depth"])
    best_min_samples_leaf = int(best_row["min_samples_leaf"])

    print("\nBest parameters:")
    print(f"criterion:        {best_criterion}")
    print(f"max_depth:        {best_max_depth}")
    print(f"min_samples_leaf: {best_min_samples_leaf}")
    print(f"Best CV ROC-AUC:  {best_row['cv_auc_mean']:.3f}")
    print(f"Best CV accuracy: {best_row['cv_accuracy_mean']:.3f}")

    # ------------------------------------------------------------
    # Train final model
    # ------------------------------------------------------------

    model = make_decision_tree(
        max_depth=best_max_depth,
        min_samples_leaf=best_min_samples_leaf,
        criterion=best_criterion,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    accuracy = (y_pred == y_test).mean()
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print(f"\nTest accuracy: {accuracy:.3f}")
    print(f"Test ROC-AUC:  {roc_auc:.3f}")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malignant"],
        zero_division=0,
    )

    print("\nClassification report:")
    print(report)

    # ------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------

    safe_name = feature_set_name

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        cm,
        display_labels=["Benign", "Malignant"],
    ).plot(ax=ax, colorbar=False)

    ax.set_title(f"Decision Tree Confusion Matrix — {safe_name}")
    plt.tight_layout()

    confusion_matrix_path = FIGURES_DIR / f"decision_tree_{safe_name}_confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=150)
    plt.close()

    print(f"Saved: {confusion_matrix_path}")

    # Tree plot
    tree_plot_path = FIGURES_DIR / f"decision_tree_{safe_name}_plot.png"

    plt.figure(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=feature_cols,
        class_names=["Benign", "Malignant"],
        filled=True,
        rounded=True,
        max_depth=4,
        fontsize=8,
    )
    plt.title(f"Decision Tree Plot — {safe_name}, truncated to depth 4")
    plt.tight_layout()
    plt.savefig(tree_plot_path, dpi=150)
    plt.close()

    print(f"Saved: {tree_plot_path}")

    # Feature importance plot
    feature_importance_fig_path = FIGURES_DIR / f"decision_tree_{safe_name}_feature_importance.png"

    top_n = min(15, len(importance_df))
    plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Feature importance")
    plt.title(f"Decision Tree Feature Importance — {safe_name}")
    plt.tight_layout()
    plt.savefig(feature_importance_fig_path, dpi=150)
    plt.close()

    print(f"Saved: {feature_importance_fig_path}")

    # Predictions
    pred_df = df_model.loc[idx_test].copy().reset_index(drop=True)

    pred_df["feature_set"] = feature_set_name
    pred_df["predicted_label"] = y_pred
    pred_df["predicted_prob_malignant"] = y_pred_prob.round(4)
    pred_df["correct"] = y_pred == y_test

    predictions_path = PREDICTIONS_DIR / f"decision_tree_{safe_name}_predictions.csv"
    pred_df.to_csv(predictions_path, index=False)

    print(f"Saved: {predictions_path}")

    # Feature importances
    feature_importance_path = REPORTS_DIR / f"decision_tree_{safe_name}_feature_importances.csv"
    importance_df.to_csv(feature_importance_path, index=False)

    print(f"Saved: {feature_importance_path}")

    # Grid search
    grid_search_path = REPORTS_DIR / f"decision_tree_{safe_name}_grid_search.csv"
    grid_results_df.to_csv(grid_search_path, index=False)

    print(f"Saved: {grid_search_path}")

    # Model
    model_path = MODELS_DIR / f"decision_tree_{safe_name}_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "features": feature_cols,
            "feature_set": feature_set_name,
            "best_criterion": best_criterion,
            "best_max_depth": best_max_depth,
            "best_min_samples_leaf": best_min_samples_leaf,
            "malignant_classes": sorted(MALIGNANT),
            "benign_classes": sorted(BENIGN),
        }, f)

    print(f"Saved: {model_path}")

    # Text report
    report_path = REPORTS_DIR / f"decision_tree_{safe_name}_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Decision Tree Classification Report — {feature_set_name}\n")
        f.write("=" * 70)
        f.write("\n\n")

        f.write(f"Description: {feature_set_info['description']}\n")
        f.write(f"Features CSV: {FEATURES_CSV}\n")
        f.write(f"Test size: {TEST_SIZE}\n")
        f.write(f"Random state: {RANDOM_STATE}\n")
        f.write(f"Class weight: balanced\n\n")

        f.write("Diagnostic mapping:\n")
        f.write("1 = malignant: BCC, SCC, MEL\n")
        f.write("0 = benign: ACK, NEV, SEK\n\n")

        f.write("Best hyperparameters:\n")
        f.write(f"criterion: {best_criterion}\n")
        f.write(f"max_depth: {best_max_depth}\n")
        f.write(f"min_samples_leaf: {best_min_samples_leaf}\n\n")

        f.write("Features used:\n")
        for feature in feature_cols:
            f.write(f"- {feature}\n")

        f.write("\nDataset:\n")
        f.write(f"Total samples after cleaning: {len(df_model)}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Malignant samples: {int(df_model['label'].sum())}\n")
        f.write(f"Benign samples: {int((df_model['label'] == 0).sum())}\n")

        f.write("\nBest cross-validation result:\n")
        f.write(f"CV ROC-AUC: {best_row['cv_auc_mean']:.3f} ± {best_row['cv_auc_std']:.3f}\n")
        f.write(f"CV Accuracy: {best_row['cv_accuracy_mean']:.3f} ± {best_row['cv_accuracy_std']:.3f}\n\n")

        f.write("Test performance:\n")
        f.write(f"Accuracy: {accuracy:.3f}\n")
        f.write(f"ROC-AUC: {roc_auc:.3f}\n\n")

        f.write("Classification report:\n")
        f.write(report)
        f.write("\n\n")

        f.write("Feature importances:\n")
        f.write(importance_df.to_string(index=False))

    print(f"Saved: {report_path}")

    result_summary = {
        "feature_set": feature_set_name,
        "description": feature_set_info["description"],
        "features_used": ", ".join(feature_cols),
        "n_features": len(feature_cols),
        "n_samples": len(df_model),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "malignant_samples": int(df_model["label"].sum()),
        "benign_samples": int((df_model["label"] == 0).sum()),
        "best_criterion": best_criterion,
        "best_max_depth": best_max_depth,
        "best_min_samples_leaf": best_min_samples_leaf,
        "best_cv_accuracy": best_row["cv_accuracy_mean"],
        "best_cv_auc": best_row["cv_auc_mean"],
        "test_accuracy": accuracy,
        "test_roc_auc": roc_auc,
        "confusion_matrix_path": str(confusion_matrix_path),
        "tree_plot_path": str(tree_plot_path),
        "feature_importance_fig_path": str(feature_importance_fig_path),
        "predictions_path": str(predictions_path),
        "model_path": str(model_path),
        "report_path": str(report_path),
        "feature_importance_path": str(feature_importance_path),
    }

    return result_summary, grid_results_df


# ============================================================
# 5. Main
# ============================================================

def main():
    print("\nLoading data...")

    df = pd.read_csv(FEATURES_CSV)

    print("\nColumns in features_all.csv:")
    print(df.columns.tolist())

    if "image_name" not in df.columns:
        raise ValueError("The column 'image_name' was not found in features_all.csv.")

    df = add_diagnostic_if_missing(df)
    df = clean_diagnostics(df)

    print(f"\nTotal valid samples: {len(df)}")
    print(f"Malignant (1):      {int(df['label'].sum())}")
    print(f"Benign    (0):      {int((df['label'] == 0).sum())}")

    summaries = []
    all_grid_results = []

    for feature_set_name, feature_set_info in FEATURE_SETS.items():
        summary, grid_results = run_one_decision_tree_model(
            df=df,
            feature_set_name=feature_set_name,
            feature_set_info=feature_set_info,
        )

        summaries.append(summary)
        all_grid_results.append(grid_results)

    # Save combined comparison summary
    summary_df = pd.DataFrame(summaries)

    summary_path = REPORTS_DIR / "decision_tree_feature_set_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved combined summary: {summary_path}")

    # Save combined grid-search results
    all_grid_results_df = pd.concat(all_grid_results, ignore_index=True)

    all_grid_results_path = REPORTS_DIR / "decision_tree_feature_set_comparison_grid_search.csv"
    all_grid_results_df.to_csv(all_grid_results_path, index=False)

    print(f"Saved combined grid-search results: {all_grid_results_path}")

    print("\nFinal comparison:")
    print(
        summary_df[
            [
                "feature_set",
                "n_features",
                "best_criterion",
                "best_max_depth",
                "best_min_samples_leaf",
                "best_cv_accuracy",
                "best_cv_auc",
                "test_accuracy",
                "test_roc_auc",
            ]
        ]
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
