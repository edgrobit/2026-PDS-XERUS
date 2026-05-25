"""
KNN Model — Three Feature Sets
==============================

This script runs three KNN models using the same data structure:

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
    python KNN_Three_Feature_Sets.py

or from the project root:
    python src/KNN_Three_Feature_Sets.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
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

# If the script is inside src/, project root is one folder above.
if SCRIPT_DIR.name == "src":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

FEATURES_CSV = PROJECT_ROOT / "data" / "Separate csv of features" / "features_all.csv"

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

K_VALUES = range(1, 21)


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

def check_required_columns(df):
    """
    Checks that the minimum columns needed to run the model exist.
    """

    if "diagnostic" not in df.columns:
        raise ValueError(
            "The column 'diagnostic' was not found in features_all.csv. "
            "The model needs this column to create the malignant/benign label."
        )

    if "image_name" not in df.columns:
        raise ValueError(
            "The column 'image_name' was not found in features_all.csv."
        )


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
        raise ValueError(
            "Not enough samples per class for cross-validation."
        )

    return StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def run_one_knn_model(df, feature_set_name, feature_set_info):
    """
    Runs the full KNN workflow for one feature set:
    - select features
    - train/test split
    - scale features
    - search best k
    - train final model
    - evaluate
    - save outputs
    """

    print("\n" + "=" * 70)
    print(f"Running feature set: {feature_set_name}")
    print(feature_set_info["description"])
    print("=" * 70)

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

    X = df_model[feature_cols].values
    y = df_model["label"].values

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

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cv = get_cv_object(y_train)

    # ------------------------------------------------------------
    # Search best k
    # ------------------------------------------------------------

    print("\nSearching for optimal k from 1 to 20...")

    k_results = []

    for k in K_VALUES:
        knn_k = KNeighborsClassifier(
            n_neighbors=k,
            metric="euclidean",
        )

        acc_scores = cross_val_score(
            knn_k,
            X_train_scaled,
            y_train,
            cv=cv,
            scoring="accuracy",
        )

        auc_scores = cross_val_score(
            knn_k,
            X_train_scaled,
            y_train,
            cv=cv,
            scoring="roc_auc",
        )

        k_results.append({
            "feature_set": feature_set_name,
            "k": k,
            "cv_accuracy_mean": acc_scores.mean(),
            "cv_accuracy_std": acc_scores.std(),
            "cv_auc_mean": auc_scores.mean(),
            "cv_auc_std": auc_scores.std(),
        })

        print(
            f"k={k:2d} | "
            f"CV accuracy={acc_scores.mean():.3f} ± {acc_scores.std():.3f} | "
            f"CV ROC-AUC={auc_scores.mean():.3f} ± {auc_scores.std():.3f}"
        )

    k_results_df = pd.DataFrame(k_results)

    # Choose best k by ROC-AUC.
    best_row = k_results_df.sort_values("cv_auc_mean", ascending=False).iloc[0]
    best_k = int(best_row["k"])

    print(f"\nBest k: {best_k}")
    print(f"Best CV ROC-AUC: {best_row['cv_auc_mean']:.3f}")
    print(f"Best CV accuracy: {best_row['cv_accuracy_mean']:.3f}")

    # ------------------------------------------------------------
    # Train final model
    # ------------------------------------------------------------

    knn = KNeighborsClassifier(
        n_neighbors=best_k,
        metric="euclidean",
    )

    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)
    y_pred_prob = knn.predict_proba(X_test_scaled)[:, 1]

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

    ax.set_title(f"KNN Confusion Matrix — {safe_name}")
    plt.tight_layout()

    confusion_matrix_path = FIGURES_DIR / f"knn_{safe_name}_confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=150)
    plt.close()

    print(f"Saved: {confusion_matrix_path}")

    # k search plot
    plt.figure(figsize=(8, 4))
    plt.plot(k_results_df["k"], k_results_df["cv_accuracy_mean"], marker="o", label="CV Accuracy")
    plt.plot(k_results_df["k"], k_results_df["cv_auc_mean"], marker="o", label="CV ROC-AUC")
    plt.axvline(best_k, linestyle="--", label=f"Best k={best_k}")

    plt.xlabel("k")
    plt.ylabel("Score")
    plt.title(f"KNN k Search — {safe_name}")
    plt.legend()
    plt.tight_layout()

    k_search_path = FIGURES_DIR / f"knn_{safe_name}_k_search.png"
    plt.savefig(k_search_path, dpi=150)
    plt.close()

    print(f"Saved: {k_search_path}")

    # Predictions
    pred_df = df_model.loc[idx_test].copy().reset_index(drop=True)

    pred_df["feature_set"] = feature_set_name
    pred_df["predicted_label"] = y_pred
    pred_df["predicted_prob_malignant"] = y_pred_prob.round(4)
    pred_df["correct"] = y_pred == y_test

    predictions_path = PREDICTIONS_DIR / f"knn_{safe_name}_predictions.csv"
    pred_df.to_csv(predictions_path, index=False)

    print(f"Saved: {predictions_path}")

    # Model
    model_path = MODELS_DIR / f"knn_{safe_name}_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": knn,
                "scaler": scaler,
                "features": feature_cols,
                "feature_set": feature_set_name,
                "best_k": best_k,
                "malignant_classes": sorted(MALIGNANT),
                "benign_classes": sorted(BENIGN),
            },
            f,
        )

    print(f"Saved: {model_path}")

    # k results
    k_results_path = REPORTS_DIR / f"knn_{safe_name}_k_search.csv"
    k_results_df.to_csv(k_results_path, index=False)

    print(f"Saved: {k_results_path}")

    # Text report
    report_path = REPORTS_DIR / f"knn_{safe_name}_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"KNN Classification Report — {feature_set_name}\n")
        f.write("=" * 60)
        f.write("\n\n")

        f.write(f"Description: {feature_set_info['description']}\n")
        f.write(f"Features CSV: {FEATURES_CSV}\n")
        f.write(f"Test size: {TEST_SIZE}\n")
        f.write(f"Random state: {RANDOM_STATE}\n")
        f.write(f"Best k: {best_k}\n\n")

        f.write("Diagnostic mapping:\n")
        f.write("1 = malignant: BCC, SCC, MEL\n")
        f.write("0 = benign: ACK, NEV, SEK\n\n")

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
        f.write(f"CV accuracy: {best_row['cv_accuracy_mean']:.3f} ± {best_row['cv_accuracy_std']:.3f}\n")
        f.write(f"CV ROC-AUC: {best_row['cv_auc_mean']:.3f} ± {best_row['cv_auc_std']:.3f}\n")

        f.write("\nTest performance:\n")
        f.write(f"Accuracy: {accuracy:.3f}\n")
        f.write(f"ROC-AUC: {roc_auc:.3f}\n\n")

        f.write("Classification report:\n")
        f.write(report)

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
        "best_k": best_k,
        "best_cv_accuracy": best_row["cv_accuracy_mean"],
        "best_cv_auc": best_row["cv_auc_mean"],
        "test_accuracy": accuracy,
        "test_roc_auc": roc_auc,
        "confusion_matrix_path": str(confusion_matrix_path),
        "predictions_path": str(predictions_path),
        "model_path": str(model_path),
        "report_path": str(report_path),
    }

    return result_summary, k_results_df


# ============================================================
# 5. Main
# ============================================================

def main():
    print("\nLoading features...")

    df = pd.read_csv(FEATURES_CSV)

    print("\nColumns in features_all.csv:")
    print(df.columns.tolist())

    check_required_columns(df)

    df = clean_diagnostics(df)

    print(f"\nTotal valid samples: {len(df)}")
    print(f"Malignant (1):      {int(df['label'].sum())}")
    print(f"Benign    (0):      {int((df['label'] == 0).sum())}")

    summaries = []
    all_k_results = []

    for feature_set_name, feature_set_info in FEATURE_SETS.items():
        summary, k_results = run_one_knn_model(
            df=df,
            feature_set_name=feature_set_name,
            feature_set_info=feature_set_info,
        )

        summaries.append(summary)
        all_k_results.append(k_results)

    # Save combined comparison summary
    summary_df = pd.DataFrame(summaries)

    summary_path = REPORTS_DIR / "knn_feature_set_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved combined summary: {summary_path}")

    # Save combined k-search results
    all_k_results_df = pd.concat(all_k_results, ignore_index=True)

    all_k_results_path = REPORTS_DIR / "knn_feature_set_comparison_k_search.csv"
    all_k_results_df.to_csv(all_k_results_path, index=False)

    print(f"Saved combined k-search results: {all_k_results_path}")

    print("\nFinal comparison:")
    print(
        summary_df[
            [
                "feature_set",
                "n_features",
                "best_k",
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
