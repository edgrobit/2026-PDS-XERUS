"""
Decision Tree Model — Skin Lesion Malignancy Classification
===========================================================

This script follows the same project structure as the KNN and Logistic Regression models.

It uses:
    data/Separate csv of features/features_all.csv

Target:
    1 = malignant: BCC, SCC, MEL
    0 = benign: ACK, NEV, SEK

The script automatically ignores:
    - image identifiers
    - labels/diagnosis columns
    - dominant color columns beginning with dom_color_
    - rotations_used, because it is a processing artifact, not a lesion feature

Outputs:
    results/models/decision_tree_model.pkl
    results/predictions/decision_tree_predictions.csv
    results/figures/decision_tree_confusion_matrix.png
    results/figures/decision_tree_plot.png
    results/figures/decision_tree_feature_importance.png
    results/reports/decision_tree_report.txt
    results/reports/decision_tree_feature_importances.csv
    results/reports/decision_tree_grid_search.csv

Run:
    python DecisionTree.py

or, if the file is inside src/:
    python src/DecisionTree.py
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

# If this script is inside src/, the project root is one folder above.
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

# Hyperparameters to test.
# Decision trees can overfit easily, so we search for a simpler tree.
MAX_DEPTH_VALUES = [2, 3, 4, 5, 6, 8, 10, None]
MIN_SAMPLES_LEAF_VALUES = [1, 2, 5, 10, 20]
CRITERION_VALUES = ["gini", "entropy"]


# ============================================================
# 3. Helper functions
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


def select_feature_columns(df):
    """
    Selects usable numeric feature columns from features_all.csv.

    Dominant color columns are ignored automatically.
    """

    dominant_color_cols = [
        col for col in df.columns
        if col.startswith("dom_color_")
    ]

    non_feature_cols = {
        "image_name",
        "mask_name",
        "img_id",
        "stem",
        "diagnostic",
        "diagnosis",
        "label",
        "target",
        "rotations_used",
    }

    feature_cols = [
        col for col in df.columns
        if col not in non_feature_cols
        and col not in dominant_color_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    return feature_cols, dominant_color_cols


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


# ============================================================
# 4. Load data
# ============================================================

print("\nLoading data...")

df = pd.read_csv(FEATURES_CSV)

print("\nColumns in features_all.csv:")
print(df.columns.tolist())

if "image_name" not in df.columns:
    raise ValueError("The column 'image_name' was not found in features_all.csv.")

df = add_diagnostic_if_missing(df)

df["diagnostic"] = df["diagnostic"].astype(str).str.strip().str.upper()

# Keep only known diagnostic groups.
df = df[df["diagnostic"].isin(VALID_DIAGNOSTICS)].copy()

if df.empty:
    raise ValueError(
        "After filtering diagnostics, no rows remained. "
        "Check whether diagnostic values are BCC, SCC, MEL, ACK, NEV, or SEK."
    )

# Binary target:
# 1 = malignant
# 0 = benign
df["label"] = df["diagnostic"].apply(lambda x: 1 if x in MALIGNANT else 0)

print(f"\nTotal labelled samples: {len(df)}")
print(f"Malignant (1):         {df['label'].sum()}")
print(f"Benign    (0):         {(df['label'] == 0).sum()}")


# ============================================================
# 5. Select features
# ============================================================

FEATURE_COLS, dominant_color_cols = select_feature_columns(df)

print("\nDominant color columns ignored:")
print(dominant_color_cols)

print("\nFeatures used:")
for feature in FEATURE_COLS:
    print("-", feature)

if not FEATURE_COLS:
    raise ValueError("No usable numeric feature columns found.")

df_model = df[["image_name", "diagnostic", "label"] + FEATURE_COLS].copy()

# Replace infinite values with missing values.
df_model = df_model.replace([np.inf, -np.inf], np.nan)

# Drop rows with missing feature values.
df_model = df_model.dropna(subset=FEATURE_COLS + ["label"])

print(f"\nRows after dropping missing values: {len(df_model)}")

if df_model["label"].nunique() < 2:
    raise ValueError("The modelling data contains only one class. Decision tree needs both classes.")

X = df_model[FEATURE_COLS]
y = df_model["label"]


# ============================================================
# 6. Train/test split
# ============================================================

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


# ============================================================
# 7. Cross-validation hyperparameter search
# ============================================================

print("\nSearching for best Decision Tree parameters...")

min_class_count_train = y_train.value_counts().min()
cv_folds = min(5, int(min_class_count_train))

if cv_folds < 2:
    raise ValueError("Not enough samples per class for cross-validation.")

cv = StratifiedKFold(
    n_splits=cv_folds,
    shuffle=True,
    random_state=RANDOM_STATE,
)

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

grid_search_path = REPORTS_DIR / "decision_tree_grid_search.csv"
grid_results_df.to_csv(grid_search_path, index=False)

best_row = grid_results_df.sort_values("cv_auc_mean", ascending=False).iloc[0]

best_criterion = best_row["criterion"]
best_max_depth = None if pd.isna(best_row["max_depth"]) else best_row["max_depth"]
best_min_samples_leaf = int(best_row["min_samples_leaf"])

# Pandas may read max_depth as float because of None values.
if best_max_depth is not None:
    best_max_depth = int(best_max_depth)

print("\nBest parameters:")
print(f"criterion:        {best_criterion}")
print(f"max_depth:        {best_max_depth}")
print(f"min_samples_leaf: {best_min_samples_leaf}")
print(f"Best CV ROC-AUC:  {best_row['cv_auc_mean']:.3f}")


# ============================================================
# 8. Train final Decision Tree model
# ============================================================

print("\nTraining final Decision Tree model...")

model = make_decision_tree(
    max_depth=best_max_depth,
    min_samples_leaf=best_min_samples_leaf,
    criterion=best_criterion,
)

model.fit(X_train, y_train)


# ============================================================
# 9. Evaluate on test set
# ============================================================

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


# ============================================================
# 10. Confusion matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    cm,
    display_labels=["Benign", "Malignant"],
).plot(ax=ax, colorbar=False)

ax.set_title("Decision Tree Confusion Matrix")
plt.tight_layout()

confusion_matrix_path = FIGURES_DIR / "decision_tree_confusion_matrix.png"
plt.savefig(confusion_matrix_path, dpi=150)
plt.close()

print(f"\nSaved: {confusion_matrix_path}")


# ============================================================
# 11. Plot decision tree
# ============================================================

tree_plot_path = FIGURES_DIR / "decision_tree_plot.png"

plt.figure(figsize=(24, 12))
plot_tree(
    model,
    feature_names=FEATURE_COLS,
    class_names=["Benign", "Malignant"],
    filled=True,
    rounded=True,
    max_depth=4,
    fontsize=8,
)
plt.title("Decision Tree Plot, truncated to depth 4")
plt.tight_layout()
plt.savefig(tree_plot_path, dpi=150)
plt.close()

print(f"Saved: {tree_plot_path}")


# ============================================================
# 12. Feature importance
# ============================================================

importance_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importances_,
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False,
).reset_index(drop=True)

feature_importance_path = REPORTS_DIR / "decision_tree_feature_importances.csv"
importance_df.to_csv(feature_importance_path, index=False)

print(f"Saved: {feature_importance_path}")

# Plot feature importance
feature_importance_fig_path = FIGURES_DIR / "decision_tree_feature_importance.png"

top_n = min(15, len(importance_df))
plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

plt.figure(figsize=(8, 6))
plt.barh(plot_df["feature"], plot_df["importance"])
plt.xlabel("Feature importance")
plt.title("Decision Tree Feature Importance")
plt.tight_layout()
plt.savefig(feature_importance_fig_path, dpi=150)
plt.close()

print(f"Saved: {feature_importance_fig_path}")


# ============================================================
# 13. Save predictions
# ============================================================

pred_df = df_model.loc[idx_test].copy().reset_index(drop=True)

pred_df["predicted_label"] = y_pred
pred_df["predicted_prob_malignant"] = y_pred_prob.round(4)
pred_df["correct"] = y_pred == y_test

predictions_path = PREDICTIONS_DIR / "decision_tree_predictions.csv"
pred_df.to_csv(predictions_path, index=False)

print(f"Saved: {predictions_path}")


# ============================================================
# 14. Save model
# ============================================================

model_path = MODELS_DIR / "decision_tree_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump({
        "model": model,
        "features": FEATURE_COLS,
        "best_criterion": best_criterion,
        "best_max_depth": best_max_depth,
        "best_min_samples_leaf": best_min_samples_leaf,
        "malignant_classes": sorted(MALIGNANT),
        "benign_classes": sorted(BENIGN),
    }, f)

print(f"Saved: {model_path}")


# ============================================================
# 15. Save text report
# ============================================================

report_path = REPORTS_DIR / "decision_tree_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("Decision Tree Classification Report\n")
    f.write("=" * 42)
    f.write("\n\n")

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

    f.write("Dominant color columns ignored:\n")
    if dominant_color_cols:
        for col in dominant_color_cols:
            f.write(f"- {col}\n")
    else:
        f.write("None found.\n")

    f.write("\nFeatures used:\n")
    for feature in FEATURE_COLS:
        f.write(f"- {feature}\n")

    f.write("\nDataset:\n")
    f.write(f"Total labelled samples after cleaning: {len(df_model)}\n")
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

print("\nDone.")
