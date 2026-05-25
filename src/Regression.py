"""
Logistic Regression Model — Skin Lesion Malignancy Classification
=================================================================

This script follows the same project structure as the KNN model.

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
    results/models/logistic_regression_model.pkl
    results/predictions/logistic_regression_predictions.csv
    results/figures/logistic_regression_confusion_matrix.png
    results/figures/logistic_regression_roc_curve.png
    results/reports/logistic_regression_report.txt
    results/reports/logistic_regression_coefficients.csv

Run:
    python logistic_regression.py

or, if the file is inside src/:
    python src/logistic_regression.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
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

# Values tested for the regularization strength.
# Smaller C = stronger regularization.
C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]


# ============================================================
# 3. Helper functions
# ============================================================

def clean_stem(series):
    """
    Creates comparable image stems by removing common extensions.
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

    This keeps the script compatible with both structures:
    1. features_all.csv already contains diagnostic
    2. diagnostic is stored only in annotations_combined.csv
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


def make_logistic_model(C):
    """
    Creates a Logistic Regression pipeline.

    StandardScaler is important because the features have different scales:
    compactness, area, RGB variance, ITA, etc.
    """

    return Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(
            C=C,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )),
    ])


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
    raise ValueError("The modelling data contains only one class. Logistic regression needs both classes.")

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
# 7. Cross-validation to select C
# ============================================================

print("\nSearching for best regularization C...")

min_class_count_train = y_train.value_counts().min()
cv_folds = min(5, int(min_class_count_train))

if cv_folds < 2:
    raise ValueError("Not enough samples per class for cross-validation.")

cv = StratifiedKFold(
    n_splits=cv_folds,
    shuffle=True,
    random_state=RANDOM_STATE,
)

c_results = []

for C in C_VALUES:
    model_c = make_logistic_model(C)

    auc_scores = cross_val_score(
        model_c,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
    )

    acc_scores = cross_val_score(
        model_c,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
    )

    c_results.append({
        "C": C,
        "cv_auc_mean": auc_scores.mean(),
        "cv_auc_std": auc_scores.std(),
        "cv_accuracy_mean": acc_scores.mean(),
        "cv_accuracy_std": acc_scores.std(),
    })

    print(
        f"C={C:<7} | "
        f"CV ROC-AUC={auc_scores.mean():.3f} ± {auc_scores.std():.3f} | "
        f"CV Accuracy={acc_scores.mean():.3f} ± {acc_scores.std():.3f}"
    )

c_results_df = pd.DataFrame(c_results)
c_results_path = REPORTS_DIR / "logistic_regression_c_search.csv"
c_results_df.to_csv(c_results_path, index=False)

best_row = c_results_df.sort_values("cv_auc_mean", ascending=False).iloc[0]
best_C = float(best_row["C"])

print(f"\nBest C: {best_C}")
print(f"Best CV ROC-AUC: {best_row['cv_auc_mean']:.3f}")


# ============================================================
# 8. Train final Logistic Regression model
# ============================================================

print("\nTraining final Logistic Regression model...")

model = make_logistic_model(best_C)
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

ax.set_title("Logistic Regression Confusion Matrix")
plt.tight_layout()

confusion_matrix_path = FIGURES_DIR / "logistic_regression_confusion_matrix.png"
plt.savefig(confusion_matrix_path, dpi=150)
plt.close()

print(f"\nSaved: {confusion_matrix_path}")


# ============================================================
# 11. ROC curve
# ============================================================

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(
    y_test,
    y_pred_prob,
    ax=ax,
)

ax.set_title("Logistic Regression ROC Curve")
plt.tight_layout()

roc_curve_path = FIGURES_DIR / "logistic_regression_roc_curve.png"
plt.savefig(roc_curve_path, dpi=150)
plt.close()

print(f"Saved: {roc_curve_path}")


# ============================================================
# 12. Save predictions
# ============================================================

pred_df = df_model.loc[idx_test].copy().reset_index(drop=True)

pred_df["predicted_label"] = y_pred
pred_df["predicted_prob_malignant"] = y_pred_prob.round(4)
pred_df["correct"] = y_pred == y_test

predictions_path = PREDICTIONS_DIR / "logistic_regression_predictions.csv"
pred_df.to_csv(predictions_path, index=False)

print(f"Saved: {predictions_path}")


# ============================================================
# 13. Save coefficients
# ============================================================

logistic_step = model.named_steps["logistic"]

coef_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "coefficient": logistic_step.coef_[0],
})

coef_df["abs_coefficient"] = coef_df["coefficient"].abs()

coef_df = coef_df.sort_values(
    "abs_coefficient",
    ascending=False,
).reset_index(drop=True)

coefficients_path = REPORTS_DIR / "logistic_regression_coefficients.csv"
coef_df.to_csv(coefficients_path, index=False)

print(f"Saved: {coefficients_path}")


# ============================================================
# 14. Save model
# ============================================================

model_path = MODELS_DIR / "logistic_regression_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump({
        "model": model,
        "features": FEATURE_COLS,
        "best_C": best_C,
        "malignant_classes": sorted(MALIGNANT),
        "benign_classes": sorted(BENIGN),
    }, f)

print(f"Saved: {model_path}")


# ============================================================
# 15. Save text report
# ============================================================

report_path = REPORTS_DIR / "logistic_regression_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("Logistic Regression Classification Report\n")
    f.write("=" * 48)
    f.write("\n\n")

    f.write(f"Features CSV: {FEATURES_CSV}\n")
    f.write(f"Test size: {TEST_SIZE}\n")
    f.write(f"Random state: {RANDOM_STATE}\n")
    f.write(f"Best C: {best_C}\n")
    f.write("Class weight: balanced\n\n")

    f.write("Diagnostic mapping:\n")
    f.write("1 = malignant: BCC, SCC, MEL\n")
    f.write("0 = benign: ACK, NEV, SEK\n\n")

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

    f.write("\nCross-validation C search:\n")
    f.write(c_results_df.to_string(index=False))
    f.write("\n\n")

    f.write("Test performance:\n")
    f.write(f"Accuracy: {accuracy:.3f}\n")
    f.write(f"ROC-AUC: {roc_auc:.3f}\n\n")

    f.write("Classification report:\n")
    f.write(report)
    f.write("\n\n")

    f.write("Standardized coefficients, sorted by absolute value:\n")
    f.write(coef_df.to_string(index=False))

print(f"Saved: {report_path}")

print("\nDone.")
