"""
KNN Model — Skin Lesion Malignancy Classification
=================================================

This script uses only features_all.csv.

It ignores dominant color columns:
- dom_color_1_ratio
- dom_color_1_h
- dom_color_1_s
- dom_color_1_v
- ...
- dom_color_5_v

Target:
    1 = malignant: BCC, SCC, MEL
    0 = benign: ACK, NEV, SEK

Run:
    python KNN.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
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

# If the script is inside src/, project root is one folder above
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

K = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ============================================================
# 3. Load data
# ============================================================

print("\nLoading features...")

df = pd.read_csv(FEATURES_CSV)

print("\nColumns in features_all.csv:")
print(df.columns.tolist())

if "diagnostic" not in df.columns:
    raise ValueError(
        "The column 'diagnostic' was not found in features_all.csv. "
        "The model needs this column to create the malignant/benign label."
    )

if "image_name" not in df.columns:
    raise ValueError(
        "The column 'image_name' was not found in features_all.csv."
    )

df["diagnostic"] = df["diagnostic"].astype(str).str.strip().str.upper()

df["label"] = df["diagnostic"].apply(
    lambda x: 1 if x in MALIGNANT else 0
)

print(f"\nTotal samples: {len(df)}")
print(f"Malignant (1): {df['label'].sum()}")
print(f"Benign    (0): {(df['label'] == 0).sum()}")


# ============================================================
# 4. Select feature columns
# ============================================================

# Columns that are identifiers or labels, not model features
NON_FEATURE_COLS = {
    "image_name",
    "mask_name",
    "diagnostic",
    "label",
}

# Ignore dominant color columns
dominant_color_cols = [
    col for col in df.columns
    if col.startswith("dom_color_")
]

# Use only numerical columns that are not identifiers, labels, or dominant colors
FEATURE_COLS = [
    col for col in df.columns
    if col not in NON_FEATURE_COLS
    and col not in dominant_color_cols
    and pd.api.types.is_numeric_dtype(df[col])
]

print("\nDominant color columns ignored:")
print(dominant_color_cols)

print("\nFeatures used in the model:")
print(FEATURE_COLS)

if not FEATURE_COLS:
    raise ValueError("No usable numerical feature columns found.")


# ============================================================
# 5. Prepare modelling dataset
# ============================================================

df_model = df[["image_name", "diagnostic", "label"] + FEATURE_COLS].copy()

# Replace infinite values with NaN
df_model = df_model.replace([np.inf, -np.inf], np.nan)

# Drop rows with missing values in the selected features
df_model = df_model.dropna(subset=FEATURE_COLS + ["label"])

print(f"\nRows after dropping missing values: {len(df_model)}")

X = df_model[FEATURE_COLS].values
y = df_model["label"].values


# ============================================================
# 6. Train/test split
# ============================================================

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X,
    y,
    df_model.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTrain samples: {len(X_train)}")
print(f"Test samples:  {len(X_test)}")


# ============================================================
# 7. Scale features
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 8. Train KNN
# ============================================================

print(f"\nTraining KNN with k={K}...")

knn = KNeighborsClassifier(
    n_neighbors=K,
    metric="euclidean"
)

knn.fit(X_train_scaled, y_train)


# ============================================================
# 9. Cross-validation
# ============================================================

cv_scores = cross_val_score(
    knn,
    X_train_scaled,
    y_train,
    cv=5,
    scoring="accuracy"
)

print(f"\n5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


# ============================================================
# 10. Test evaluation
# ============================================================

y_pred = knn.predict(X_test_scaled)
y_pred_prob = knn.predict_proba(X_test_scaled)[:, 1]

accuracy = (y_pred == y_test).mean()
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"\nTest accuracy: {accuracy:.3f}")
print(f"ROC-AUC:       {roc_auc:.3f}")

report = classification_report(
    y_test,
    y_pred,
    target_names=["Benign", "Malignant"]
)

print("\nClassification report:")
print(report)


# ============================================================
# 11. Confusion matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay(
    cm,
    display_labels=["Benign", "Malignant"]
).plot(ax=ax, colorbar=False)

ax.set_title(f"KNN Confusion Matrix, k={K}")

plt.tight_layout()

confusion_matrix_path = FIGURES_DIR / "knn_confusion_matrix.png"
plt.savefig(confusion_matrix_path, dpi=150)
plt.close()

print(f"\nSaved: {confusion_matrix_path}")


# ============================================================
# 12. Find optimal k
# ============================================================

print("\nSearching for optimal k from 1 to 20...")

k_scores = []

for k in range(1, 21):
    knn_k = KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean"
    )

    score = cross_val_score(
        knn_k,
        X_train_scaled,
        y_train,
        cv=5,
        scoring="accuracy"
    ).mean()

    k_scores.append((k, score))

    print(f"k={k:2d} | CV accuracy={score:.3f}")

best_k, best_score = max(k_scores, key=lambda x: x[1])

print(f"\nBest k: {best_k}")
print(f"Best CV accuracy: {best_score:.3f}")


# ============================================================
# 13. Plot k search
# ============================================================

ks, scores = zip(*k_scores)

plt.figure(figsize=(8, 4))
plt.plot(ks, scores, marker="o")
plt.axvline(best_k, linestyle="--", label=f"Best k={best_k}")

plt.xlabel("k")
plt.ylabel("5-fold CV accuracy")
plt.title("KNN — Accuracy by k")
plt.legend()
plt.tight_layout()

k_search_path = FIGURES_DIR / "knn_k_search.png"
plt.savefig(k_search_path, dpi=150)
plt.close()

print(f"Saved: {k_search_path}")


# ============================================================
# 14. Save predictions
# ============================================================

pred_df = df_model.loc[idx_test].copy().reset_index(drop=True)

pred_df["predicted_label"] = y_pred
pred_df["predicted_prob_malignant"] = y_pred_prob.round(4)
pred_df["correct"] = y_pred == y_test

predictions_path = PREDICTIONS_DIR / "knn_predictions.csv"
pred_df.to_csv(predictions_path, index=False)

print(f"Saved: {predictions_path}")


# ============================================================
# 15. Save model
# ============================================================

model_path = MODELS_DIR / "knn_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(
        {
            "model": knn,
            "scaler": scaler,
            "features": FEATURE_COLS,
            "k": K,
        },
        f
    )

print(f"Saved: {model_path}")


# ============================================================
# 16. Save report
# ============================================================

report_path = REPORTS_DIR / "knn_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("KNN Classification Report\n")
    f.write("=" * 40)
    f.write("\n\n")

    f.write(f"Features CSV: {FEATURES_CSV}\n")
    f.write(f"Dominant color columns ignored: {dominant_color_cols}\n\n")

    f.write("Features used:\n")
    for feature in FEATURE_COLS:
        f.write(f"- {feature}\n")

    f.write("\nModel settings:\n")
    f.write(f"k: {K}\n")
    f.write(f"Test size: {TEST_SIZE}\n")
    f.write(f"Random state: {RANDOM_STATE}\n")

    f.write("\nDataset:\n")
    f.write(f"Total samples: {len(df_model)}\n")
    f.write(f"Train samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n")
    f.write(f"Malignant samples: {df_model['label'].sum()}\n")
    f.write(f"Benign samples: {(df_model['label'] == 0).sum()}\n")

    f.write("\nPerformance:\n")
    f.write(f"5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n")
    f.write(f"Test accuracy: {accuracy:.3f}\n")
    f.write(f"ROC-AUC: {roc_auc:.3f}\n")
    f.write(f"Best k: {best_k}\n")
    f.write(f"Best k CV accuracy: {best_score:.3f}\n")

    f.write("\nClassification report:\n")
    f.write(report)

print(f"Saved: {report_path}")

print("\nDone.")