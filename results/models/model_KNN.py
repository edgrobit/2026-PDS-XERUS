"""
model_KNN.py — KNN classifier
==============================
KNN classifies by finding the k nearest neighbours in feature space
and taking a majority vote. It is distance-based, so StandardScaler
is essential — without it, large-valued features (ITA ~20) would
dominate small-valued ones (asymmetry ~0.3).

Three feature sets are compared to directly answer the research question:
    A — Shape only (baseline, no color)
    B — Shape + raw color variance + ITA
    C — Shape + color + FST (full model)

Outputs:
    results/models/knn_model_<name>.pkl
    results/predictions/knn_predictions_<name>.csv
    results/figures/knn_*.png
    results/reports/knn_report.txt
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score,
)

# ── Configuration ─────────────────────────────────────────────────────────────
FEATURES_CSV = Path("data/features.csv")
MALIGNANT    = {"BCC", "SCC", "MEL"}
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# Three feature sets — progressive addition to isolate skin color contribution
FEATURE_SETS = {
    "A_Baseline": [
        # Shape only — no color information
        "asymmetry_score", "compactness", "lesion_percentage",
    ],
    "B_PlusColor": [
        # Shape + color heterogeneity (how varied the color is across the lesion)
        "asymmetry_score", "compactness", "lesion_percentage",
        "rgb_var_r", "rgb_var_g", "rgb_var_b",
        "hsv_var_h", "hsv_var_s", "hsv_var_v",
    ],
    "C_PlusITA": [
        # Shape + color heterogeneity + ITA (continuous skin tone)
        # This phase directly addresses the research question
        "asymmetry_score", "compactness", "lesion_percentage",
        "rgb_var_r", "rgb_var_g", "rgb_var_b",
        "hsv_var_h", "hsv_var_s", "hsv_var_v",
        "ita_mean",
    ],
}

for d in ["results/models", "results/predictions",
          "results/figures", "results/reports"]:
    Path(d).mkdir(parents=True, exist_ok=True)


def run(features_csv=FEATURES_CSV):
    print("\n" + "="*50)
    print("KNN — PROGRESSIVE FEATURE COMPARISON")
    print("="*50)

    df = pd.read_csv(features_csv)
    df["label"] = df["diagnostic"].apply(lambda x: 1 if x in MALIGNANT else 0)

    print(f"  Total: {len(df)} | Malignant: {df['label'].sum()} "
          f"| Benign: {(df['label']==0).sum()}")

    results = {}

    for model_name, feature_cols in FEATURE_SETS.items():
        print(f"\n  --- {model_name} ---")

        available = [c for c in feature_cols if c in df.columns]
        missing   = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"    WARNING: skipping missing: {missing}")

        df_model = df[["img_id", "diagnostic", "label"] + available].dropna()

        X = df_model[available].values
        y = df_model["label"].values

        positions = np.arange(len(df_model))
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, positions,
            test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Find best k on training set only
        k_scores = []
        for k in range(1, 21):
            score = cross_val_score(
                KNeighborsClassifier(n_neighbors=k, metric="euclidean"),
                X_train, y_train, cv=5, scoring="roc_auc"
            ).mean()
            k_scores.append((k, score))
        best_k, best_cv = max(k_scores, key=lambda x: x[1])

        knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        y_prob = knn.predict_proba(X_test)[:, 1]
        accuracy = (y_pred == y_test).mean()
        roc_auc  = roc_auc_score(y_test, y_prob)
        report   = classification_report(y_test, y_pred,
                       target_names=["Benign", "Malignant"])

        print(f"    k={best_k}  acc={accuracy:.3f}  AUC={roc_auc:.3f}")

        results[model_name] = {
            "features":  available,
            "best_k":    best_k,
            "best_cv":   best_cv,
            "accuracy":  accuracy,
            "roc_auc":   roc_auc,
            "report":    report,
            "y_pred":    y_pred,
            "y_test":    y_test,
            "k_scores":  k_scores,
        }

        # Save predictions and model per feature set
        pred_df = df_model.iloc[idx_test].copy().reset_index(drop=True)
        pred_df["predicted_label"]    = y_pred
        pred_df["predicted_prob_mal"] = y_prob.round(4)
        pred_df["correct"]            = (y_pred == y_test)
        pred_df.to_csv(f"results/predictions/knn_predictions_{model_name}.csv",
                       index=False)

        with open(f"results/models/knn_model_{model_name}.pkl", "wb") as f:
            pickle.dump({"model": knn, "scaler": scaler,
                         "features": available, "k": best_k}, f)

    # ── Comparison summary ────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    labels = {
        "A_Baseline":  "A: Shape only",
        "B_PlusColor": "B: + Color variance",
        "C_PlusITA":   "C: + ITA (skin tone)",
    }
    base_acc = results["A_Baseline"]["accuracy"]
    base_auc = results["A_Baseline"]["roc_auc"]

    for name, r in results.items():
        d_acc = f"  Δacc={r['accuracy']-base_acc:+.3f}" if name != "A_Baseline" else ""
        d_auc = f"  Δauc={r['roc_auc']-base_auc:+.3f}"  if name != "A_Baseline" else ""
        print(f"  {labels[name]:<20} acc={r['accuracy']:.3f}  "
              f"AUC={r['roc_auc']:.3f}{d_acc}{d_auc}")

    # ── Figures ───────────────────────────────────────────────────────────────
    names  = list(results.keys())
    colors = ["#95a5a6", "#4C72B0", "#DD8452"]
    xlabels = [labels[n] for n in names]

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    accs = [results[n]["accuracy"] for n in names]
    aucs = [results[n]["roc_auc"]  for n in names]

    for ax, vals, metric in zip(axes, [accs, aucs], ["Accuracy", "ROC-AUC"]):
        bars = ax.bar(xlabels, vals, color=colors, alpha=0.85)
        ax.axhline(vals[0], color="gray", linestyle="--", alpha=0.6,
                   label=f"Baseline {metric} ({vals[0]:.3f})")
        ax.set_ylim(0.55, 0.72)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Feature Set")
        ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + 0.005, f"{v:.3f}", ha="center",
                    fontsize=9, fontweight="bold")

    plt.suptitle("KNN — Does skin color improve malignancy prediction?",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/knn_comparison.png", dpi=150)
    plt.close()

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, color) in zip(axes, zip(names, colors)):
        cm = confusion_matrix(results[name]["y_test"], results[name]["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["Benign","Malignant"]).plot(
            ax=ax, colorbar=False)
        ax.set_title(f"{labels[name]}\nacc={results[name]['accuracy']:.3f}")
    plt.suptitle("KNN Confusion Matrices", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/knn_confusion_matrices.png", dpi=150)
    plt.close()

    # ── Save report ───────────────────────────────────────────────────────────
    with open("results/reports/knn_report.txt", "w") as f:
        f.write("KNN Classification Report\n" + "="*50 + "\n")
        f.write("Research question: Does skin color influence malignancy?\n\n")
        for name, r in results.items():
            f.write(f"Model {labels[name]}\n")
            f.write(f"  Features: {r['features']}\n")
            f.write(f"  Best k:   {r['best_k']}\n")
            f.write(f"  Acc:      {r['accuracy']:.3f}\n")
            f.write(f"  AUC:      {r['roc_auc']:.3f}\n\n")
            f.write(r["report"] + "\n" + "-"*50 + "\n\n")

    print("\n  Saved: results/figures/knn_comparison.png")
    print("  Saved: results/figures/knn_confusion_matrices.png")
    print("  Saved: results/reports/knn_report.txt")

    return results


if __name__ == "__main__":
    run()
