"""
model_logistic_regression.py — Logistic Regression classifier
==============================================================
Logistic Regression estimates the probability of malignancy as a weighted
sum of features passed through a sigmoid function. It assumes a linear
relationship between features and log-odds of malignancy.

Key advantage: the coefficient for each feature directly shows its direction
and strength — positive coefficient = feature increases malignancy probability,
negative = decreases it. This makes LR uniquely interpretable for your
research question: the ITA coefficient tells you directly whether lighter
skin (higher ITA) is associated with higher or lower malignancy likelihood.

Three feature sets compared (same as KNN and DT for cross-model comparison):
    A_Baseline  — Shape only
    B_PlusColor — Shape + color variance + ITA
    C_PlusFST   — Shape + color + FST (full model)

Outputs:
    results/models/lr_model_<name>.pkl
    results/predictions/lr_predictions_<name>.csv
    results/figures/lr_confusion_matrices.png
    results/figures/lr_coefficients.png
    results/figures/lr_comparison.png
    results/reports/lr_report.txt
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score,
)

# ── Configuration ─────────────────────────────────────────────────────────────
FEATURES_CSV = Path("data/features.csv")
MALIGNANT    = {"BCC", "SCC", "MEL"}
TEST_SIZE    = 0.2
RANDOM_STATE = 42

FEATURE_SETS = {
    "A_Baseline": [
        "asymmetry_score", "compactness", "lesion_percentage",
    ],
    "B_PlusColor": [
        "asymmetry_score", "compactness", "lesion_percentage",
        "rgb_var_r", "rgb_var_g", "rgb_var_b",
        "hsv_var_h", "hsv_var_s", "hsv_var_v",
        "ita_mean",
    ],
    "C_PlusFST": [
        "asymmetry_score", "compactness", "lesion_percentage",
        "rgb_var_r", "rgb_var_g", "rgb_var_b",
        "hsv_var_h", "hsv_var_s", "hsv_var_v",
        "ita_mean", "fst_predicted",
    ],
}

for d in ["results/models", "results/predictions",
          "results/figures", "results/reports"]:
    Path(d).mkdir(parents=True, exist_ok=True)


def run(features_csv=FEATURES_CSV):
    print("\n" + "="*50)
    print("LOGISTIC REGRESSION — PROGRESSIVE FEATURE COMPARISON")
    print("="*50)

    df = pd.read_csv(features_csv)
    df["label"] = df["diagnostic"].apply(lambda x: 1 if x in MALIGNANT else 0)
    print(f"  Total: {len(df)} | Malignant: {df['label'].sum()} "
          f"| Benign: {(df['label']==0).sum()}")

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

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

        # Pipeline: scale then fit — prevents data leakage inside CV folds
        model = Pipeline([
            ("scaler",   StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1000,
                                            random_state=RANDOM_STATE)),
        ])

        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv, scoring="roc_auc")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        accuracy = (y_pred == y_test).mean()
        roc_auc  = roc_auc_score(y_test, y_prob)
        report   = classification_report(y_test, y_pred,
                       target_names=["Benign", "Malignant"])

        # Coefficients — unique to LR, directly interpretable
        coefs  = model.named_steps["logistic"].coef_[0]
        coef_df = pd.DataFrame({
            "feature":     available,
            "coefficient": coefs.round(4),
            "odds_ratio":  np.exp(coefs).round(4),
        }).sort_values("coefficient", key=abs, ascending=False)

        print(f"    acc={accuracy:.3f}  AUC={roc_auc:.3f}  "
              f"CV AUC={cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        results[model_name] = {
            "features":  available,
            "cv_auc":    cv_scores.mean(),
            "cv_std":    cv_scores.std(),
            "accuracy":  accuracy,
            "roc_auc":   roc_auc,
            "report":    report,
            "y_pred":    y_pred,
            "y_test":    y_test,
            "coefs":     coef_df,
        }

        # Save predictions and model
        pred_df = df_model.iloc[idx_test].copy().reset_index(drop=True)
        pred_df["predicted_label"]    = y_pred
        pred_df["predicted_prob_mal"] = y_prob.round(4)
        pred_df["correct"]            = (y_pred == y_test)
        pred_df.to_csv(f"results/predictions/lr_predictions_{model_name}.csv",
                       index=False)

        with open(f"results/models/lr_model_{model_name}.pkl", "wb") as f:
            pickle.dump({"model": model, "features": available}, f)

    # ── Comparison summary ────────────────────────────────────────────────────
    names    = list(results.keys())
    labels   = {"A_Baseline":  "A: Shape only",
                 "B_PlusColor": "B: + Color/ITA",
                 "C_PlusFST":   "C: + FST"}
    colors   = ["#95a5a6", "#4C72B0", "#DD8452"]
    base_acc = results["A_Baseline"]["accuracy"]
    base_auc = results["A_Baseline"]["roc_auc"]

    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    for name, r in results.items():
        d_acc = f"  Δacc={r['accuracy']-base_acc:+.3f}" if name != "A_Baseline" else ""
        d_auc = f"  Δauc={r['roc_auc']-base_auc:+.3f}"  if name != "A_Baseline" else ""
        print(f"  {labels[name]:<22} acc={r['accuracy']:.3f}  "
              f"AUC={r['roc_auc']:.3f}{d_acc}{d_auc}")

    # ── Figures ───────────────────────────────────────────────────────────────

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, color) in zip(axes, zip(names, colors)):
        cm = confusion_matrix(results[name]["y_test"], results[name]["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["Benign","Malignant"]).plot(
            ax=ax, colorbar=False)
        ax.set_title(f"{labels[name]}\nacc={results[name]['accuracy']:.3f}")
    plt.suptitle("Logistic Regression Confusion Matrices", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/lr_confusion_matrices.png", dpi=150)
    plt.close()

    # Coefficients for full model (C_PlusFST) — most interpretable figure
    coef_df = results["C_PlusFST"]["coefs"]
    colors_coef = ["#DD8452" if v > 0 else "#4C72B0"
                   for v in coef_df["coefficient"]]
    plt.figure(figsize=(8, 5))
    plt.barh(coef_df["feature"], coef_df["coefficient"], color=colors_coef)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Coefficient (positive = increases malignancy probability)")
    plt.title("Logistic Regression Coefficients (Full model)\n"
              "Orange = positive effect, Blue = negative effect")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("results/figures/lr_coefficients.png", dpi=150)
    plt.close()

    # Bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    accs = [results[n]["accuracy"] for n in names]
    aucs = [results[n]["roc_auc"]  for n in names]
    xlabels = [labels[n] for n in names]

    for ax, vals, metric in zip(axes, [accs, aucs], ["Accuracy", "ROC-AUC"]):
        bars = ax.bar(xlabels, vals, color=colors, alpha=0.85)
        ax.axhline(vals[0], color="gray", linestyle="--", alpha=0.6,
                   label=f"Baseline ({vals[0]:.3f})")
        ax.set_ylim(0.50, 0.80)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Feature Set")
        ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + 0.005, f"{v:.3f}", ha="center",
                    fontsize=9, fontweight="bold")

    plt.suptitle("Logistic Regression — Does skin color improve prediction?",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/lr_comparison.png", dpi=150)
    plt.close()

    # ── Save report ───────────────────────────────────────────────────────────
    with open("results/reports/lr_report.txt", "w") as f:
        f.write("Logistic Regression Report\n" + "="*50 + "\n")
        f.write("Research question: Does skin color influence malignancy?\n\n")
        for name, r in results.items():
            f.write(f"Model {labels[name]}\n")
            f.write(f"  Features: {r['features']}\n")
            f.write(f"  CV AUC:   {r['cv_auc']:.3f} ± {r['cv_std']:.3f}\n")
            f.write(f"  Acc:      {r['accuracy']:.3f}\n")
            f.write(f"  AUC:      {r['roc_auc']:.3f}\n\n")
            f.write(r["report"] + "\n")
            f.write("Coefficients (sorted by magnitude):\n")
            f.write(r["coefs"].to_string(index=False) + "\n")
            f.write("-"*50 + "\n\n")

    print("\n  Saved: results/figures/lr_confusion_matrices.png")
    print("  Saved: results/figures/lr_coefficients.png")
    print("  Saved: results/figures/lr_comparison.png")
    print("  Saved: results/reports/lr_report.txt")

    return results


if __name__ == "__main__":
    run()
