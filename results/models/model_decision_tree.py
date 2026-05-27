"""
model_decision_tree.py — Decision Tree classifier
==================================================
Decision Trees split data on feature thresholds chosen to maximise
information gain. They handle both continuous (ITA) and categorical-like
(FST) features well and produce interpretable split rules.

FST is particularly suited to Decision Trees since its discrete 1-6 values
create natural, clinically meaningful split points — unlike KNN where
continuous ITA is preferred.

Three feature sets compared (same as KNN for cross-model comparison):
    A_Baseline  — Shape only (asymmetry, compactness, lesion_percentage)
    B_PlusColor — Shape + raw color variance + ITA
    C_PlusFST   — Shape + color + FST (full model)

Outputs:
    results/models/dt_model_<name>.pkl
    results/predictions/dt_predictions_<name>.csv
    results/figures/dt_confusion_matrices.png
    results/figures/dt_depth_search.png
    results/figures/dt_feature_importance.png
    results/reports/dt_report.txt
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
    print("DECISION TREE — PROGRESSIVE FEATURE COMPARISON")
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

        # Search optimal depth on training set only
        depth_scores = []
        for depth in range(1, 11):
            dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20,
                                        random_state=RANDOM_STATE)
            score = cross_val_score(dt, X_train, y_train, cv=cv,
                                    scoring="roc_auc").mean()
            depth_scores.append((depth, score))

        best_depth, best_cv = max(depth_scores, key=lambda x: x[1])

        dt = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=20,
                                    random_state=RANDOM_STATE)
        dt.fit(X_train, y_train)

        y_pred = dt.predict(X_test)
        y_prob = dt.predict_proba(X_test)[:, 1]
        accuracy = (y_pred == y_test).mean()
        roc_auc  = roc_auc_score(y_test, y_prob)
        report   = classification_report(y_test, y_pred,
                       target_names=["Benign", "Malignant"])

        imp_df = pd.DataFrame({
            "feature":    available,
            "importance": dt.feature_importances_.round(4),
        }).sort_values("importance", ascending=False)

        print(f"    depth={best_depth}  acc={accuracy:.3f}  AUC={roc_auc:.3f}")

        results[model_name] = {
            "features":    available,
            "best_depth":  best_depth,
            "best_cv":     best_cv,
            "accuracy":    accuracy,
            "roc_auc":     roc_auc,
            "report":      report,
            "y_pred":      y_pred,
            "y_test":      y_test,
            "depth_scores": depth_scores,
            "importances": imp_df,
        }

        # Save predictions and model per feature set
        pred_df = df_model.iloc[idx_test].copy().reset_index(drop=True)
        pred_df["predicted_label"]    = y_pred
        pred_df["predicted_prob_mal"] = y_prob.round(4)
        pred_df["correct"]            = (y_pred == y_test)
        pred_df.to_csv(f"results/predictions/dt_predictions_{model_name}.csv",
                       index=False)

        with open(f"results/models/dt_model_{model_name}.pkl", "wb") as f:
            pickle.dump({"model": dt, "features": available,
                         "depth": best_depth}, f)

    # ── Comparison summary ────────────────────────────────────────────────────
    names    = list(results.keys())
    labels   = {"A_Baseline": "A: Shape only",
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
        ax.set_title(f"{labels[name]}\ndepth={results[name]['best_depth']}  "
                     f"acc={results[name]['accuracy']:.3f}")
    plt.suptitle("Decision Tree Confusion Matrices", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/dt_confusion_matrices.png", dpi=150)
    plt.close()

    # Depth search curves
    plt.figure(figsize=(10, 4))
    for name, color in zip(names, colors):
        depths, scores = zip(*results[name]["depth_scores"])
        plt.plot(depths, scores, marker="o", label=labels[name], color=color)
        plt.axvline(results[name]["best_depth"], color=color,
                    linestyle="--", alpha=0.4)
    plt.xlabel("Max Depth")
    plt.ylabel("5-fold CV AUC")
    plt.title("Decision Tree — AUC vs Depth per Feature Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/dt_depth_search.png", dpi=150)
    plt.close()

    # Feature importance for best model (C_PlusFST)
    best_imp = results["C_PlusFST"]["importances"]
    plt.figure(figsize=(8, 5))
    plt.barh(best_imp["feature"], best_imp["importance"], color="#4C72B0")
    plt.xlabel("Importance")
    plt.title("Decision Tree Feature Importance (Full model)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("results/figures/dt_feature_importance.png", dpi=150)
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

    plt.suptitle("Decision Tree — Does skin color improve prediction?",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/dt_comparison.png", dpi=150)
    plt.close()

    # ── Save report ───────────────────────────────────────────────────────────
    with open("results/reports/dt_report.txt", "w") as f:
        f.write("Decision Tree Report\n" + "="*50 + "\n")
        f.write("Research question: Does skin color influence malignancy?\n\n")
        for name, r in results.items():
            f.write(f"Model {labels[name]}\n")
            f.write(f"  Features:   {r['features']}\n")
            f.write(f"  Best depth: {r['best_depth']}\n")
            f.write(f"  CV AUC:     {r['best_cv']:.3f}\n")
            f.write(f"  Acc:        {r['accuracy']:.3f}\n")
            f.write(f"  AUC:        {r['roc_auc']:.3f}\n\n")
            f.write(r["report"] + "\n")
            f.write("Feature importances:\n")
            f.write(r["importances"].to_string(index=False) + "\n")
            f.write("-"*50 + "\n\n")

    print("\n  Saved: results/figures/dt_confusion_matrices.png")
    print("  Saved: results/figures/dt_depth_search.png")
    print("  Saved: results/figures/dt_feature_importance.png")
    print("  Saved: results/figures/dt_comparison.png")
    print("  Saved: results/reports/dt_report.txt")

    return results


if __name__ == "__main__":
    run()
