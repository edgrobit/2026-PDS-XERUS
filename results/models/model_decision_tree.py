"""
model_decision_tree.py — Decision Tree classifier
==================================================
Decision Trees split data on feature thresholds chosen to maximise
information gain. They handle both continuous (ITA) and categorical-like
features well and produce interpretable split rules.

Three feature sets compared:
    A_Baseline  — Shape only
    B_PlusColor — Shape + raw color variance
    C_PlusITA   — Shape + color + ITA

Outputs:
    results/models/dt_model_<name>.pkl
    results/predictions/dt_predictions_<name>.csv
    results/figures/dt_confusion_matrices.png
    results/figures/dt_depth_search.png
    results/figures/dt_depth_cv_<scenario>.png
    results/figures/dt_feature_importance.png
    results/figures/dt_comparison.png
    results/reports/dt_report.txt
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

# ── Configuration ─────────────────────────────────────────────────────────────
FEATURES_CSV = Path("data/features.csv")

MALIGNANT = {"BCC", "SCC", "MEL"}

TEST_SIZE = 0.2
RANDOM_STATE = 42

MAX_DEPTH_TO_TEST = 20
MIN_SAMPLES_LEAF = 20

FEATURE_SETS = {
    "A_Baseline": [
        # Shape only
        "asymmetry_score",
        "compactness",
        "lesion_percentage",
    ],

    "B_PlusColor": [
        # Shape + color heterogeneity
        "asymmetry_score",
        "compactness",
        "lesion_percentage",
        "rgb_var_r",
        "rgb_var_g",
        "rgb_var_b",
        "hsv_var_h",
        "hsv_var_s",
        "hsv_var_v",
    ],

    "C_PlusITA": [
        # Shape + color heterogeneity + ITA
        "asymmetry_score",
        "compactness",
        "lesion_percentage",
        "rgb_var_r",
        "rgb_var_g",
        "rgb_var_b",
        "hsv_var_h",
        "hsv_var_s",
        "hsv_var_v",
        "ita_mean",
    ],
}

for d in [
    "results/models",
    "results/predictions",
    "results/figures",
    "results/reports",
]:
    Path(d).mkdir(parents=True, exist_ok=True)


# ── New function: save one depth CV plot per scenario ─────────────────────────

def save_depth_cv_plot(model_name, label, depths, mean_scores, std_scores, best_depth):
    """
    Saves a cross-validation plot for tree depth selection.

    The plot shows the mean ROC-AUC for each max_depth, with error bars
    representing ±1 standard deviation across CV folds.
    """
    output_path = Path(f"results/figures/dt_depth_cv_{model_name}.png")

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        depths,
        mean_scores,
        yerr=std_scores,
        fmt="o-",
        capsize=3,
    )

    plt.axvline(
        best_depth,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"Best depth = {best_depth}",
    )

    plt.xlabel("Tree Depth")
    plt.ylabel("Mean AUC (+/- 1 std)")
    plt.title(f"Decision Tree, Cross-Validation for depth determination\n{label}")
    plt.grid(True, alpha=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"    Saved depth CV plot: {output_path}")


# ── Main model function ───────────────────────────────────────────────────────

def run(features_csv=FEATURES_CSV):
    print("\n" + "=" * 50)
    print("DECISION TREE — PROGRESSIVE FEATURE COMPARISON")
    print("=" * 50)

    df = pd.read_csv(features_csv)

    df["label"] = df["diagnostic"].apply(
        lambda x: 1 if x in MALIGNANT else 0
    )

    print(
        f"  Total: {len(df)} | "
        f"Malignant: {df['label'].sum()} | "
        f"Benign: {(df['label'] == 0).sum()}"
    )

    results = {}

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for model_name, feature_cols in FEATURE_SETS.items():
        print(f"\n  --- {model_name} ---")

        available = [c for c in feature_cols if c in df.columns]
        missing = [c for c in feature_cols if c not in df.columns]

        if missing:
            print(f"    WARNING: skipping missing: {missing}")

        df_model = df[["img_id", "diagnostic", "label"] + available].dropna()

        X = df_model[available].values
        y = df_model["label"].values

        positions = np.arange(len(df_model))

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X,
            y,
            positions,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        # ============================================================
        # DEPTH SEARCH WITH CROSS-VALIDATION
        # ============================================================
        depths = list(range(1, MAX_DEPTH_TO_TEST + 1))

        depth_scores = []
        depth_stds = []

        for depth in depths:
            dt_candidate = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                random_state=RANDOM_STATE,
            )

            cv_scores = cross_val_score(
                dt_candidate,
                X_train,
                y_train,
                cv=cv,
                scoring="roc_auc",
            )

            depth_scores.append((depth, cv_scores.mean()))
            depth_stds.append(cv_scores.std())

        best_depth, best_cv = max(depth_scores, key=lambda x: x[1])
        best_cv_std = depth_stds[depths.index(best_depth)]

        # Save one graph for this specific scenario
        save_depth_cv_plot(
            model_name=model_name,
            label=model_name,
            depths=depths,
            mean_scores=[score for _, score in depth_scores],
            std_scores=depth_stds,
            best_depth=best_depth,
        )

        # ============================================================
        # FINAL MODEL WITH BEST DEPTH
        # ============================================================
        dt = DecisionTreeClassifier(
            max_depth=best_depth,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
        )

        dt.fit(X_train, y_train)

        y_pred = dt.predict(X_test)
        y_prob = dt.predict_proba(X_test)[:, 1]

        accuracy = (y_pred == y_test).mean()
        roc_auc = roc_auc_score(y_test, y_prob)

        report = classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Malignant"],
        )

        imp_df = pd.DataFrame({
            "feature": available,
            "importance": dt.feature_importances_.round(4),
        }).sort_values("importance", ascending=False)

        print(
            f"    depth={best_depth}  "
            f"CV AUC={best_cv:.3f} ± {best_cv_std:.3f}  "
            f"acc={accuracy:.3f}  "
            f"AUC={roc_auc:.3f}"
        )

        results[model_name] = {
            "features": available,
            "best_depth": best_depth,
            "best_cv": best_cv,
            "best_cv_std": best_cv_std,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "report": report,
            "y_pred": y_pred,
            "y_test": y_test,
            "depth_scores": depth_scores,
            "depth_stds": depth_stds,
            "importances": imp_df,
        }

        # Save predictions
        pred_df = df_model.iloc[idx_test].copy().reset_index(drop=True)
        pred_df["predicted_label"] = y_pred
        pred_df["predicted_prob_mal"] = y_prob.round(4)
        pred_df["correct"] = y_pred == y_test

        pred_df.to_csv(
            f"results/predictions/dt_predictions_{model_name}.csv",
            index=False,
        )

        # Save model
        with open(f"results/models/dt_model_{model_name}.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": dt,
                    "features": available,
                    "depth": best_depth,
                },
                f,
            )

    # ── Comparison summary ────────────────────────────────────────────────────
    names = list(results.keys())

    labels = {
        "A_Baseline": "A: Shape only",
        "B_PlusColor": "B: + Color variance",
        "C_PlusITA": "C: + ITA (skin tone)",
    }

    colors = ["#95a5a6", "#4C72B0", "#DD8452"]

    base_acc = results["A_Baseline"]["accuracy"]
    base_auc = results["A_Baseline"]["roc_auc"]

    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)

    for name, r in results.items():
        d_acc = (
            f"  Δacc={r['accuracy'] - base_acc:+.3f}"
            if name != "A_Baseline"
            else ""
        )

        d_auc = (
            f"  Δauc={r['roc_auc'] - base_auc:+.3f}"
            if name != "A_Baseline"
            else ""
        )

        print(
            f"  {labels[name]:<22} "
            f"acc={r['accuracy']:.3f}  "
            f"AUC={r['roc_auc']:.3f}"
            f"{d_acc}{d_auc}"
        )

    # ── Figures ───────────────────────────────────────────────────────────────

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, name in zip(axes, names):
        cm = confusion_matrix(
            results[name]["y_test"],
            results[name]["y_pred"],
        )

        ConfusionMatrixDisplay(
            cm,
            display_labels=["Benign", "Malignant"],
        ).plot(
            ax=ax,
            colorbar=False,
        )

        ax.set_title(
            f"{labels[name]}\n"
            f"depth={results[name]['best_depth']}  "
            f"acc={results[name]['accuracy']:.3f}"
        )

    plt.suptitle("Decision Tree Confusion Matrices", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/dt_confusion_matrices.png", dpi=150)
    plt.close()

    # Combined depth search curves
    plt.figure(figsize=(10, 4))

    for name, color in zip(names, colors):
        depths, scores = zip(*results[name]["depth_scores"])

        plt.plot(
            depths,
            scores,
            marker="o",
            label=labels[name],
            color=color,
        )

        plt.axvline(
            results[name]["best_depth"],
            color=color,
            linestyle="--",
            alpha=0.4,
        )

    plt.xlabel("Max Depth")
    plt.ylabel("5-fold CV AUC")
    plt.title("Decision Tree — AUC vs Depth per Feature Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/dt_depth_search.png", dpi=150)
    plt.close()

    # Feature importance for full model
    best_imp = results["C_PlusITA"]["importances"]

    plt.figure(figsize=(8, 5))
    plt.barh(
        best_imp["feature"],
        best_imp["importance"],
        color="#4C72B0",
    )
    plt.xlabel("Importance")
    plt.title("Decision Tree Feature Importance (Full model)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("results/figures/dt_feature_importance.png", dpi=150)
    plt.close()

    # Bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    accs = [results[n]["accuracy"] for n in names]
    aucs = [results[n]["roc_auc"] for n in names]
    xlabels = [labels[n] for n in names]

    for ax, vals, metric in zip(
        axes,
        [accs, aucs],
        ["Accuracy", "ROC-AUC"],
    ):
        bars = ax.bar(
            xlabels,
            vals,
            color=colors,
            alpha=0.85,
        )

        ax.axhline(
            vals[0],
            color="gray",
            linestyle="--",
            alpha=0.6,
            label=f"Baseline ({vals[0]:.3f})",
        )

        ax.set_ylim(0.50, 0.80)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Feature Set")
        ax.legend(fontsize=8)

        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.005,
                f"{v:.3f}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    plt.suptitle(
        "Decision Tree — Does skin color improve prediction?",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("results/figures/dt_comparison.png", dpi=150)
    plt.close()

    # ── Save report ───────────────────────────────────────────────────────────
    with open("results/reports/dt_report.txt", "w") as f:
        f.write("Decision Tree Report\n" + "=" * 50 + "\n")
        f.write("Research question: Does skin color influence malignancy?\n\n")

        for name, r in results.items():
            f.write(f"Model {labels[name]}\n")
            f.write(f"  Features:   {r['features']}\n")
            f.write(f"  Best depth: {r['best_depth']}\n")
            f.write(f"  CV AUC:     {r['best_cv']:.3f} ± {r['best_cv_std']:.3f}\n")
            f.write(f"  Acc:        {r['accuracy']:.3f}\n")
            f.write(f"  AUC:        {r['roc_auc']:.3f}\n\n")
            f.write(r["report"] + "\n")
            f.write("Feature importances:\n")
            f.write(r["importances"].to_string(index=False) + "\n")
            f.write("-" * 50 + "\n\n")

    print("\n  Saved: results/figures/dt_confusion_matrices.png")
    print("  Saved: results/figures/dt_depth_search.png")
    print("  Saved: results/figures/dt_depth_cv_A_Baseline.png")
    print("  Saved: results/figures/dt_depth_cv_B_PlusColor.png")
    print("  Saved: results/figures/dt_depth_cv_C_PlusITA.png")
    print("  Saved: results/figures/dt_feature_importance.png")
    print("  Saved: results/figures/dt_comparison.png")
    print("  Saved: results/reports/dt_report.txt")

    return results


if __name__ == "__main__":
    run()