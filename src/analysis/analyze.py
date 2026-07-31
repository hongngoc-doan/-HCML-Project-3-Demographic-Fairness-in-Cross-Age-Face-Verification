"""
analyze.py
----------
Implements Steps 5-7 of the Project 3 methodology (Verification Threshold,
Group-Level & Interaction Analysis, Metrics/Plots/Outputs) on top of the
pair-level table produced by build_pairs.py (pairs.csv).

For EACH dataset (CALFW, AgeDB30) -- kept fully independent, never pooled:

  Step 5  Verification threshold
          - EER, FMR=1%, FMR=0.1% computed from the dataset's own ROC curve
            (similarity vs. label). FMR=1% is used as the fixed operating
            point for every fairness analysis below.

  Step 6  Group-level & interaction analysis
          - FNMR by age-gap band (0-5 / 6-15 / 16-30 / 30+ years), overall
            and cross-tabulated by gender and by race (small cells n<20
            flagged as indicative-only).
          - Race categories with fewer than MIN_N_RACE genuine pairs in
            that dataset are pooled into "Other" before regression.
          - Four independent logistic-regression audits are fit:
                (dataset) x (CR-FIQA-S, CR-FIQA-L)
            each with two models:
                incorrect ~ age_gap_est * C(gender1)  + quality_min_{S,L}
                incorrect ~ age_gap_est * C(race_grp) + quality_min_{S,L}
            fit on genuine pairs only, using that dataset's own FMR=1%
            threshold. CR-FIQA-S and CR-FIQA-L results are NEVER combined,
            averaged, or compared to one another -- each is a self-
            contained audit.

  Step 7  Metrics, plots & outputs
          - All descriptive tables, regression summaries, and plots are
            written to disk (see --outdir).

Usage:
    python3 src/analysis/analyze.py --pairs_csv outputs/pairs.csv --outdir outputs/analysis

Outputs (under --outdir):
    thresholds_summary.csv                  EER / FMR=1% / FMR=0.1% per dataset
    fnmr_by_agegap_<dataset>.csv             overall age-gap-band FNMR
    fnmr_by_gender_agegap_<dataset>.csv      gender x age-gap-band FNMR
    fnmr_by_race_agegap_<dataset>.csv        race x age-gap-band FNMR (n<20 flagged)
    interaction_audit_summary.csv            one row per significant/tested term,
                                              across all 4 dataset x backbone audits
    regression_<dataset>_<backbone>_<gender|race>.txt   full statsmodels summary
    results_<dataset>.csv                    genuine-pair table with derived columns
    plots/<dataset>_roc_by_race.png
    plots/<dataset>_accuracy_by_agegap.png
    plots/<dataset>_heatmap_race_agegap.png
    plots/<dataset>_heatmap_gender_agegap.png
    plots/<dataset>_accuracy_by_agegap_per_race.png
    plots/<dataset>_similarity_kde_by_agegap.png
    plots/<dataset>_<backbone>_quality_vs_similarity.png
    plots/<dataset>_<backbone>_3way_quality_split.png             (race x age-gap x quality)
    plots/<dataset>_<backbone>_3way_quality_split_gender.png      (gender x age-gap x quality)
    plots/<dataset>_<backbone>_quality_by_race_agegap.png
    plots/<dataset>_<backbone>_quality_by_gender_agegap.png
    plots/<dataset>_<backbone>_<gender|race>_regression_coefficients.png   (logistic-regression forest plot)
    plots/thresholds_summary.png                                    (EER / FNMR@FMR=1% / FNMR@FMR=0.1% bar chart)
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

AGE_BIN_EDGES = [-0.1, 5, 15, 30, 999]
AGE_BIN_LABELS = ["0-5", "6-15", "16-30", "30+"]
MIN_N_RACE = 60       # below this many genuine pairs (per dataset), a race is pooled into "Other"
SMALL_N_FLAG = 20      # below this many pairs, a descriptive cell is flagged indicative-only
DATASETS = ["CALFW", "AgeDB30"]
BACKBONES = ["S", "L"]


# ----------------------------------------------------------------------
# Step 5 -- Verification threshold
# ----------------------------------------------------------------------
def compute_thresholds(df):
    """ROC-based EER / FMR=1% / FMR=0.1% thresholds for one dataset."""
    fpr, tpr, thr = roc_curve(df["label"], df["similarity"])
    fnr = 1 - tpr

    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer_threshold = thr[eer_idx]
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0

    def at_fmr(target):
        idx = int(np.argmin(np.abs(fpr - target)))
        return thr[idx], fpr[idx], fnr[idx]

    t1, fmr1, fnmr1 = at_fmr(0.01)
    t01, fmr01, fnmr01 = at_fmr(0.001)

    return {
        "eer_threshold": eer_threshold, "eer": eer,
        "fmr1pct_threshold": t1, "fmr1pct_fmr": fmr1, "fmr1pct_fnmr": fnmr1,
        "fmr0_1pct_threshold": t01, "fmr0_1pct_fmr": fmr01, "fmr0_1pct_fnmr": fnmr01,
    }


def apply_threshold(df, threshold):
    df = df.copy()
    df["pred_genuine"] = (df["similarity"] >= threshold).astype(int)
    df["correct"] = (df["pred_genuine"] == df["label"]).astype(int)
    df["incorrect"] = 1 - df["correct"]
    return df


# ----------------------------------------------------------------------
# Step 6 -- Group-level breakdowns and interaction models
# ----------------------------------------------------------------------
def add_age_gap_band(df):
    df = df.copy()
    df["age_gap_band"] = pd.cut(df["age_gap_est"], bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS)
    return df


def pool_small_races(df, min_n=MIN_N_RACE):
    """Pool race1 categories with < min_n genuine pairs (in this dataset) into 'Other'."""
    genuine_counts = df.loc[df["label"] == 1, "race1"].value_counts()
    keep = set(genuine_counts[genuine_counts >= min_n].index)
    df = df.copy()
    df["race_grp"] = df["race1"].where(df["race1"].isin(keep), other="Other")
    return df, sorted(keep)


def fnmr_table(df_genuine, group_cols):
    """FNMR (and N) for genuine pairs, grouped by group_cols (list, e.g. ['age_gap_band'])."""
    g = df_genuine.groupby(group_cols, observed=True)["correct"]
    out = g.agg(N="size", FNMR=lambda s: 1 - s.mean()).reset_index()
    out["flag_small_n"] = out["N"] < SMALL_N_FLAG
    return out


def fit_interaction_model(df_genuine, demo_col, ref_level, quality_col):
    """Logit: incorrect ~ age_gap_est * C(demo_col, Treatment(ref)) + quality_col"""
    d = df_genuine.dropna(subset=["age_gap_est", quality_col, demo_col]).copy()
    formula = f"incorrect ~ age_gap_est * C({demo_col}, Treatment(reference='{ref_level}')) + {quality_col}"
    model = smf.logit(formula, data=d).fit(disp=0, maxiter=200)
    return model


def collect_audit_rows(model, dataset, backbone, demo_label, quality_col):
    """Extract age-gap main effect, each interaction term, and the quality effect into tidy rows."""
    rows = []
    for term, coef in model.params.items():
        pval = model.pvalues[term]
        kind = None
        if term == "Intercept":
            continue
        elif term == quality_col:
            kind = "quality_effect"
        elif ":" in term:
            kind = "interaction"
        elif term == "age_gap_est":
            kind = "age_gap_main_effect"
        else:
            kind = "group_main_effect"
        rows.append({
            "dataset": dataset, "fiqa_backbone": backbone, "demographic_variable": demo_label,
            "term": term, "kind": kind, "coef": coef, "p_value": pval,
            "significant_p<0.05": bool(pval < 0.05),
        })
    return rows


# ----------------------------------------------------------------------
# Step 7 -- Plots
# ----------------------------------------------------------------------
def plot_roc_by_race(df, dataset, outdir):
    plt.figure(figsize=(7, 6))
    for race, g in df.groupby("race1"):
        if len(g) < 30 or g["label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(g["label"], g["similarity"])
        plt.plot(fpr, tpr, label=f"{race} (AUC={auc(fpr, tpr):.3f}, n={len(g)})")
    fpr, tpr, _ = roc_curve(df["label"], df["similarity"])
    plt.plot(fpr, tpr, "k--", linewidth=2, label=f"Overall (AUC={auc(fpr, tpr):.3f})")
    plt.xlabel("False Match Rate (FMR)")
    plt.ylabel("True Match Rate (1 - FNMR)")
    plt.title(f"{dataset}: ROC by predicted race group")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_roc_by_race.png"), dpi=150)
    plt.close()


def plot_accuracy_by_agegap(df_genuine, dataset, outdir):
    g = df_genuine.groupby("age_gap_band", observed=True)["correct"].agg(["mean", "size"])
    plt.figure(figsize=(6, 5))
    bars = plt.bar(g.index.astype(str), g["mean"], color="tab:blue")
    for bar, n in zip(bars, g["size"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                  f"n={n}", ha="center", fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy (genuine pairs)")
    plt.xlabel("Age-gap band")
    plt.title(f"{dataset}: Accuracy by age-gap band")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_accuracy_by_agegap.png"), dpi=150)
    plt.close()


def plot_heatmap_race_agegap(df_genuine, dataset, outdir):
    pivot = df_genuine.pivot_table(index="race_grp", columns="age_gap_band",
                                    values="correct", aggfunc="mean", observed=True)
    pivot = pivot.reindex(columns=AGE_BIN_LABELS)
    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={"label": "Accuracy"})
    plt.title(f"{dataset}: Accuracy heatmap (race x age-gap)")
    plt.ylabel("Race group")
    plt.xlabel("Age-gap band")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_heatmap_race_agegap.png"), dpi=150)
    plt.close()


def plot_heatmap_gender_agegap(df_genuine, dataset, outdir):
    """Gender-parity counterpart of plot_heatmap_race_agegap."""
    pivot = df_genuine.pivot_table(index="gender1", columns="age_gap_band",
                                    values="correct", aggfunc="mean", observed=True)
    pivot = pivot.reindex(columns=AGE_BIN_LABELS)
    plt.figure(figsize=(7, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={"label": "Accuracy"})
    plt.title(f"{dataset}: Accuracy heatmap (gender x age-gap)")
    plt.ylabel("Gender")
    plt.xlabel("Age-gap band")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_heatmap_gender_agegap.png"), dpi=150)
    plt.close()


def plot_accuracy_by_agegap_per_race(df_genuine, dataset, outdir):
    g = (df_genuine.groupby(["race_grp", "age_gap_band"], observed=True)["correct"]
         .agg(mean="mean", size="size").reset_index())
    g = g[g["size"] >= 10]
    plt.figure(figsize=(7, 5))
    for race, sub in g.groupby("race_grp"):
        sub = sub.set_index("age_gap_band").reindex(AGE_BIN_LABELS)
        plt.plot(AGE_BIN_LABELS, sub["mean"], marker="o", label=race)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.xlabel("Age-gap band")
    plt.title(f"{dataset}: Accuracy by age-gap band, per race (n>=10 cells only)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_accuracy_by_agegap_per_race.png"), dpi=150)
    plt.close()


def plot_similarity_kde_by_agegap(df_genuine, dataset, outdir):
    plt.figure(figsize=(7, 5))
    for band in AGE_BIN_LABELS:
        sub = df_genuine[df_genuine["age_gap_band"] == band]
        if len(sub) < 10:
            continue
        sns.kdeplot(sub["similarity"], label=f"{band} (n={len(sub)})")
    plt.xlabel("Cosine similarity (genuine pairs)")
    plt.title(f"{dataset}: Similarity distribution by age-gap band")
    plt.legend(title="Age-gap band", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_similarity_kde_by_agegap.png"), dpi=150)
    plt.close()


def plot_quality_vs_similarity(df, dataset, backbone, outdir):
    col = f"quality_min_{backbone}"
    plt.figure(figsize=(6, 5))
    for lab, marker, color, name in [(1, "o", "tab:blue", "Genuine"), (0, "x", "tab:red", "Impostor")]:
        sub = df[df["label"] == lab]
        plt.scatter(sub[col], sub["similarity"], s=8, alpha=0.35, marker=marker, color=color, label=name)
    plt.xlabel(f"quality_min_{backbone} (CR-FIQA-{backbone})")
    plt.ylabel("Cosine similarity")
    plt.title(f"{dataset} / CR-FIQA-{backbone}: Quality vs. similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_{backbone}_quality_vs_similarity.png"), dpi=150)
    plt.close()


def plot_3way_quality_split(df_genuine, dataset, backbone, outdir):
    col = f"quality_min_{backbone}"
    median = df_genuine[col].median()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, mask, title in [
        (axes[0], df_genuine[col] < median, f"Below-median quality (< {median:.2f})"),
        (axes[1], df_genuine[col] >= median, f"At/above-median quality (>= {median:.2f})"),
    ]:
        sub = df_genuine[mask]
        pivot = sub.pivot_table(index="race_grp", columns="age_gap_band",
                                 values="correct", aggfunc="mean", observed=True)
        pivot = pivot.reindex(columns=AGE_BIN_LABELS)
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax,
                    cbar=(ax is axes[1]))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Age-gap band")
    axes[0].set_ylabel("Race group")
    fig.suptitle(f"{dataset} / CR-FIQA-{backbone}: 3-way accuracy (race x age-gap), split by quality")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_{backbone}_3way_quality_split.png"), dpi=150)
    plt.close()


def plot_3way_quality_split_gender(df_genuine, dataset, backbone, outdir):
    """Gender-parity counterpart of plot_3way_quality_split (race x age-gap x quality).
    Same construction, with gender1 in place of race_grp."""
    col = f"quality_min_{backbone}"
    median = df_genuine[col].median()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, mask, title in [
        (axes[0], df_genuine[col] < median, f"Below-median quality (< {median:.2f})"),
        (axes[1], df_genuine[col] >= median, f"At/above-median quality (>= {median:.2f})"),
    ]:
        sub = df_genuine[mask]
        pivot = sub.pivot_table(index="gender1", columns="age_gap_band",
                                 values="correct", aggfunc="mean", observed=True)
        pivot = pivot.reindex(columns=AGE_BIN_LABELS)
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax,
                    cbar=(ax is axes[1]))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Age-gap band")
    axes[0].set_ylabel("Gender")
    fig.suptitle(f"{dataset} / CR-FIQA-{backbone}: 3-way accuracy (gender x age-gap), split by quality")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_{backbone}_3way_quality_split_gender.png"), dpi=150)
    plt.close()


def plot_quality_by_race_agegap(df_genuine, dataset, backbone, outdir):
    col = f"quality_mean_{backbone}"
    pivot = df_genuine.pivot_table(index="race_grp", columns="age_gap_band",
                                    values=col, aggfunc="mean", observed=True)
    pivot = pivot.reindex(columns=AGE_BIN_LABELS)
    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": f"Mean {col}"})
    plt.title(f"{dataset} / CR-FIQA-{backbone}: Mean quality (race x age-gap)")
    plt.ylabel("Race group")
    plt.xlabel("Age-gap band")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_{backbone}_quality_by_race_agegap.png"), dpi=150)
    plt.close()


def plot_quality_by_gender_agegap(df_genuine, dataset, backbone, outdir):
    """Gender-parity counterpart of plot_quality_by_race_agegap."""
    col = f"quality_mean_{backbone}"
    pivot = df_genuine.pivot_table(index="gender1", columns="age_gap_band",
                                    values=col, aggfunc="mean", observed=True)
    pivot = pivot.reindex(columns=AGE_BIN_LABELS)
    plt.figure(figsize=(7, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": f"Mean {col}"})
    plt.title(f"{dataset} / CR-FIQA-{backbone}: Mean quality (gender x age-gap)")
    plt.ylabel("Gender")
    plt.xlabel("Age-gap band")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_{backbone}_quality_by_gender_agegap.png"), dpi=150)
    plt.close()


def plot_threshold_metrics(thresholds_df, outdir):
    """Visualizes Step 5's EER / FMR=1% / FMR=0.1% table (previously text-only)."""
    d = thresholds_df.set_index("dataset")
    metrics = pd.DataFrame({
        "EER": d["eer"] * 100,
        "FNMR @ FMR=1%": d["fmr1pct_fnmr"] * 100,
        "FNMR @ FMR=0.1%": d["fmr0_1pct_fnmr"] * 100,
    })
    ax = metrics.plot(kind="bar", figsize=(7, 5), color=["tab:gray", "tab:blue", "tab:orange"])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f%%", fontsize=8)
    plt.ylabel("Error rate (%)")
    plt.title("Verification error rates by dataset (Step 5)")
    plt.xticks(rotation=0)
    plt.legend(title=None)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "thresholds_summary.png"), dpi=150)
    plt.close()


def plot_regression_coefficients(model, dataset, backbone, demo_label, outdir):
    """Forest/coefficient plot for one logistic-regression audit: shows each term's
    coefficient and 95% CI, colored by significance. Makes the age-gap main effect,
    each interaction term, and the quality effect visually comparable at a glance."""
    ci = model.conf_int()
    ci.columns = ["lo", "hi"]
    coefs = pd.DataFrame({
        "coef": model.params,
        "lo": ci["lo"],
        "hi": ci["hi"],
        "p": model.pvalues,
    })
    coefs = coefs.drop(index="Intercept", errors="ignore")
    coefs["label"] = [t if len(t) < 42 else t[:39] + "..." for t in coefs.index]
    coefs = coefs.iloc[::-1]  # top-to-bottom reading order

    colors = ["tab:red" if p < 0.05 else "tab:gray" for p in coefs["p"]]
    plt.figure(figsize=(8, 0.45 * len(coefs) + 1.5))
    y = np.arange(len(coefs))
    plt.errorbar(coefs["coef"], y, xerr=[coefs["coef"] - coefs["lo"], coefs["hi"] - coefs["coef"]],
                 fmt="o", ecolor="black", elinewidth=1, capsize=3, markersize=0)
    plt.scatter(coefs["coef"], y, c=colors, zorder=3, s=50)
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.yticks(y, coefs["label"], fontsize=8)
    plt.xlabel("Logistic regression coefficient (log-odds of an incorrect verification), 95% CI")
    plt.title(f"{dataset} / CR-FIQA-{backbone} / {demo_label} model: coefficients "
              f"(red = p<0.05)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_{backbone}_{demo_label}_regression_coefficients.png"), dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Steps 5-7: threshold, group/interaction analysis, plots.")
    ap.add_argument("--pairs_csv", default="pairs.csv")
    ap.add_argument("--outdir", default="outputs/analysis")
    ap.add_argument("--min_n_race", type=int, default=MIN_N_RACE)
    args = ap.parse_args()

    min_n_race = args.min_n_race
    outdir = args.outdir
    plotdir = os.path.join(outdir, "plots")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(plotdir, exist_ok=True)

    df_all = pd.read_csv(args.pairs_csv)
    print(f"Loaded {args.pairs_csv}: {len(df_all)} pairs across {df_all['dataset'].nunique()} dataset(s).")

    threshold_rows = []
    audit_rows = []

    for dataset in DATASETS:
        df = df_all[df_all["dataset"] == dataset].copy()
        if df.empty:
            print(f"WARNING: no rows for dataset={dataset}, skipping.")
            continue
        print(f"\n=== {dataset} ({len(df)} pairs) ===")

        # ---------------- Step 5 ----------------
        thr = compute_thresholds(df)
        thr_row = {"dataset": dataset, **thr}
        threshold_rows.append(thr_row)
        print(f"  EER={thr['eer']*100:.2f}%  |  FMR=1% -> threshold={thr['fmr1pct_threshold']:.3f}, "
              f"FNMR={thr['fmr1pct_fnmr']*100:.2f}%  |  FMR=0.1% -> threshold={thr['fmr0_1pct_threshold']:.3f}")

        df = apply_threshold(df, thr["fmr1pct_threshold"])   # fixed operating point
        df = add_age_gap_band(df)
        df, kept_races = pool_small_races(df, min_n_race)
        print(f"  Race categories kept individually (n>={min_n_race} genuine pairs): {kept_races}")

        genuine = df[df["label"] == 1].copy()

        # ---------------- Step 6: descriptive breakdowns ----------------
        fnmr_age = fnmr_table(genuine, ["age_gap_band"])
        fnmr_age.to_csv(os.path.join(outdir, f"fnmr_by_agegap_{dataset}.csv"), index=False)

        fnmr_gender_age = fnmr_table(genuine, ["gender1", "age_gap_band"])
        fnmr_gender_age.to_csv(os.path.join(outdir, f"fnmr_by_gender_agegap_{dataset}.csv"), index=False)

        fnmr_race_age = fnmr_table(genuine, ["race1", "age_gap_band"])
        fnmr_race_age.to_csv(os.path.join(outdir, f"fnmr_by_race_agegap_{dataset}.csv"), index=False)

        genuine.to_csv(os.path.join(outdir, f"results_{dataset}.csv"), index=False)

        # ---------------- Step 6: four independent interaction audits ----------------
        for backbone in BACKBONES:
            qcol = f"quality_min_{backbone}"

            for demo_col, ref_level, demo_label in [
                ("gender1", "Male", "gender"),
                ("race_grp", "White", "race"),
            ]:
                try:
                    model = fit_interaction_model(genuine, demo_col, ref_level, qcol)
                    with open(os.path.join(outdir, f"regression_{dataset}_{backbone}_{demo_label}.txt"), "w") as f:
                        f.write(str(model.summary()))
                    audit_rows.extend(collect_audit_rows(model, dataset, backbone, demo_label, qcol))
                    plot_regression_coefficients(model, dataset, backbone, demo_label, plotdir)
                    print(f"  [{backbone}/{demo_label}] model fit OK "
                          f"({(model.pvalues < 0.05).sum()} terms significant at p<0.05)")
                except Exception as e:
                    print(f"  [{backbone}/{demo_label}] model FAILED to fit: {e}")

            # ---------------- Step 7: backbone-specific plots ----------------
            plot_quality_vs_similarity(df, dataset, backbone, plotdir)
            plot_3way_quality_split(genuine, dataset, backbone, plotdir)
            plot_3way_quality_split_gender(genuine, dataset, backbone, plotdir)
            plot_quality_by_race_agegap(genuine, dataset, backbone, plotdir)
            plot_quality_by_gender_agegap(genuine, dataset, backbone, plotdir)

        # ---------------- Step 7: dataset-level plots (FIQA-independent) ----------------
        plot_roc_by_race(df, dataset, plotdir)
        plot_accuracy_by_agegap(genuine, dataset, plotdir)
        plot_heatmap_race_agegap(genuine, dataset, plotdir)
        plot_heatmap_gender_agegap(genuine, dataset, plotdir)
        plot_accuracy_by_agegap_per_race(genuine, dataset, plotdir)
        plot_similarity_kde_by_agegap(genuine, dataset, plotdir)

    pd.DataFrame(threshold_rows).to_csv(os.path.join(outdir, "thresholds_summary.csv"), index=False)
    pd.DataFrame(audit_rows).to_csv(os.path.join(outdir, "interaction_audit_summary.csv"), index=False)
    plot_threshold_metrics(pd.DataFrame(threshold_rows), plotdir)

    print(f"\nDone. Tables written to {outdir}/, plots written to {plotdir}/")


if __name__ == "__main__":
    main()
