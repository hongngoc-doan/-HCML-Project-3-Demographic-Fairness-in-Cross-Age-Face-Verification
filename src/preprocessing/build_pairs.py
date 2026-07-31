"""
build_pairs.py
---------------
Combines the manifest + per-image feature cache (embeddings, FairFace
demographic predictions) + genuine CR-FIQA quality scores into the final
PAIR-LEVEL table required by the project brief:

  1. similarity          - cosine similarity between the two embeddings
  2. label               - 1 = genuine, 0 = impostor
  3. age_gap_est         - |pred_age_mid(img1) - pred_age_mid(img2)|,
                            ESTIMATED from FairFace age-bin midpoints
                            (no ground-truth age metadata survives in the
                            anonymized CALFW/AgeDB-30 zips -> flagged
                            explicitly, see README / report)
  4. quality1_S/L, quality2_S/L - per-dataset-normalized CR-FIQA quality
                            score, for EACH backbone independently
  5. quality_min/mean/diff_{S,L} - pair-level aggregates of the quality
                            score, for EACH backbone independently
  6. demographic_group   - predicted FairFace race x gender group (pair-level:
                            uses img1's prediction as the pair's group, with
                            a flag if img1/img2 predictions disagree)

QUALITY SCORE PROVENANCE (methodology Step 4: "Two models used in
parallel"):
Genuine CR-FIQA(S) (iresnet50, official checkpoint 32572backbone.pth) AND
CR-FIQA(L) (iresnet100, official checkpoint 181952backbone.pth) quality
scores, produced independently by two runs of crfiqa_scorer.py
(--backbone S and --backbone L). Both backbones are carried all the way
through to the final pair table as parallel *_S / *_L column families --
this is a correction versus an earlier version of this script, which only
loaded a single --crfiqa_scores file and produced unsuffixed quality
columns (quality1_raw, quality1, quality_min, ...). That version could
only ever represent ONE backbone at a time and could not reproduce the
actual required table schema (quality*_S and quality*_L side by side, as
consumed by the Step 6/7 analysis and as verified against the real
pairs_CALFW.csv / pairs_AgeDB30.csv deliverables). See project summary for
details of this fix.

Each backbone's raw scores are min-max normalized to [0, 1] SEPARATELY,
per dataset (methodology Step 4: "min-max normalisation across all 12,000
images in the dataset") -- S is never normalized against L's min/max or
vice versa, since the two backbones produce raw scores on different,
unrelated scales.

INDEPENDENCE OF THE TWO BACKBONES: FIQA-S and FIQA-L are independent
analyses of the same pairs, not two halves of one analysis -- their scores
come from different models and are never compared, combined, or
correlated with each other. Concretely, a pair missing a CR-FIQA(S) score
(or CR-FIQA(L) score) for either image is NOT dropped from the whole
table; only that backbone's quality*_S (or quality*_L) columns are left
NaN for that row, and the row is still built normally with its
similarity/age/demographic fields plus whichever backbone(s) did resolve.
This is a fix versus an earlier version of this script, which required
BOTH backbones to have scores before emitting a row at all -- coupling
FIQA-S's coverage to FIQA-L's (and vice versa) and silently shrinking
both "independent" analyses whenever only one backbone had a gap.
"""
import os
import argparse
import hashlib
import numpy as np
import pandas as pd


def stable_key(path: str) -> str:
    """Must match extract_features.py's stable_key exactly."""
    return hashlib.sha1(path.encode("utf-8")).hexdigest()


def emb_path_for(img_path: str, emb_dir: str) -> str:
    return os.path.join(emb_dir, stable_key(img_path) + ".npy")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Methodology Step 2: sim(a,b) = (a . b) / (||a|| . ||b||). Dividing by
    both norms explicitly means this is correct cosine similarity
    regardless of whether the inputs happen to already be unit-norm."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_crfiqa_scores(path: str, backbone: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="img_path", keep="last").set_index("img_path")
    if "crfiqa_backbone" in df.columns:
        wrong = df[df["crfiqa_backbone"] != backbone]
        if len(wrong):
            raise ValueError(
                f"{path}: expected all rows to have crfiqa_backbone == '{backbone}', "
                f"found {len(wrong)} rows with a different value. Did you pass "
                f"--crfiqa_scores_s / --crfiqa_scores_l the right way round?")
    return df


def normalize_quality_per_dataset(df: pd.DataFrame, raw_col1: str, raw_col2: str,
                                   out_col1: str, out_col2: str) -> None:
    """Step 4: 'min-max normalisation across all 12,000 images in the
    dataset' -> per-dataset (CALFW normalized separately from AgeDB-30),
    computed independently for each CR-FIQA backbone. Mutates df in place."""
    for ds in df["dataset"].unique():
        mask = df["dataset"] == ds
        lo = df.loc[mask, [raw_col1, raw_col2]].min().min()
        hi = df.loc[mask, [raw_col1, raw_col2]].max().max()
        span = (hi - lo) if hi > lo else 1.0
        df.loc[mask, out_col1] = (df.loc[mask, raw_col1] - lo) / span
        df.loc[mask, out_col2] = (df.loc[mask, raw_col2] - lo) / span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/manifest.csv")
    ap.add_argument("--features", default="outputs/image_features.csv")
    ap.add_argument("--emb_dir", default="outputs/embeddings")
    ap.add_argument("--crfiqa_scores_s", default="outputs/crfiqa_scores_S.csv",
                     help="Output of `crfiqa_scorer.py --backbone S`.")
    ap.add_argument("--crfiqa_scores_l", default="outputs/crfiqa_scores_L.csv",
                     help="Output of `crfiqa_scorer.py --backbone L`.")
    ap.add_argument("--out_csv", default="outputs/pairs.csv")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    feats = pd.read_csv(args.features)
    feats = feats.drop_duplicates(subset="img_path", keep="last").set_index("img_path")

    crfiqa_s = load_crfiqa_scores(args.crfiqa_scores_s, "S")
    crfiqa_l = load_crfiqa_scores(args.crfiqa_scores_l, "L")

    missing_s = set(feats.index) - set(crfiqa_s.index)
    missing_l = set(feats.index) - set(crfiqa_l.index)
    if missing_s:
        print(f"WARNING: {len(missing_s)} images have embeddings/demographics but no "
              f"CR-FIQA(S) score yet -- pairs referencing them will keep quality*_S as "
              f"NaN (FIQA-S is analyzed independently of FIQA-L, so this does not affect "
              f"FIQA-L coverage).")
    if missing_l:
        print(f"WARNING: {len(missing_l)} images have embeddings/demographics but no "
              f"CR-FIQA(L) score yet -- pairs referencing them will keep quality*_L as "
              f"NaN (FIQA-L is analyzed independently of FIQA-S, so this does not affect "
              f"FIQA-S coverage).")

    missing_imgs = set(manifest["img1_path"]) | set(manifest["img2_path"])
    missing_imgs -= set(feats.index)
    if missing_imgs:
        print(f"WARNING: {len(missing_imgs)} images referenced in manifest are not yet "
              f"in the feature cache. Pairs referencing them will be dropped from this "
              f"run (re-run once extract_features.py finishes).")

    rows = []
    dropped_core = 0
    missing_quality_s = 0
    missing_quality_l = 0
    for r in manifest.itertuples(index=False):
        # Only similarity/age/demographic inputs gate whether a pair gets a row
        # at all. CR-FIQA(S) and CR-FIQA(L) are independent analyses of that
        # same row -- a gap in one backbone's coverage must not drop the pair
        # from the other backbone's analysis (see module docstring).
        if r.img1_path not in feats.index or r.img2_path not in feats.index:
            dropped_core += 1
            continue
        f1 = feats.loc[r.img1_path]
        f2 = feats.loc[r.img2_path]

        e1 = np.load(emb_path_for(r.img1_path, args.emb_dir))
        e2 = np.load(emb_path_for(r.img2_path, args.emb_dir))
        sim = cosine_sim(e1, e2)
        del e1, e2

        age_gap = abs(float(f1["pred_age_mid"]) - float(f2["pred_age_mid"]))

        group1 = f"{f1['pred_race']}_{f1['pred_gender']}"
        group2 = f"{f2['pred_race']}_{f2['pred_gender']}"

        row = {
            "dataset": r.dataset,
            "pair_id": r.pair_id,
            "label": int(r.label),
            "similarity": sim,
            "age_gap_est": age_gap,
            "age1_est": f1["pred_age_mid"],
            "age2_est": f2["pred_age_mid"],
            "race1": f1["pred_race"], "race1_conf": f1["pred_race_conf"],
            "race2": f2["pred_race"], "race2_conf": f2["pred_race_conf"],
            "gender1": f1["pred_gender"], "gender1_conf": f1["pred_gender_conf"],
            "gender2": f2["pred_gender"], "gender2_conf": f2["pred_gender_conf"],
            "demographic_group": group1,
            "demographic_group_agrees": group1 == group2,
        }

        if r.img1_path in crfiqa_s.index and r.img2_path in crfiqa_s.index:
            row["quality1_raw_S"] = crfiqa_s.loc[r.img1_path, "crfiqa_raw"]
            row["quality2_raw_S"] = crfiqa_s.loc[r.img2_path, "crfiqa_raw"]
        else:
            row["quality1_raw_S"] = np.nan
            row["quality2_raw_S"] = np.nan
            missing_quality_s += 1

        if r.img1_path in crfiqa_l.index and r.img2_path in crfiqa_l.index:
            row["quality1_raw_L"] = crfiqa_l.loc[r.img1_path, "crfiqa_raw"]
            row["quality2_raw_L"] = crfiqa_l.loc[r.img2_path, "crfiqa_raw"]
        else:
            row["quality1_raw_L"] = np.nan
            row["quality2_raw_L"] = np.nan
            missing_quality_l += 1

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Built {len(df)} pairs, dropped {dropped_core} "
          f"(missing embeddings/demographic features).")
    if missing_quality_s:
        print(f"  -> {missing_quality_s} of those rows have NaN quality*_S "
              f"(no CR-FIQA(S) score for one/both images); FIQA(L) columns are unaffected.")
    if missing_quality_l:
        print(f"  -> {missing_quality_l} of those rows have NaN quality*_L "
              f"(no CR-FIQA(L) score for one/both images); FIQA(S) columns are unaffected.")

    # Per-DATASET, per-BACKBONE normalization of the quality proxy (min-max
    # to [0,1]). Per-dataset because CALFW / AgeDB-30 differ in average
    # norm/quality (different capture conditions), so pooling before
    # normalizing would bake in a dataset effect that isn't really about
    # "quality". Per-backbone because S and L are independently-trained
    # models with unrelated raw score scales.
    normalize_quality_per_dataset(df, "quality1_raw_S", "quality2_raw_S", "quality1_S", "quality2_S")
    normalize_quality_per_dataset(df, "quality1_raw_L", "quality2_raw_L", "quality1_L", "quality2_L")

    df["quality_min_S"] = df[["quality1_S", "quality2_S"]].min(axis=1)
    df["quality_mean_S"] = df[["quality1_S", "quality2_S"]].mean(axis=1)
    df["quality_diff_S"] = (df["quality1_S"] - df["quality2_S"]).abs()

    df["quality_min_L"] = df[["quality1_L", "quality2_L"]].min(axis=1)
    df["quality_mean_L"] = df[["quality1_L", "quality2_L"]].mean(axis=1)
    df["quality_diff_L"] = (df["quality1_L"] - df["quality2_L"]).abs()

    # Final column order for the combined outputs/pairs.csv (both datasets,
    # distinguished by the "dataset" column; see analyze.py / methodology_analysis.py).
    ordered = [
        "dataset", "pair_id", "label", "similarity", "age_gap_est", "age1_est", "age2_est",
        "race1", "race1_conf", "race2", "race2_conf", "gender1", "gender1_conf", "gender2", "gender2_conf",
        "demographic_group", "demographic_group_agrees",
        "quality1_raw_S", "quality2_raw_S", "quality1_raw_L", "quality2_raw_L",
        "quality1_S", "quality2_S", "quality1_L", "quality2_L",
        "quality_min_S", "quality_mean_S", "quality_diff_S",
        "quality_min_L", "quality_mean_L", "quality_diff_L",
    ]
    df = df[ordered]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)
    print(df.groupby(["dataset", "label"]).size())
    print(df[["similarity", "age_gap_est", "quality_mean_S", "quality_mean_L"]].describe())


if __name__ == "__main__":
    main()
