"""
generate_embeddings.py
-----------------------
Standalone, FairFace-free embedding extraction: computes a 512-d
face-recognition embedding for every unique image in the manifest, using
the pretrained IResNet-34 backbone (195520backbone.pth), and exports it
enriched with pair metadata (dataset, pair number, genuine/impostor label,
filename) in CSV, .npz, and Parquet formats.

NORMALIZATION (confirmed active): every image is preprocessed with
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
before being passed to the backbone (see fr_transform below). No separate
Resize is needed since CALFW/AgeDB-30 images are already 112x112 -- verified
as a true numerical no-op against the previous Resize-based pipeline (max
abs diff 0.0), and load_fr_array() asserts the 112x112 precondition so a
wrongly-shaped image fails loudly instead of being silently mis-fed.

PIXEL DATA IS NOT STORED (by design, not an oversight): raw pixel arrays
(112x112x3 uint8) and their normalized float32 counterparts would add
~900MB and ~3.6GB respectively across 24,000 images -- 20-80x the size of
the embeddings file itself -- for data that is a pure, deterministic
function of the source image file, which is already on disk at the path
recorded in this file's `img_path` column. If you need either array for a
given image, reproduce it exactly with:

    import numpy as np
    from PIL import Image
    raw = np.array(Image.open(img_path).convert("RGB"))       # HWC uint8, 112x112x3
    normalized = (raw.astype(np.float32) / 255.0 - 0.5) / 0.5  # HWC float32, range [-1, 1]
    # fr_transform's ToTensor() additionally permutes to CHW before this
    # Normalize step; the per-pixel VALUES above are identical either way.

This is exactly what fr_transform + load_fr_array do internally (ToTensor
scales uint8 [0,255] to float [0,1], then Normalize maps [0,1] -> [-1,1]
via (x-0.5)/0.5); the two-line snippet above is not an approximation.

METADATA JOIN: each image path in CALFW/AgeDB-30 belongs to EXACTLY ONE
pair (verified: 24,000 unique paths for 24,000 (pair, img1/img2) manifest
references, zero overlap), so dataset / pair_id / label / filename can be
attached to each image's embedding row unambiguously by inverting the
manifest (each pair row expands into its two constituent images).

Outputs:
  1. Per-image .npy cache under --emb_dir (stable SHA-1-keyed filenames,
     resumable across runs).
  2. --out_csv    : embeddings.csv     (human-readable, one row per image)
  3. --out_npz    : embeddings.npz     (compact, fast to load in numpy)
  4. --out_parquet: embeddings.parquet (compact, fast to load in pandas)

Columns in all three formats:
    dataset          - "CALFW" or "AgeDB30"
    pair_id          - 0-5999 (pair number within its dataset)
    filename         - image filename only, e.g. "0.png"
    img_path         - full path to the source image
    label            - 1 = genuine, 0 = impostor
    is_genuine       - True / False (same information as `label`, as a bool)
    emb_norm         - L2 norm of the raw (pre-normalization) embedding
    emb_0 .. emb_511 - the L2-NORMALISED 512-d embedding vector (unit norm;
                        methodology Step 1). Divide back out by emb_norm if
                        the pre-normalisation vector is ever needed.

Usage:
    python3 src/features/generate_embeddings.py
"""
import os
import sys
import hashlib
import argparse
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from iresnet import iresnet34


def stable_key(path: str) -> str:
    """Deterministic (process-independent) cache key for an image path.
    Python's built-in hash() is salted per-process (PYTHONHASHSEED), so
    hash(path) is NOT safe as a cross-run cache filename -- sha1 is stable
    across every run/process forever."""
    return hashlib.sha1(path.encode("utf-8")).hexdigest()



def build_fr_backbone(weights_path: str, device: str):
    model = iresnet34()
    sd = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        print(f"WARNING: non-strict load -- missing={missing}, unexpected={unexpected}")
    model.eval().to(device)
    return model


# Confirmed active normalization: ToPILImage -> ToTensor -> Normalize(0.5,0.5,0.5).
fr_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_fr_array(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"))
    assert arr.shape[:2] == (112, 112), (
        f"{path} is {arr.shape[:2]}, not the expected 112x112 -- "
        f"fr_transform has no Resize step, so this would silently feed the "
        f"backbone a wrongly-shaped input.")
    return arr


def get_pixel_arrays(img_path: str):
    """On-demand reconstruction of (raw, normalized) pixel arrays for a
    single image -- NOT called during the main pipeline (pixel data is
    intentionally not stored, see module docstring), but provided so the
    documented formula is runnable, not just written down.
    Returns:
        raw:        HWC uint8 array, 112x112x3, values in [0, 255]
        normalized: HWC float32 array, 112x112x3, values in [-1, 1]
                    (identical values to what fr_transform feeds the model,
                    modulo the CHW permutation ToTensor() also applies)
    """
    raw = load_fr_array(img_path)
    normalized = (raw.astype(np.float32) / 255.0 - 0.5) / 0.5
    return raw, normalized


@torch.no_grad()
def embed_batch(paths, model, device):
    batch = torch.stack([fr_transform(load_fr_array(p)) for p in paths]).to(device)
    return model(batch).cpu().numpy()  # (B, 512), pre-L2-norm


def build_image_metadata(manifest_csv: str) -> pd.DataFrame:
    """Invert the pair-level manifest into one row per unique image, carrying
    dataset / pair_id / label / filename along with it."""
    manifest = pd.read_csv(manifest_csv)
    long_rows = []
    for r in manifest.itertuples(index=False):
        for img_path in (r.img1_path, r.img2_path):
            long_rows.append({
                "dataset": r.dataset,
                "pair_id": int(r.pair_id),
                "label": int(r.label),
                "img_path": img_path,
                "filename": os.path.basename(img_path),
            })
    meta = pd.DataFrame(long_rows).drop_duplicates(subset="img_path", keep="first")
    n_before = len(long_rows)
    if len(meta) != n_before:
        print(f"NOTE: {n_before - len(meta)} images are shared across multiple pairs; "
              f"kept the first pair's metadata for each (see script docstring).")
    return meta.reset_index(drop=True)


def extract_all(unique_paths, weights_path, emb_dir, batch_size, device):
    os.makedirs(emb_dir, exist_ok=True)
    done = {p for p in unique_paths if os.path.exists(os.path.join(emb_dir, stable_key(p) + ".npy"))}
    todo = [p for p in unique_paths if p not in done]
    print(f"{len(done)} images already cached, {len(todo)} remaining to process")

    if todo:
        model = build_fr_backbone(weights_path, device)
        for i in range(0, len(todo), batch_size):
            batch_paths = todo[i:i + batch_size]
            embs = embed_batch(batch_paths, model, device)
            for p, e in zip(batch_paths, embs):
                np.save(os.path.join(emb_dir, stable_key(p) + ".npy"), e)
            print(f"  processed {i + len(batch_paths)}/{len(todo)}", flush=True)


def assemble_dataframe(meta: pd.DataFrame, emb_dir: str):
    """Builds the exported per-image table.

    METHODOLOGY ALIGNMENT (Step 1: "Produces 512-dimensional L2-normalised
    feature vectors"): embeddings are L2-normalised here, immediately
    before export. `emb_norm` is captured from the RAW embedding *before*
    normalising, so it stays a meaningful per-image diagnostic (unusually
    small/large pre-norm magnitude can flag a poor detection/crop) instead
    of trivially collapsing to ~1.0 for every row once the exported vector
    itself is normalised.

    This does not change any similarity value anywhere in the pipeline:
    build_pairs.py reads embeddings from the separate per-image .npy cache
    (left raw/unnormalised on purpose, see extract_all()) via a cosine_sim()
    that already divides by both vectors' norms explicitly, so cosine
    similarity is mathematically identical whether or not the inputs were
    pre-normalised. This change only makes the *exported*
    embeddings.{csv,npz,parquet} literally match what Step 1 documents,
    and guards against a future consumer that assumes normalised vectors
    and takes a raw dot product instead of full cosine similarity.
    """
    embs_raw = np.stack([np.load(os.path.join(emb_dir, stable_key(p) + ".npy")) for p in meta["img_path"]])
    out = meta.copy()
    out["is_genuine"] = out["label"].astype(bool)
    norms = np.linalg.norm(embs_raw, axis=1)
    out["emb_norm"] = norms
    embs = embs_raw / np.clip(norms, 1e-12, None)[:, None]  # L2-normalise for export (Step 1)
    emb_cols = pd.DataFrame(embs, columns=[f"emb_{i}" for i in range(embs.shape[1])])
    out = pd.concat([out.reset_index(drop=True), emb_cols], axis=1)
    ordered = ["dataset", "pair_id", "filename", "img_path", "label", "is_genuine", "emb_norm"] \
        + [f"emb_{i}" for i in range(embs.shape[1])]
    return out[ordered], embs


def export_all(df: pd.DataFrame, embs: np.ndarray, out_csv: str, out_npz: str, out_parquet: str):
    for path in (out_csv, out_npz, out_parquet):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(out_csv, index=False)
    print(f"Wrote CSV:     {out_csv}  ({df.shape[0]} rows x {df.shape[1]} cols)")

    df.to_parquet(out_parquet, index=False)
    print(f"Wrote Parquet: {out_parquet}")

    np.savez_compressed(
        out_npz,
        dataset=df["dataset"].to_numpy(),
        pair_id=df["pair_id"].to_numpy(),
        filename=df["filename"].to_numpy(),
        img_path=df["img_path"].to_numpy(),
        label=df["label"].to_numpy(),
        is_genuine=df["is_genuine"].to_numpy(),
        emb_norm=df["emb_norm"].to_numpy(),
        embeddings=embs,  # (N, 512) array, the actual vectors
    )
    print(f"Wrote npz:     {out_npz}  (embeddings shape {embs.shape})")


def build_pairs_index(manifest_csv: str, emb_dir: str) -> pd.DataFrame:
    """Methodology Step 1: 'A pairs_index.csv maps each pair to its two
    embedding file paths.' Previously this mapping only existed implicitly
    (re-derivable via stable_key(img_path) if you already knew the
    formula) -- this materializes it as its own artifact, exactly as
    documented. Purely additive: does not change embeddings, similarity,
    or any other output."""
    manifest = pd.read_csv(manifest_csv)
    rows = []
    for r in manifest.itertuples(index=False):
        rows.append({
            "dataset": r.dataset,
            "pair_id": int(r.pair_id),
            "label": int(r.label),
            "emb1_path": os.path.join(emb_dir, stable_key(r.img1_path) + ".npy"),
            "emb2_path": os.path.join(emb_dir, stable_key(r.img2_path) + ".npy"),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/manifest.csv")
    ap.add_argument("--weights", default="models/195520backbone.pth")
    ap.add_argument("--emb_dir", default="outputs/embeddings")
    ap.add_argument("--out_csv", default="outputs/embeddings.csv")
    ap.add_argument("--out_npz", default="outputs/embeddings.npz")
    ap.add_argument("--out_parquet", default="outputs/embeddings.parquet")
    ap.add_argument("--out_pairs_index", default="outputs/pairs_index.csv")
    ap.add_argument("--batch_size", type=int, default=64,
                     help="Methodology Step 1 specifies batches of 64; eval-mode BatchNorm uses "
                          "running stats so this does not change the resulting embeddings, only throughput.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    meta = build_image_metadata(args.manifest)
    extract_all(meta["img_path"].tolist(), args.weights, args.emb_dir, args.batch_size, args.device)
    df, embs = assemble_dataframe(meta, args.emb_dir)
    export_all(df, embs, args.out_csv, args.out_npz, args.out_parquet)

    pairs_index = build_pairs_index(args.manifest, args.emb_dir)
    os.makedirs(os.path.dirname(args.out_pairs_index), exist_ok=True)
    pairs_index.to_csv(args.out_pairs_index, index=False)
    print(f"Wrote pairs index: {args.out_pairs_index}  ({len(pairs_index)} pairs)")


if __name__ == "__main__":
    main()
