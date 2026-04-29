# Polynomial Autoencoder for Embeddings

Reference implementation for the post **"Полиномиальный автоэнкодер для
эмбеддингов"**: a closed-form autoencoder with **linear encoder** (top-d
PCA) and **quadratic decoder** (polynomial lift over the PCA coordinate
+ Ridge OLS). No SGD: both encoder and decoder are closed-form. Evaluated
on eight embeddings (transformer text, CNN images, word vectors,
classical SVD).

```
encoder:   p = (V - V̄) @ Q                   # linear, via PCA
decoder:   V̂ = polynomial_lift(p) @ W + b    # quadratic, via Ridge
residual:  V_resid = V - V̂
```

Mechanically the same recipe is **PPCR** ([Vong, Geladi, Wold & Esbensen
1989][ppcr]) with the target set to the input itself — turning supervised
polynomial PC regression into self-reconstruction (autoencoding). The
post shows that on sufficiently anisotropic embeddings (cond ≥ 48),
this autoencoder lifts R² by **+9.5–13.4 percentage points** over the
linear-decoder baseline (= plain PCA) at the same per-vector budget; on
already-isotropic ones (GloVe, NYTimes), the lift is within noise.

[ppcr]: https://www.sciencedirect.com/science/article/abs/pii/0169743989801169

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reproduce the post's headline table (≈30 min, includes encoding 4
# transformer models on 20K DBpedia texts):
python reproduce.py

# Or only the four classical pre-computed embeddings (≈5-10 min):
python reproduce.py --classical-only

# Single embedding × d sweep:
python run.py --embedding dbpedia-openai --d 50,100,200
python run.py --embedding all-MiniLM-L6-v2 --d 40 --n 20000
```

## Datasets

The four classical embeddings are downloaded automatically on first
use:

| Embedding         | Source                                            | D    |
|-------------------|---------------------------------------------------|------|
| `dbpedia-openai`  | `KShivendu/dbpedia-entities-openai-1M` (HF)       | 1536 |
| `glove-100`       | `ann-benchmarks` (`glove-100-angular.hdf5`)       |  100 |
| `nytimes-256`     | `ann-benchmarks` (`nytimes-256-angular.hdf5`)     |  256 |
| `deep-image-96`   | `ann-benchmarks` (`deep-image-96-angular.hdf5`)   |   96 |

The `ann-benchmarks` HDF5 files are large (300 MB – 4 GB). On first
load you'll be prompted to download them via `curl`:

```
curl -sL --output cache/glove-100-angular.hdf5 \
  http://ann-benchmarks.com/glove-100-angular.hdf5
```

The four transformer-encoded embeddings re-use the **text** field of
DBpedia (first 20K rows by default) and run it through these models on
CPU:

| Encoder                  | Source                                 | D   |
|--------------------------|----------------------------------------|-----|
| `sentence-t5-base`       | `sentence-transformers/sentence-t5-base` | 768 |
| `all-mpnet-base-v2`      | `sentence-transformers/all-mpnet-base-v2`| 768 |
| `all-MiniLM-L6-v2`       | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `clip-text-vit-b32`      | `openai/clip-vit-base-patch32` (text)    | 512 |

Encoding is cached as `.npy` after the first run.

## What the algorithm does

```
V (N × D)
    │  ENCODER (closed-form via PCA on train)
    ├─►  fit_pca on train  →  (mean, Q (D×d), eigvals)
    └─►  project            →  P (N × d), ‖p‖ ≤ 0.9 (per-axis std=1)

P (N × d)
    │  DECODER (closed-form via Ridge OLS)
    ├─►  polynomial_lift   →  L (N × M),  M = 1 + d + d(d+1)/2
    └─►  ridge solve       →  W (M × D)

   reconstruction:   V̂ = L @ W
   residual:         V_resid = V − V̂

   R²_test = 1 − ‖V_test − V̂_test‖² / ‖V_test − mean_train‖²
```

All steps are closed-form (no SGD). Per d, the dominant cost is the
ridge fit: O(N M²) for the Gram matrix and O(M³) for the Cholesky
solve. For d=100 (M=5151) on 100K vectors this runs under a minute on
CPU; for d=200 (M=20301) under five minutes.

## Files

- [`poly_pca.py`](poly_pca.py) — core: `fit_pca`, `polynomial_lift`,
  `fit_ridge`, `r2`, `anisotropy`, `run_experiment`. ~150 LOC.
- [`dataloaders.py`](dataloaders.py) — loaders for the four classical
  embeddings (HF + ann-benchmarks HDF5). Named to avoid colliding with
  the HuggingFace `datasets` package.
- [`encoders.py`](encoders.py) — transformer encoders for DBpedia text
  (sentence-transformers + CLIP).
- [`run.py`](run.py) — single-embedding CLI with d-sweep.
- [`reproduce.py`](reproduce.py) — reproduces the eight-row table from
  the post.

## Expected output

For `dbpedia-openai` at d=100 you should see numbers close to:

```
   d      M  PCA R²  Poly R²_train  Poly R²_test  Δ over PCA  cond_orig  cond_resid
 100   5151   0.61          0.762         0.729      +0.120       72.0       3.42
```

— ≈12 percentage points of variance recovered by the quadratic decoder
that linear PCA misses, with a small (~3 p.p.) train/test overfit gap.

For `glove-100` at d=50:

```
   d      M  PCA R²  Poly R²_train  Poly R²_test  Δ over PCA  cond_orig  cond_resid
  50   1326   0.60          0.621         0.617      +0.014        6.5       2.9
```

— a ~1 p.p. lift, within noise: GloVe is already linearly structured.

## License

MIT — feel free to copy, fork, or run on your own embeddings.
