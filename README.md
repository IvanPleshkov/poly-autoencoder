# Polynomial autoencoder for embeddings

Reference implementation for the post [**Polynomial
autoencoder**](https://ivanpleshkov.dev/blog/polynomial-autoencoder/):
a closed-form autoencoder with **linear encoder** (top-d PCA) and
**quadratic decoder** (degree-2 polynomial lift over the PCA latent
plus Ridge OLS). No SGD, no epochs; both encoder and decoder are
closed-form.

```
encoder:   p = (V - V̄) @ Q                   # linear, via PCA
decoder:   V̂ = polynomial_lift(p) @ W        # quadratic, via Ridge OLS
residual:  V_resid = V - V̂
```

The construction is what the model-reduction literature calls a
**quadratic manifold** (see the post for the lineage and prior work).
This repo applies it to neural-network embeddings and benchmarks it on
BEIR retrieval.

## Quick start

```bash
git clone https://github.com/IvanPleshkov/poly-autoencoder.git
cd poly-autoencoder
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reproduce one row of the post's headline table:
python beir_eval.py --model nomic-embed-text-v1.5 --dataset fiqa --d 128,256
```

The first run downloads the model and the BEIR dataset, encodes the
corpus (5–15 min on M-series MacBook depending on the model), and over
the next 5–20 minutes prints a table with NDCG@10 and Recall@10 for all
four methods (raw / matryoshka / PCA / poly-AE). Subsequent runs reuse
the corpus encoding from `cache/`.

## Supported models

| Name in CLI                  | HuggingFace                                  | D    | Matryoshka |
|------------------------------|----------------------------------------------|------|------------|
| `nomic-embed-text-v1.5`      | `nomic-ai/nomic-embed-text-v1.5`             | 768  | yes        |
| `mxbai-embed-large-v1`       | `mixedbread-ai/mxbai-embed-large-v1`         | 1024 | yes        |
| `bge-base-en-v1.5`           | `BAAI/bge-base-en-v1.5`                      | 768  | no         |
| `e5-base-v2`                 | `intfloat/e5-base-v2`                        | 768  | no         |
| `bge-large-en-v1.5`          | `BAAI/bge-large-en-v1.5`                     | 1024 | no         |
| `bge-m3`                     | `BAAI/bge-m3`                                | 1024 | no         |
| `arctic-embed-m-v2.0`        | `Snowflake/snowflake-arctic-embed-m-v2.0`    | 768  | yes¹       |

The first four are what the post benchmarks. `bge-large-en-v1.5` and
`bge-m3` are kept for users who want to try more models.

¹ `arctic-embed-m-v2.0` requires `xformers` for its custom RoPE code.
On Apple Silicon (no xformers wheel) the load fails or produces
uninitialized indexing in the rope path. Use a CUDA box if you want to
benchmark this one; otherwise stick to the others.

## Supported BEIR datasets

| Name in CLI    | Description                       | Corpus  | Queries |
|----------------|-----------------------------------|---------|---------|
| `scifact`      | Scientific claim verification     |   5,183 |     300 |
| `nfcorpus`     | Medical document retrieval        |   3,633 |     323 |
| `arguana`      | Argument retrieval                |   8,674 |   1,406 |
| `fiqa`         | Financial Q&A                     |  57,638 |     648 |
| `trec-covid`   | COVID-19 literature retrieval     | 171,332 |      50 |

The post primarily uses **FiQA**. SciFact appears in §7 of the post
as a small-corpus contrast that demonstrates the in-sample-magic
overfitting failure mode.

## What the algorithm does

```
V (N × D)
    │  ENCODER (closed-form, PCA on the corpus)
    ├─►  fit_pca           →  (mean, Q (D×d), eigvals)
    └─►  project           →  P (N × d), ‖p‖ ≤ 0.9 (per-axis std=1)

P (N × d)
    │  DECODER (closed-form, Ridge OLS)
    ├─►  polynomial_lift   →  L (N × M),  M = 1 + d + d(d+1)/2
    └─►  ridge solve       →  W (M × D)

   reconstruction:   V̂ = L @ W
   residual:         V_resid = V − V̂
```

The dominant cost per `d` is the Ridge fit: O(N·M²) for the Gram
matrix plus O(M³) for `np.linalg.solve`. At d=128 (M=8385) on 57K
vectors this runs in seconds; at d=256 (M=33153) it takes 5–15
minutes on CPU.

## Files

- [`poly_pca.py`](poly_pca.py) — core primitives: `fit_pca`,
  `project`, `polynomial_lift`, `lift_dim`, `fit_ridge`. ~100 LOC.
- [`beir_eval.py`](beir_eval.py) — BEIR retrieval evaluation:
  downloads a BEIR dataset, encodes corpus and queries with a chosen
  model, fits PCA + Ridge transductively on the corpus, and computes
  NDCG@10 / Recall@10 for all four methods.

## License

MIT — copy, fork, or run on your own embeddings.
