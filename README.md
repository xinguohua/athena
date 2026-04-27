# ATHENA

ATHENA is a provenance-based intrusion detection system for stealthy Advanced Persistent Threats (APTs). It builds time-windowed snapshots from system audit logs, learns discriminative graph representations through LLM-guided contrastive learning, and reconstructs interpretable multi-stage attack chains by aligning anomalous behaviors against MITRE ATT&CK techniques.

## Method Overview

ATHENA consists of four stages:

1. **Snapshot Construction.** Audit events are partitioned into 1-minute non-overlapping windows; each window is materialized as a typed provenance graph and decomposed into node-centered *r*-hop subgraphs.
2. **LLM-Guided Graph Augmentation** (`process/mutation/`). For each benign anchor, structurally similar attack subgraphs are retrieved via the Weisfeiler–Leman subtree kernel, an LLM selects the most plausible substitution, semantic mutation rewrites command-line and path attributes to blend into the benign context, and a unified verification step filters mutations that violate operation legality, attribute feasibility, imperceptibility, or hardness.
3. **Adaptive Contrastive Learning** (`process/embedders/gcc_embedder_dev.py`). A 3-layer GIN with GRU temporal encoding is trained with a hard-sample-weighted supervised contrastive loss; a 2-layer MLP head produces per-snapshot binary anomaly labels.
4. **Global Anomaly Interpretation** (`process/technique_semantic_mapper.py`). Key causal paths are extracted, both sides are semantically enhanced, paths are mapped to ATT&CK techniques via Sentence-BERT similarity, and the resulting technique sequence is aligned against multi-stage attack patterns from AttackSeqBench using LCS.

## Repository Layout

```
process/
├── train_all.py                 # Training entry point
├── test_all.py                  # Inference / evaluation entry point
├── benchmark_augmentation.py    # Augmentation strategy benchmark
├── config.py / config.yaml      # Hyperparameters and dataset path mapping
├── datahandlers/                # DARPA E3/E5, OpTC, ATLAS loaders
├── embedders/
│   └── gcc_embedder_dev.py      # GIN + GRU encoder with contrastive learning
├── classfy/
│   ├── mlp_classify.py          # 2-layer MLP classifier
│   └── svm_classify.py          # Top-K deviation baseline
├── mutation/                    # LLM-guided mutation pipeline
│   ├── wl_kernel.py             # WL subtree kernel similarity
│   ├── structural.py            # Aligned-subgraph search and replacement
│   ├── semantic.py              # Three semantic mutation strategies
│   ├── verification.py          # Four-criterion unified verification
│   └── pipeline.py              # End-to-end mutation orchestration
├── llm_clients/                 # OpenAI-compatible LLM clients
├── technique_semantic_mapper.py # ATT&CK mapping (Sentence-BERT + Chroma)
├── data/
│   └── mitreembed_master_Chroma.csv  # ATT&CK technique corpus
└── chroma_db/                   # Persistent ATT&CK vector store
```

## Datasets

| Dataset | Source |
|---|---|
| DARPA E3 (Trace, Theia, Cadets, ClearScope) | https://github.com/darpa-i2o/Transparent-Computing |
| DARPA E5 (Trace, Theia, Cadets, ClearScope) | https://github.com/darpa-i2o/Transparent-Computing-5D |
| DARPA OpTC | https://github.com/FiveDirections/OpTC-data |
| ATLAS | https://github.com/purseclab/ATLAS |

Configure local dataset paths in `process/config.yaml` under the `local` or `remote` section.

## Environment

- Ubuntu 20.04, Python 3.9, CUDA-capable GPU (tested on NVIDIA RTX 4090)
- Major dependencies: PyTorch, igraph, leidenalg, scikit-learn, PyKEEN, pandas, numpy, orjson, LangChain, ChromaDB, sentence-transformers, psutil

```bash
conda create -n prographer python=3.9 -y
conda activate prographer
pip install torch igraph leidenalg scikit-learn pykeen pandas numpy orjson \
            langchain chromadb sentence-transformers psutil
```

## Configuration

1. Copy the API key template and fill in your credentials (used by the LLM-guided mutation stage):
   ```bash
   cp process/local_settings_example.py process/local_settings.py
   # then edit process/local_settings.py
   ```
2. Edit `process/config.yaml` so the `path_map` under your environment (`local` or `remote`) points to the unpacked dataset directories.
3. Open `process/train_all.py` and set:
   - `DATASET_NAME`: one of `cadets`, `theia`, `trace`, `clearscope`, `atlas`, `optcday1`
   - `EMBEDDER_NAME`: `gcc_dev` (default)
   - `CLASSIFY_NAME`: `mlp` (default)
   - `GLOBAL_ID`: a user identifier used as a model-file suffix

## Usage

```bash
conda activate prographer
cd /path/to/this/repo

# Train the encoder + MLP classifier
python -m process.train_all

# Evaluate on the held-out split
python -m process.test_all

# Augmentation strategy benchmark (single strategy)
python -m process.benchmark_augmentation --strategy llm_guided \
       --dataset cadets --scene cadets314

# Augmentation strategy benchmark (all five strategies on one scene)
python -m process.benchmark_augmentation --dataset cadets --scene cadets314
```

Training produces `process/prographer_encoder_<GLOBAL_ID>.pth` and `process/prographer_detector_<GLOBAL_ID>.pth`. Snapshot caches are written as `process/snapshot_data_<GLOBAL_ID>.pkl`.

## Notes

- `process/local_settings.py` and `*.pkl` / `*.pth` model files are git-ignored.
- For questions or issues, please open a GitHub issue.
