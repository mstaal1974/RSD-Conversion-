# ESCO offline assets

This directory holds the data and model files the ESCO Alignment page
needs for offline / fast matching.

## 1 — ESCO classification CSVs

Drop the **ESCO v1.2.1 classification CSV release** in this directory.

1. Visit https://esco.ec.europa.eu/en/use-esco/download
2. Download the *ESCO dataset – classification – v1.2.1 – CSV* archive.
3. Unzip into this directory. The matcher needs at minimum:
   - `skills_en.csv`
   - `occupationSkillRelations_en.csv`

## 2 — Sentence-transformer model

The matcher uses `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
English). Bundle a local copy under `data/esco/model/` so neither the
Docker build nor the runtime needs Hugging Face access:

```bash
# On any machine with HF access:
pip install -U huggingface_hub
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir data/esco/model/ \
    --include "*.json" "*.txt" "*.safetensors" "tokenizer*" \
              "1_Pooling/*" "modules.json" "sentence_bert_config.json"

# Drop the duplicate pytorch_model.bin if HF mirrored both formats —
# safetensors is enough and keeps us under GitHub's 100 MB/file limit.
rm -f data/esco/model/pytorch_model.bin

# Commit (~90 MB total) into the repo:
git add -f data/esco/model/
git commit -m "Bundle MiniLM model for offline ESCO matching"
```

If `model.safetensors` exceeds GitHub's 100 MB hard limit, enable Git
LFS for the file: `git lfs install && git lfs track "data/esco/model/*.safetensors"`.

The model directory should contain at minimum: `config.json`,
`tokenizer.json` (or `tokenizer_config.json` + `vocab.txt`), and
`model.safetensors` (or `pytorch_model.bin`).

To use a model that lives somewhere else, set the `ESCO_MODEL_PATH`
environment variable to its absolute path — the loader resolves model
source in this order: `$ESCO_MODEL_PATH` → `data/esco/model/` → HF Hub.

## 3 — Embedding cache (auto-generated)

On first use the matcher embeds the ~14k skills and writes
`embeddings_sentence-transformers_all-MiniLM-L6-v2.npz` here. Subsequent
loads take ~2s. This file is gitignored; delete it to force a rebuild.

## Override the data directory

Set the `ESCO_DATA_DIR` env var to point this whole thing somewhere
else (e.g. a mounted volume).
