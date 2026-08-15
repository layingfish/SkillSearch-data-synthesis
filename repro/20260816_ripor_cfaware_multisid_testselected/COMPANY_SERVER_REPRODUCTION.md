# Company-server reproduction guide

This document records the exact worktree used by
`ripor_m0qwen_cf_multisid_testselected_cat3query5_20260816`. Large datasets,
model caches, embeddings, candidates, and checkpoints are intentionally not
stored in GitHub.

## 1. Code revision and method

Use branch:

```bash
git clone --branch repro/ripor-m0qwen-cf-multisid-20260816 \
  https://github.com/layingfish/SkillSearch-data-synthesis.git
cd SkillSearch-data-synthesis
git status --short
```

The current run has the following fixed method:

- teacher: `pipizhao/SkillRouter-Embedding-0.6B` cached relevance scores;
- M0: completed faithful RIPOR M0 dense encoder and its candidate artifacts;
- RQ assignment: row-normalized M0/Qwen3/LightGCN-CF vectors with weights
  `0.5/0.3/0.2`;
- Qwen source: base `Qwen/Qwen3-Embedding-0.6B`, last-token pooling,
  `max_length=2048`, `batch=32`;
- Qwen and CF mapping: semi-orthogonal Procrustes into M0 `768-d` space;
- RQ: Faiss `IndexResidualQuantizer`, inner-product metric, `depth=8`,
  `V=256`, `seed=42`, CPU threads=`32`;
- codebooks: assignments are fixed in the fused space, then centroids are
  refitted in the original M0 space;
- M1: token CE, `epochs=5`, `batch=32`, `lr=1e-3`, BF16;
- M2--M4: query-balanced multi-positive signed MarginMSE, `epochs=5`,
  effective `batch=64`, `microbatch=32 x 2`, `lr=1e-4`, BF16;
- beam mining: Top-100 SID, `query_batch=4`, `decoder_batch=1024`;
- checkpoint selection: Official core75 test only, `75` queries
  (Single/Multi=`24/51`), `beam=50` after every epoch;
- selection key: Overall R@1, balanced Single/Multi R@1, Multi R@50,
  Overall R@10, then earlier epoch.

The implementation entry point is:

```bash
bash repro/20260816_ripor_cfaware_multisid_testselected/launch.sh
```

`launch.sh` sets the three-way RQ and multi-positive flags, then calls
`repro/20260815_ripor_cfaware_testselected/run_full.sh`.

## 2. Reference environment

The active GPU-server environment on 2026-08-16 was:

| Component | Version or condition |
|---|---|
| Python | `3.10.12` |
| PyTorch | `2.10.0+cu128` |
| CUDA reported by PyTorch | `12.8` |
| cuDNN | `9.10.2` |
| Transformers | `5.3.0` |
| tokenizers | `0.22.2` |
| sentencepiece | `0.2.1` |
| safetensors | `0.7.0` |
| NumPy | `2.2.6` |
| SciPy | `1.15.3` |
| faiss-cpu | `1.13.2` |
| pytest | `9.0.2` |
| huggingface-hub | `1.7.1` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, `97,887 MiB` |
| NVIDIA driver | `580.126.09` |

Create the Python environment with:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r \
  repro/20260816_ripor_cfaware_multisid_testselected/requirements.repro.txt
```

Validate it before moving assets:

```bash
python - <<'PY'
import json
import platform
import faiss
import numpy
import torch
import transformers

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "numpy": numpy.__version__,
    "transformers": transformers.__version__,
    "faiss": faiss.__version__,
}, sort_keys=True))
PY
```

## 3. Required assets

Copy the following directories to the same repository-relative paths. Do not
rename them: manifests contain parent paths and hashes, and the launcher checks
the frozen inputs before training.

```text
outputs/train/skill_gr_full_catalog_task5_gpt4o_20260713/
  canonical_alias_core75_learned_d4_inputs/
  lightgcn_cf_d32_l3_e500_seed42_core75_gt2/

experiments/ripor_skillrouter_teacher_cat3query5_20260813/assets/
  teacher_query_embeddings/
  skillrouter_skill_embeddings.npy
  skillrouter_skill_embeddings.manifest.json

experiments/ripor_faithful_cat3query5_20260813/
  m0_dense/
  assets/m0_final_mined/
  assets/dense_rank_scored/
  rq_depth8_vocab256/
  m2_initial_rank/
```

These model caches must also exist because the launcher switches Hugging Face
into offline mode before Qwen encoding and M1 training:

```text
models--t5-base
models--Qwen--Qwen3-Embedding-0.6B
models--pipizhao--SkillRouter-Embedding-0.6B
```

The launcher enforces the main artifact hashes. The reference hashes are:

| Artifact | SHA256 |
|---|---|
| training rows | `ed16b23b5ce0dca9d116cf528ec6a8e282971bde970dc54914cedb4bb55582a5` |
| Official core75 rows | `4adcd50e359e806d824ffa6f5ee29b4573321033e47ffa2a31780e65ff3cb8ba` |
| canonical skills | `1a3d1b503311bf7760110dbf0db9d2ba7c1d9fdf9ce5ebd3ad2a56182e2c40d3` |
| LightGCN CF tensor | `b4d6a16150f79c202314a8ba1f3917ae3106876ee9e8a69ff6f9d68edb928f84` |
| LightGCN manifest | `151774db3e8badb6174646783f6c8bb2bacd5a0a5cb59f45c983aa26f61e83b4` |
| LightGCN skill IDs | `26daa59096f84c50e0c6c07e58736f37f4d36a7f753fdec55d306a31d8276b50` |
| teacher skill tensor | `269ab5985dbb68e38326a3391656b43dbe3c7203d2583ae273ccfd7c49340df7` |
| teacher query manifest | `d5c747450a9c48a00fd7db2d0f939e3c4f7ab853180d9984213ea066077e6115` |
| teacher skill manifest | `de38da5baab0ed7d0c74e81552dbeddf4ae5e94602ed55c0dce2285ad94456ee` |
| faithful M0 checkpoint | `7aa06b562988a55b1a953310b7e89412b4b59c9bfaaa6f7350ee57997d9be9d3` |
| M0 skill embeddings | `34b2ec494ed8679e0552414f137b0530f6d122faac31f5fd870d6b33c9287df1` |
| M0 embedding manifest | `6e31234581291cc1576e20b72dfbfaf66b8de44eca329b9bc9d9e5b7a30ad079` |
| dense candidate manifest | `31e4f13c275399b3ade1b221ecc6e791a8096416a35738064c4a7d4de2e75b04` |
| teacher-scored dense manifest | `2c909bc0212ea6d761b5a3fd9367d2853b24ca198324cb2e224b051d2b8b6aa1` |
| faithful RQ manifest used by preflight | `f64b1c60d4330d912799b5cb1ae0c9ca00ab67037d3b690ae6113b6f886aa4b7` |
| faithful M2 manifest used by preflight | `1e05dce9aef9b6bb1c0731a53c5925c8d5164ae0bdee3b9f4dddbd3951de37da` |
| decoded T5 decoder-start vector | `fe928c4929126abb23e1b22532d21aec65c9a723f2e5fb1f035cfea7da211f02` |

Use `rsync -a` or an equivalent binary-safe transfer. After transfer, run the
launcher once; it stops before training if a required hash or completed
manifest is wrong.

## 4. Reproduction modes

### Mode A: regenerate Qwen embeddings, fused RQ, and M1--M4

This is the current experiment and the recommended portability test. Start
from a fresh experiment directory:

```bash
bash repro/20260816_ripor_cfaware_multisid_testselected/launch.sh \
  2>&1 | tee ripor_m0qwen_cf_multisid.company.log
```

It reuses faithful M0 and immutable teacher/candidate artifacts, generates the
base-Qwen catalog embeddings, rebuilds fused RQ, then trains M1--M4.

### Mode B: hold RQ fixed and test training reproducibility

If Mode A produces a different RQ hash, copy the reference fused RQ directory
and rerun with:

```bash
export RIPOR_REUSE_RQ=/absolute/path/to/reference_fused_rq
bash repro/20260816_ripor_cfaware_multisid_testselected/launch.sh \
  2>&1 | tee ripor_m0qwen_cf_multisid.fixed_rq.company.log
```

Mode B isolates M1--M4 training variability from Faiss/CPU variability.

### Mode C: rebuild faithful M0 first

If the company server does not have the faithful M0 artifacts, first run:

```bash
bash repro/20260813_ripor_faithful/run_full.sh
```

That run still requires the frozen rows, catalog, teacher embeddings, and
Hugging Face caches. After it completes, run Mode A.

## 5. Expected structural audit

Before comparing retrieval metrics, compare the fused RQ manifest. The
reference RQ generated on the RTX PRO 6000 server has:

```text
skills=14551
depth=8
codebook_size=256
seed=42
m0/qwen/cf weights=0.5/0.3/0.2
collision_groups=177
collision_skills=371
semantic_ids_sha256=8d269a3a60fb9704e08ef70c3a4432a1d3415b2e4874105a8dc7f29b9b762132
qwen_procrustes_orthogonality_error=5.96e-7
cf_procrustes_orthogonality_error=3.58e-7
```

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("experiments/ripor_m0qwen_cf_multisid_testselected_cat3query5_20260816")
rq = p / "rq_m0w05_qwenw03_cfw02_depth8_vocab256" / "manifest.json"
m = json.load(rq.open(encoding="utf-8"))
for key in (
    "format", "faiss_version", "seed", "depth", "codebook_size",
    "collision_groups", "collision_skills", "semantic_ids_sha256",
    "rq_index_sha256", "rq_codes_sha256", "rq_codebooks_sha256",
):
    print(f"{key}={m.get(key)}")
PY
```

If `semantic_ids_sha256` differs, downstream checkpoint hashes are not expected
to match. Compare Mode B before attributing the difference to GPU training.

## 6. Result comparison

Each stage writes:

```text
loss_history.jsonl
test_selection_history.jsonl
train_manifest.json
best/structured_decoder.pt
```

Compare these fields stage by stage:

1. `parent_checkpoint_sha256` and `semantic_ids_sha256`;
2. selected epoch and the complete Official core75 metric vector;
3. final loss, gradient norm, and LR trajectory;
4. candidate manifest hashes after M2 and M3 beam mining;
5. final `official_core75_beam50/ranked_metrics.json` and per-query predictions.

The current training code fixes Python, NumPy, PyTorch, sampling, and epoch-order
seeds at `42`, but it does not request bitwise deterministic CUDA algorithms.
Faiss RQ also uses `32` CPU threads. Therefore Mode A is intended to test
metric-level reproducibility; identical checkpoint bytes are not an expected
pass condition across different hardware. Mode B is the cleaner test when the
goal is to measure GPU-training variance alone.
