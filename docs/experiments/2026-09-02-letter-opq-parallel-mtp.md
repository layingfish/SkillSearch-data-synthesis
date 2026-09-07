# LETTER OPQ-32/64 parallel-MTP validation

## Objective

Test the architecture-aligned OPQ hypothesis after the sequential H1 run showed that a long unordered
OPQ tuple is a poor fit for autoregressive trie decoding.  Only two groups are active:

- OPQ-32 parallel-MTP;
- OPQ-64 parallel-MTP.

The result scope is Official Core75 only.  No private-set evaluation is launched or accepted as a result.

## Fixed protocol

- Catalog: the same 14,551 canonical skills.
- DocIDs: frozen OPQ semantic IDs built from the same 1,024-d Qwen catalog embeddings, native Faiss
  position order, codebook size 256, seed 42, and collision-free resolved tuples.
- OPQ-32 logical SID SHA-256:
  `6cfcac4c030a96976ba6fc1a2d1fe0a15b3aab4d61ee63cff22aa3de2a69fe15`.
- OPQ-64 logical SID SHA-256:
  `2d3bf422e62defd26f4183034d5963ab288398f23762547303fe5912a52a0fa9`.
- Backbone: frozen initialization from `t5-base@a9723ea`, then fully trainable in both branches.
- Data pipeline: existing catalog document rows for 3 epochs, then expanded single-SID query rows for
  5 epochs; canonical dev selects the best checkpoint every 1,000 query steps.
- Optimization: AdamW, constant LR `3e-4`, weight decay 0, BF16, seed 20260626, global batch 16.
- Hardware: one `rudder-r2` 8×A800 node, split into two independent 4-rank DDP groups.  OPQ-32 uses
  GPUs 0--3 and port 29632; OPQ-64 uses GPUs 4--7 and port 29664.

The only comparison variable is OPQ depth 32 versus 64.

## Architecture and scoring

The T5 encoder and a single decoder-start token produce one shared query/document hidden state.  Each OPQ
position has its own 256-way linear head, and all positions are predicted simultaneously.  Training uses
the mean cross-entropy over positions per example; this keeps the objective and LR scale comparable for
32 and 64 positions while remaining equivalent to summed MTP up to a constant within each group.

At inference, the model computes all position logits once.  Every one of the 14,551 legal catalog tuples
is scored exactly by the mean gathered per-position log probability.  Ranking therefore has no beam,
trie, prefix ordering, or pruning approximation.  Equal scores use canonical skill ID ascending as a
deterministic tie break.

Implementation:

- `scripts/skill_gr_opq_parallel.py`
- `repro/20260902_letter_opq_32_64/run_rudder_parallel_mtp.sh`
- `repro/20260902_letter_opq_32_64/run_rudder_parallel_1node_split_4x4.sh`
- `repro/20260902_letter_opq_32_64/run_rudder_parallel_node_guard.sh`

## Verification

- Focused tests: 9 passed.  They cover mean-position loss scaling, graph-connected zero-weight DDP tails,
  exact catalog scoring, simultaneous position-permutation invariance, semantic-ID fail-closed checks,
  Official-only evaluation scope, and disjoint GPU/port assignment.
- Python compile, shell syntax, Ruff, Black, and `git diff --check`: passed.
- Tiny T5 depth-32/depth-64 forward and backward: passed; output shapes `[2,32,256]` and `[2,64,256]`,
  all position heads and the encoder receive gradients, and exact catalog scores are finite.
- Direct two-rank `env://` Gloo all-reduce: passed with both ranks observing sum 3.0.
- Real asset audit: both semantic-ID files contain 14,551 unique, in-range, collision-free learned-code
  tuples and match their frozen manifest logical hashes.

## Tianzhi execution

Queue inspection before submission: `rudder-r2` was Open with capability/allocated/headroom
`128/64/64` GPUs.  Eight active r2 workers accounted for the 64 allocated GPUs; nodes `.42/.46/.47`
were Ready, untainted, and had eight unrequested GPUs each.  Normal queue scheduling was used; no node
selector or direct Kubernetes bypass was added.

Failed bootstrap attempts are retained as diagnostics and excluded from results:

1. `tide-chenyufei03-11276-lqtgu`: the `skillsearch-gr` image exposed its system Python distribution from
   `/opt/conda`, while the exact-runtime wrapper requires `/usr/local`; it failed before either branch.
2. `tide-chenyufei03-11277-jbnpw`: `skilldelta:v3` passed the exact runtime, but Tianzhi's elastic launcher
   forced both local `torch.distributed.run` processes onto port 23456 despite distinct CLI ports.  OPQ-64
   failed with EADDRINUSE.  The job was canceled and no partial artifact is accepted.

The valid job is `tide-chenyufei03-11278-bajhf`, image `chenyufei03/skilldelta:v3`, queue
`rudder-r2`, node `192.168.80.47`.  The corrected launcher bypasses the elastic wrapper and starts each
rank directly with explicit `MASTER_PORT/RANK/LOCAL_RANK` environment values.

At the initial acceptance check both branches had completed catalog step 400/2730 with global example
count 16 per optimizer step, finite losses, and no pod restart:

- OPQ-32 step 400 mean-position CE: `4.3381953`;
- OPQ-64 step 400 mean-position CE: `4.5420477`.

At the later handoff check both branches completed all three catalog epochs and transitioned cleanly to
query training, while the pod remained `1/1 Running` with restart count zero:

- OPQ-32 catalog epoch losses: `4.2076065 → 3.3717037 → 2.7800862`; query step `1900/38805`;
- OPQ-64 catalog epoch losses: `4.3800689 → 3.7718955 → 3.2480471`; query step `1650/38805`.

These training losses establish healthy execution only; they are not used to choose a depth. No result is
declared until both Official Core75 exact metric files exist.

Output root:

```text
/mnt/cfs-agent/dataset/chenyufei03/models/experiments/chenyufei03/SkillSearch-data-synthesis/letter_opq_parallel_runs/rudder_r2_parallel_mtp_4gpu_20260902_v3
```

Log root:

```text
/mnt/cfs-agent/dataset/chenyufei03/models/experiments/chenyufei03/SkillSearch-data-synthesis/logs/rudder_r2_parallel_mtp_4gpu_20260902_v3
```

## Final Official Core75 result

Both branches completed successfully on 2026-09-02. OPQ-32 wrote `completion.json` at 16:51 CST and
OPQ-64 at 16:55 CST; both declare `evaluation_scope=official_core75_only`. The pod then exited and was
removed normally.

| Variant | R@1 | R@5 | R@10 | R@50 |
|---|---:|---:|---:|---:|
| OPQ-32 parallel-MTP | 31.2571 | 68.2413 | 74.5556 | **82.6667** |
| OPQ-64 parallel-MTP | **31.8127** | **70.7302** | **76.4444** | 81.4444 |
| OPQ-64 minus OPQ-32 (pp) | +0.5556 | +2.4889 | +1.8889 | -1.2222 |

OPQ-64 is the better choice for the primary top-ranked retrieval range because it improves R@1, R@5,
and R@10. OPQ-32 retains a 1.2222-point advantage at R@50, so the comparison is not a strict dominance.
The result is based on one shared seed and only 75 Official queries; the small differences should not be
presented as a broad ten-point OPQ gain.

Metric artifacts:

- OPQ-32: `opq_m32_parallel_t5base_seed20260626/official_core75_exact/ranked_metrics.json`
  (SHA-256 `75d14d0f69db9304a499181210bf628545ede0b38324860ceaa3da70e1cebd6f`)
- OPQ-64: `opq_m64_parallel_t5base_seed20260626/official_core75_exact/ranked_metrics.json`
  (SHA-256 `c795160a3f76697a5d5a92023b1133cb3487f5e09b6378da83bb1e464821c7c4`)
