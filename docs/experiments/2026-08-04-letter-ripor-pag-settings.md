# Experiment: Fixed-LETTER RIPOR/PAG component settings
Status: running
Risk Tier: 3

## Goal / Hypothesis / Change Scope

Measure the contribution of RIPOR decoder ranking and PAG lexical Set IDs while keeping the existing
Qwen + LightGCN + LETTER Semantic IDs immutable. The three requested final settings are:

1. `ripor`: fixed LETTER SID + prefix-oriented MarginMSE.
2. `pag`: fixed LETTER SID CE baseline + PAG set loss/planning-ahead.
3. `ripor_pag`: fixed LETTER SID + prefix-oriented MarginMSE + PAG set loss/planning-ahead.

All settings use the same single-SID target expansion, train/dev rows, canonical catalog, LETTER SID
hash, CE+LETTER base decoder, numeric teacher panel, seed, per-device batch, 8-GPU world size, source
policy, and final-stage step budget. PAG Msp/Mset and RIPOR Mseq are each trained once and shared.
PAG-only merges Mset with the untouched CE+LETTER base; RIPOR+PAG merges the same Mset with the shared
fixed-LETTER RIPOR Mseq. Sequential SID/code/control heads always come bitwise from the corresponding
sequential model.

## Commands

Train:

```bash
bash scripts/run_letter_ripor_pag_settings.sh <stage>
```

The launcher sources `configs/letter_ripor_pag_20260804.env`. The execution DAG is `teacher → preflight
→ {pag_shared,ripor_mseq} → {final_ripor,final_pag,final_ripor_pag} → four eval stages`; independent
stages are submitted as one-node 8-GPU jobs. The teacher stage writes `setting_contract.json` with
input SHA-256 hashes before method training.

Locked protocol: Msp 4700 steps with the official PAG sparse-encoder `max_length=128`, Mset 5000,
shared RIPOR Mseq 5000, each final stage 5000; per-device
batch 8 (global 64), AdamW with zero weight decay, LR 5e-4, 4.5% linear warmup followed by linear
decay, BF16, seed 20260626, PAG m=64/top-N=1000/weight=1, and beam=50.

Eval:

The launcher evaluates all three final checkpoints with 8-GPU single-SID constrained decoding. PAG
settings use the same lexical set manifest, planning weight, top-N candidate domain, beam count, and K.

## Outputs

Checkpoint: `/models_rw/SkillSearch-data-synthesis/experiments/letter_ripor_pag_single_sid_20260804_v1/{ripor_only,pag_only,ripor_pag}/final`

Log: `/models_rw/SkillSearch-data-synthesis/experiments/letter_ripor_pag_single_sid_20260804_v1/logs/`

Metrics: `${SKILLGR_OUTPUT_ROOT}/{ripor_only,pag_only,ripor_pag}/eval/ranked_metrics.json`

## Verification

- [x] Minimal local syntax/config check
- [ ] LETTER manifest/SID fingerprint invariant
- [ ] PAG merge preserves Mseq SID embedding/code/control heads bitwise
- [x] Focused loss-setting tests (53 passed, 7 dependency-gated skipped)
- [x] DDP global sum/count invariant on 8 ranks (`max_gradient_error=5.96e-08`)
- [x] Reviewer required? yes — independent plan review completed; CRITICAL/HIGH items incorporated or explicitly resolved

## Result / Notes

Formal train/dev/catalog, fixed LETTER, base single-SID decoder, SkillRouter teacher, and pinned T5-base
assets are resolved on CFS. Independent cross-review found no remaining CRITICAL/HIGH findings. Teacher
job `tide-chenyufei03-10230-hlrjf` failed before training because `/dataset_rw` was not initialized by
the alias helper. The launcher now passes `--with-dataset-rw` and starts CFS logging before validation.

Retry `tide-chenyufei03-10231-rshfg` completed distributed encoding but failed before scoring because
query embeddings were on CPU while the gathered catalog matrix was on each rank's GPU. The query
matrix is now moved to the local rank device immediately before matmul; this is a device-only fix.

Teacher retry `tide-chenyufei03-10232-nrekv` succeeded and produced 100078 ranked rows plus the
SHA-bound setting contract. The teacher metrics are Hit@1=0.83119167 and Hit@50=0.99351506.

Preflight `tide-chenyufei03-10233-hsqmj` passed all 53 image tests and the 8-rank DDP invariant, then
failed in real-model PAG Msp with CUDA OOM. Root cause: the launcher incorrectly reused the GR decoder
`max_source_length=2048` for T5-SPLADE. The official PAG `t5_splade_train.sh` fixes sparse-encoder
`max_length=128`; the locked config, setting contract, and launcher now use 128 for Msp while retaining
2048 for Mset/RIPOR/joint stages. A fresh preflight is required before formal training.
