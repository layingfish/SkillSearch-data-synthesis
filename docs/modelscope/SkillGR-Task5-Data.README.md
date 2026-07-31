---
license: other
task_categories:
  - text-retrieval
language:
  - en
---

# SkillGR Task5 Data

Private reproducibility dataset for SkillSearch / SkillGR experiments.

It contains the accepted mainline assets only:

- full-catalog task-style pseudo queries;
- task5 base and mode3-enhanced decoder rows;
- exact-document canonical skills and alias maps;
- canonical skill embeddings;
- train/dev/test and Official evaluation inputs;
- SkillsRouter single-only inputs and mined hard negatives;
- source catalog and the balanced 2k seed-data artifacts.

The canonical candidate pool has 14,551 skills. The main task5 dataset has 94,912 source rows, and the enhanced dataset has 100,078 source rows.

Paths are preserved relative to the original `outputs/` tree so this repository can be merged with the companion model repository after download.

See `HANDOFF_20260731.md` in the private GitHub repository `layingfish/SkillSearch-data-synthesis` for data contracts, hashes, leakage boundaries, experiment semantics, restoration commands and operational requirements.

This repository is private because the source skill-document licenses and provider terms must be reviewed before redistribution. No API keys or access tokens are stored here.
