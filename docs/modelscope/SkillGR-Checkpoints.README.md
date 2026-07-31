---
license: other
frameworks:
  - pytorch
tasks:
  - text-retrieval
---

# SkillGR Checkpoints

Private model and evaluation artifacts for SkillSearch / SkillGR.

Accepted mainline contents:

- LightGCN 32-dimensional collaborative skill embeddings;
- LETTER official-style four-level learned Semantic IDs;
- task5 base single-SID T5 decoder;
- task5+mode3 single-SID T5 decoder;
- task5+mode3 multi-SID T5 decoder;
- reproduced Qwen3-Embedding-0.6B SkillsRouter checkpoint;
- Official core75 and private-test predictions, metrics and manifests.

The GR decoder uses `t5-base`, a custom four-code target embedding/head layout, a CONTINUE/STOP control head and trie-constrained beam search. The stored checkpoint metadata binds each decoder to the exact Semantic-ID schema and hash.

Paths are preserved relative to the original `outputs/` tree. Merge this repository with the companion private dataset repository after download.

See `HANDOFF_20260731.md` in the private GitHub repository `layingfish/SkillSearch-data-synthesis` for architecture, training configuration, evaluation protocol, restoration commands and known limitations.

This repository is private. No API keys or access tokens are stored here.
