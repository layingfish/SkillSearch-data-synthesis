# SkillSearch / SkillGR development image

The project image is published as `chenyufei03/skillsearch-gr:cu128-torch28-v1`.
It is built from the verified Qwen3-VL CUDA 12.8 / torch 2.8 runtime, rather
than from the older torch 1.10 / CUDA 11.1 environments used by RIPOR and PAG.
The published image digest is
`sha256:dff4bf15312f843586cfdf7d86bf8636a7c82f9ebd1c1a3d896a98b827d88f5c`.

## Included components

- Project checkout at `/opt/skillsearch/project` (the default working directory).
- Official LightGCN-PyTorch at
  `/opt/skillsearch/external/LightGCN-PyTorch` pinned to
  `947ca2b3b1d2d3545b114145710cb06c4e57b3d2`.
- Official LETTER at `/opt/skillsearch/external/LETTER` pinned to
  `8d0154e28de37dbb6e24871c508ad8ddb1921cda`.  The adapter-facing RQ-VAE path
  is `/opt/skillsearch/external/LETTER/RQ-VAE`.
- RIPOR at `/opt/skillsearch/reference/RIPOR` pinned to
  `03fd0cbfafdeeb9cdd866e0c44727fec39a96a52`, and PAG at
  `/opt/skillsearch/reference/PAG` pinned to
  `49802454fb157fc659c1c6c42bb2d940a9f6bc1b`.

RIPOR and PAG are source references in this image. Their upstream requirements
are deliberately not installed because they pin an incompatible torch 1.10 /
CUDA 11.1 stack. The image instead contains the dependencies required by this
repository's adapters and training scripts.

The project has an `external` symlink to `/opt/skillsearch/external`, so the
handoff commands can use relative paths such as
`external/LightGCN-PyTorch/code` and `external/LETTER/RQ-VAE`.

## Validation and use

The Dockerfile runs the offline smoke test below during a conventional build.
It validates the imported Python dependencies, all pinned source revisions, and
the three project modules that consume the environment. The published image's
OCI configuration and added layer were also verified remotely; run this command
in a job for an execution-level check.

```bash
python3 docker/smoke_test.py
```

For a Tianzhi job, pass the short image name:

```text
--image chenyufei03/skillsearch-gr:cu128-torch28-v1
```

Datasets, checkpoints, outputs, local logs, and secrets are excluded from the
image. Mount the appropriate writable CFS paths at job runtime.
