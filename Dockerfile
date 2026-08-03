#
# CUDA-enabled development image for SkillSearch / SkillGR experiments.
# The base image is pinned to the verified Qwen3-VL runtime so that torch and
# CUDA are not resolved again during the build.
ARG BASE_IMAGE=ccr-2663zxft-pub.cnc.bj.baidubce.com/qianfan-train/chenyufei03/taarq-qwen3vl@sha256:ed01c7da29e6214544f0ec70ec0aa05411e24887482bbba95eb9f5553b8f40e5
FROM ${BASE_IMAGE}

ARG LIGHTGCN_COMMIT=947ca2b3b1d2d3545b114145710cb06c4e57b3d2
ARG LETTER_COMMIT=8d0154e28de37dbb6e24871c508ad8ddb1921cda
ARG RIPOR_COMMIT=03fd0cbfafdeeb9cdd866e0c44727fec39a96a52
ARG PAG_COMMIT=49802454fb157fc659c1c6c42bb2d940a9f6bc1b

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    SKILLSEARCH_HOME=/opt/skillsearch/project \
    SKILLSEARCH_LIGHTGCN_CODE_DIR=/opt/skillsearch/external/LightGCN-PyTorch/code \
    SKILLSEARCH_LETTER_RQVAE_DIR=/opt/skillsearch/external/LETTER/RQ-VAE \
    SKILLSEARCH_RIPOR_DIR=/opt/skillsearch/reference/RIPOR \
    SKILLSEARCH_PAG_DIR=/opt/skillsearch/reference/PAG \
    PYTHONPATH=/opt/skillsearch/python

# Do not install torch here.  The pinned base provides the CUDA 12.8 torch 2.8
# runtime; the project requirements below are CPU/Python-level additions only.
COPY requirements-image.txt /tmp/skillsearch-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/skillsearch-requirements.txt

# The two adapters in scripts/ intentionally import the upstream LightGCN and
# LETTER implementations.  Pin them instead of relying on a mutable branch.
# RIPOR and PAG are included as source references only: their official
# requirements target torch 1.10/CUDA 11.1 and are deliberately not installed.
RUN mkdir -p /opt/skillsearch/external /opt/skillsearch/reference \
    && git clone https://github.com/gusye1234/LightGCN-PyTorch.git /opt/skillsearch/external/LightGCN-PyTorch \
    && git -C /opt/skillsearch/external/LightGCN-PyTorch checkout --detach "${LIGHTGCN_COMMIT}" \
    && test "$(git -C /opt/skillsearch/external/LightGCN-PyTorch rev-parse HEAD)" = "${LIGHTGCN_COMMIT}" \
    && git clone https://github.com/HonghuiBao2000/LETTER.git /opt/skillsearch/external/LETTER \
    && git -C /opt/skillsearch/external/LETTER checkout --detach "${LETTER_COMMIT}" \
    && test "$(git -C /opt/skillsearch/external/LETTER rev-parse HEAD)" = "${LETTER_COMMIT}" \
    && git clone https://github.com/HansiZeng/RIPOR.git /opt/skillsearch/reference/RIPOR \
    && git -C /opt/skillsearch/reference/RIPOR checkout --detach "${RIPOR_COMMIT}" \
    && test "$(git -C /opt/skillsearch/reference/RIPOR rev-parse HEAD)" = "${RIPOR_COMMIT}" \
    && git clone https://github.com/HansiZeng/PAG.git /opt/skillsearch/reference/PAG \
    && git -C /opt/skillsearch/reference/PAG checkout --detach "${PAG_COMMIT}" \
    && test "$(git -C /opt/skillsearch/reference/PAG rev-parse HEAD)" = "${PAG_COMMIT}"

WORKDIR /opt/skillsearch/project
COPY . /opt/skillsearch/project

# Existing training commands can use paths relative to the project root.
RUN test ! -e /opt/skillsearch/project/external \
    && ln -s /opt/skillsearch/external /opt/skillsearch/project/external \
    && python3 docker/smoke_test.py

CMD ["/bin/bash"]
