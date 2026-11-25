# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RoCEnante for the Inkling-2-Mini SGLang TP4 recipe (opt-in: INKLING_TP4_ROCE=1).

Routes tensor-parallel all-reduces and all-gathers at decode sizes to the b12x
one-shot RoCE collectives; NCCL keeps everything else.  Nothing here runs unless
``tp4/sitecustomize.py`` sees INKLING_TP4_ROCE=1 and loads ``tp4/inkling_sgl_roce.py``,
which puts this directory on ``sys.path``.

Port of our MiniMax-M3.1-Flash TP4 recipe's ``mm31_roce`` (itself a port of our
GLM-5.3 TP4 recipe's ``glm_roce``), re-checked against the SGLang in
``lmsysorg/sglang:inkling-2-mini`` (nightly 03fde9cc, FlashInfer 0.7.4).  The vLLM
hooks of the MiniMax package (``install.py``, the ``.pth`` loader) are not carried
over: this recipe is SGLang only.

Credits:
- RoCEnante (``b12x.comm.roce``): Local Inference Lab, Luke Alonso (@lukealonso)
  and Jason Cook (@original-el8), local-inference-lab/b12x#295 (Apache-2.0).
- vLLM shim the adapter ports: Jason Cook (@original-el8),
  local-inference-lab/vllm#597 (``B12xRoceAllReduce``, the vote, the worker
  health check).
- Port of #597 to an upstream-shaped vLLM tree, including the all-gather switch
  and the 300M spin limit: tonyd2wild,
  GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark ``speed-night-2026-09-18/roce``.
- The idea of routing SGLang's TP all-reduce to RoCEnante: rhys101 (SG17,
  DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8).

Layout:
- ``adapter``: ``InklingRoceAllReduce``, the per-TP-group runtime wrapper.
- ``boot``: the post-import patcher the SGLang hook module installs.
"""

__all__ = ["ENV_ENABLE", "enabled"]

ENV_ENABLE = "INKLING_TP4_ROCE"


def enabled(environ=None) -> bool:
    """True when the container asked for RoCEnante (``INKLING_TP4_ROCE`` set and not 0/off)."""
    import os

    env = os.environ if environ is None else environ
    return env.get(ENV_ENABLE, "").strip().lower() not in ("", "0", "off", "false", "no")
