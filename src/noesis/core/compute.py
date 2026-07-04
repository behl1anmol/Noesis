"""Compute-device resolution for the model-loading boundaries (M4 follow-up).

Both local model runners (``LocalSTEmbedder``, ``LocalCrossEncoderReranker``)
resolve their device through :func:`resolve_device` instead of relying on
sentence-transformers' implicit ``device=None`` auto-detect — that auto-detect
was observed returning CPU on a CUDA-available box, silently running the
cross-encoder on CPU and producing an uninterpretable latency benchmark
(lesson 4). Resolving explicitly and logging the result makes device a
recorded, reproducible property of every run.

``torch`` is imported lazily inside the function so importing ``core`` (e.g.
for the FakeEmbedder test suite) never pays the heavy torch import.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_device(configured: str | None = None) -> str:
    """Return the compute device string to hand a model.

    A non-empty ``configured`` value (from ``[embedder]``/``[reranker]``
    ``device`` in config.toml) wins verbatim — the operator's explicit choice
    is never second-guessed. Otherwise auto-detect in preference order
    cuda → mps → cpu. The result is logged so a latency benchmark always
    records the hardware it ran on.
    """
    if configured:
        logger.info("compute device: %s (configured)", configured)
        return configured
    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("compute device: %s (auto-detected)", device)
    return device
