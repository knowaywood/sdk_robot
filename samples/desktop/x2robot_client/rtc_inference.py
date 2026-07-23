import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def smooth_chunk_boundary(
    prev_outputs: Dict[str, np.ndarray],
    new_outputs: Dict[str, np.ndarray],
    blend_steps: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Blend the beginning of new action chunks toward the last step of the
    previous chunk, producing smooth transitions across chunk boundaries.

    For each action array in the intersection of prev_outputs and new_outputs:
        blended[t] = α · prev_last + (1-α) · new_chunk[t]    for t < blend_steps
        blended[t] = new_chunk[t]                              for t >= blend_steps

    where α = cos(t · π / (2 · blend_steps)) decays from 1 to 0.

    Args:
        prev_outputs: Action chunks from the previous inference call.
        new_outputs:  Action chunks from the current inference call.
        blend_steps:  Number of initial steps to blend (default 3).

    Returns:
        Blended action chunks dict. Keys that only appear in one of the
        inputs are passed through unchanged.
    """
    if prev_outputs is None:
        return new_outputs

    blended = {}

    for key in new_outputs:
        new_arr = new_outputs[key]
        if new_arr is None:
            blended[key] = new_arr
            continue

        new_arr = np.asarray(new_arr, dtype=np.float64)
        if new_arr.ndim != 2 or new_arr.shape[0] < 1:
            blended[key] = new_arr
            continue

        if key not in prev_outputs:
            blended[key] = new_arr
            continue

        prev_arr = prev_outputs[key]
        if prev_arr is None:
            blended[key] = new_arr
            continue

        prev_arr = np.asarray(prev_arr, dtype=np.float64)
        if prev_arr.ndim != 2 or prev_arr.shape[0] < 1:
            blended[key] = new_arr
            continue

        blended[key] = _smooth_new_chunk(prev_arr, new_arr, blend_steps)

    return blended


def _smooth_new_chunk(
    prev_chunk: np.ndarray,
    new_chunk: np.ndarray,
    blend_steps: int,
) -> np.ndarray:
    """Apply cosine-decay blending to a single pair of chunks."""
    prev_chunk = np.asarray(prev_chunk, dtype=np.float64)
    new_chunk = np.asarray(new_chunk, dtype=np.float64)

    out = new_chunk.copy()

    b = min(blend_steps, new_chunk.shape[0])
    if b <= 0:
        return out

    prev_last = prev_chunk[-1]

    for t in range(b):
        alpha = np.cos(t * np.pi / (2.0 * b))
        out[t] = alpha * prev_last + (1.0 - alpha) * new_chunk[t]

    return out


def compute_rtc_blend(
    prev_chunk: np.ndarray,
    new_chunk: np.ndarray,
    delay_steps: int,
    overlap_steps: int,
) -> np.ndarray:
    """
    Three-zone RTC-style blending (reserved for Phase 3 / server-side use).

    Freeze zone:   t < delay_steps              -> blended[t] = prev[t]
    Overlap zone:  delay_steps <= t < delay_steps + overlap_steps
                   -> α = 1 - (t - delay_steps) / overlap_steps
                   -> blended[t] = (1-α)·new[t-delay_steps] + α·prev[t]
    Free zone:     t >= delay_steps + overlap_steps
                   -> blended[t] = new[t-delay_steps]
    """
    prev_chunk = np.asarray(prev_chunk, dtype=np.float64)
    new_chunk = np.asarray(new_chunk, dtype=np.float64)
    K = new_chunk.shape[0]

    out = np.zeros_like(new_chunk)

    for t in range(K):
        if t < delay_steps:
            if t < prev_chunk.shape[0]:
                out[t] = prev_chunk[t]
            else:
                out[t] = new_chunk[0]
        elif t < delay_steps + overlap_steps:
            alpha = 1.0 - (t - delay_steps) / overlap_steps
            prev_t = prev_chunk[t] if t < prev_chunk.shape[0] else prev_chunk[-1]
            out[t] = (1.0 - alpha) * new_chunk[t - delay_steps] + alpha * prev_t
        else:
            out[t] = new_chunk[t - delay_steps]

    return out
