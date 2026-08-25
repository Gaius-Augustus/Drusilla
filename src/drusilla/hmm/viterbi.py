"""Pure-numpy Viterbi decoder for the 6-state ORF label HMM.

States (matching ``label_transcripts.build_labels``):
    0 IR    1 START    2 E1    3 E2    4 E0    5 STOP

A canonical CDS of length N labelled by ``build_labels`` is::

    START E1 E2 E0 E1 E2 E0 ... E1 STOP

so the codon containing the stop is ``(E0, E1, STOP)`` and the transition
immediately before STOP is ``E1 -> STOP``.

Valid transitions (all others have -inf log-probability):
    IR    -> IR, START
    START -> E1
    E1    -> E2, STOP
    E2    -> E0
    E0    -> E1
    STOP  -> IR
"""

from __future__ import annotations

import numpy as np


IR, START, E1, E2, E0, STOP = 0, 1, 2, 3, 4, 5
N_STATES = 6

_NEG_INF = -1e30

_TRANS_ALLOWED: list[tuple[int, int]] = [
    (IR,    IR),
    (IR,    START),
    (START, E1),
    (E1,    E2),
    (E1,    STOP),
    (E2,    E0),
    (E0,    E1),
    (STOP,  IR),
]

_LOG_TRANS = np.full((N_STATES, N_STATES), _NEG_INF, dtype=np.float64)
for _f, _t in _TRANS_ALLOWED:
    _LOG_TRANS[_f, _t] = 0.0


def viterbi_decode(log_emission: np.ndarray) -> np.ndarray:
    """Run Viterbi decoding on per-position log-probabilities.

    Args:
        log_emission: float array of shape [L, 6] - log-prob (or logit)
            for each state at each position. Typically log_softmax(logits).

    Returns:
        int32 array of shape [L] with the most probable valid label sequence.
    """
    log_emission = np.asarray(log_emission, dtype=np.float64)
    L = log_emission.shape[0]
    if L == 0:
        return np.empty(0, dtype=np.int32)

    viterbi = np.full((L, N_STATES), _NEG_INF, dtype=np.float64)
    backptr = np.zeros((L, N_STATES), dtype=np.int32)

    viterbi[0] = log_emission[0]

    for t in range(1, L):
        scores = viterbi[t - 1, :, None] + _LOG_TRANS
        backptr[t] = np.argmax(scores, axis=0)
        viterbi[t] = scores[backptr[t], np.arange(N_STATES)] + log_emission[t]

    path = np.empty(L, dtype=np.int32)
    path[L - 1] = int(np.argmax(viterbi[L - 1]))
    for t in range(L - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]

    return path


def viterbi_decode_batch(log_emission: np.ndarray) -> np.ndarray:
    """Viterbi decode a batch of sequences of shape [B, L, 6]."""
    log_emission = np.asarray(log_emission, dtype=np.float64)
    B = log_emission.shape[0]
    return np.stack([viterbi_decode(log_emission[b]) for b in range(B)])
