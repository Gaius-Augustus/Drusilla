"""Run OrfAnnotationHMM Viterbi decoding on model outputs.

Wraps :class:`OrfAnnotationHMM` for the inference pipeline. The only fiddly
bit is the parallel-scan divisibility constraint of ``hidten.tf.TFHMM``
(``T % parallel == 0``): sequences are right-padded with a uniform-emission
suffix that the HMM can freely traverse, and the padding is sliced off
after decoding.

Padding strategy at padded positions:

* ``logits`` set to zero -> softmax is uniform -> stream-emitter
  contribution is a constant ``1/6`` for every state.
* ``nuc`` set to N (channel 4) -> ``make_kmer`` produces a uniform
  distribution over the 64 codons -> each codon emitter contributes a
  constant ``1/64`` for every state.

With every padded-position emission identical across all states, Viterbi
traceback through the suffix follows transition probabilities only and
does not bias the labelling of the un-padded prefix.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from hidten import HMMMode

from .annotation_hmm import OrfAnnotationHMM


DEFAULT_PARALLEL = 100


def build_decoder_hmm(
    parallel: int = DEFAULT_PARALLEL,
    initial_ir_len: float = 300.0,
    initial_exon_len: float = 600.0,
    emitter_epsilon: float = 0.01,
    use_codon_emitter: bool = True,
    restrict_start_to_ir_start: bool = False,
    ir_start_prior_ir: float = 0.5,
) -> OrfAnnotationHMM:
    """Build a Viterbi-mode :class:`OrfAnnotationHMM` ready for decoding."""
    return OrfAnnotationHMM(
        mode=HMMMode.VITERBI,
        parallel=parallel,
        initial_ir_len=initial_ir_len,
        initial_exon_len=initial_exon_len,
        emitter_epsilon=emitter_epsilon,
        use_codon_emitter=use_codon_emitter,
        restrict_start_to_ir_start=restrict_start_to_ir_start,
        ir_start_prior_ir=ir_start_prior_ir,
    )


def _pad_for_parallel(
    logits: np.ndarray,
    nuc: np.ndarray,
    parallel: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Right-pad along the second-to-last axis to a multiple of ``parallel``."""
    L = logits.shape[-2]
    rem = L % parallel
    if rem == 0:
        return logits, nuc, 0
    pad = parallel - rem

    logits_pw = [(0, 0)] * logits.ndim
    logits_pw[-2] = (0, pad)
    logits = np.pad(logits, logits_pw, mode="constant", constant_values=0.0)

    nuc_pw = [(0, 0)] * nuc.ndim
    nuc_pw[-2] = (0, pad)
    nuc = np.pad(nuc, nuc_pw, mode="constant", constant_values=0.0)

    sl: list = [slice(None)] * nuc.ndim
    sl[-2] = slice(L, L + pad)
    nuc[tuple(sl[:-1] + [4])] = 1.0

    return logits, nuc, pad


def _normalise_pad_mask(
    nuc: np.ndarray,
    pad_mask: np.ndarray | None,
) -> np.ndarray:
    """Force padded positions to N so the codon emitter doesn't see all-zero rows."""
    if pad_mask is None:
        return nuc
    nuc = nuc.copy()
    nuc[pad_mask] = 0.0
    nuc[pad_mask, 4] = 1.0
    return nuc


def _compiled_call(hmm: OrfAnnotationHMM):
    """Cache a tf.function-wrapped HMM call on the layer instance."""
    if not hasattr(hmm, "_tf_decode_fn"):
        @tf.function(reduce_retracing=True)
        def _fn(probs, nuc):
            return hmm(probs, nuc)
        hmm._tf_decode_fn = _fn
    return hmm._tf_decode_fn


def viterbi_decode_batch(
    hmm: OrfAnnotationHMM,
    logits: np.ndarray,
    nuc: np.ndarray,
    pad_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Decode a batch of sequences via the HMM in Viterbi mode.

    Args:
        hmm: A Viterbi-mode :class:`OrfAnnotationHMM`.
        logits: NN logits of shape ``(B, L, 6)``.
        nuc: ACGTN one-hot of shape ``(B, L, 5)``.
        pad_mask: Optional ``(B, L)`` bool array marking padded positions.

    Returns:
        Int32 array of shape ``(B, L)`` with state indices in ``{0..5}``.
    """
    nuc = _normalise_pad_mask(nuc, pad_mask)
    L_in = logits.shape[1]
    logits, nuc, _pad = _pad_for_parallel(logits, nuc, hmm.parallel)

    probs = tf.nn.softmax(tf.constant(logits, dtype=tf.float32))
    nuc_t = tf.constant(nuc, dtype=tf.float32)
    out = _compiled_call(hmm)(probs, nuc_t).numpy()
    out = out[..., 0].astype(np.int32)
    return out[:, :L_in]


def viterbi_decode_one(
    hmm: OrfAnnotationHMM,
    logits: np.ndarray,
    nuc: np.ndarray,
) -> np.ndarray:
    """Decode a single sequence (B=1 wrapper)."""
    return viterbi_decode_batch(
        hmm, logits[None, ...], nuc[None, ...],
    )[0]
