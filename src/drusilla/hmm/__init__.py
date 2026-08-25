from .viterbi import viterbi_decode, viterbi_decode_batch

__all__ = ["viterbi_decode", "viterbi_decode_batch"]

# OrfAnnotationHMM and its setup helpers depend on TensorFlow / hidten /
# bricks2marble. Import lazily so importing drusilla.hmm in a numpy-only
# environment keeps working.
try:
    from .annotation_hmm import (
        ALLOWED_TRANSITIONS,
        DEFAULT_START_CODONS,
        DEFAULT_STOP_CODONS,
        N_STATES,
        STATE_NAMES,
        OrfAnnotationHMM,
        codon_emissions,
        codon_hint_emission,
        eye_emission,
        state_start_dist,
        state_transitions,
    )
    __all__ += [
        "ALLOWED_TRANSITIONS",
        "DEFAULT_START_CODONS",
        "DEFAULT_STOP_CODONS",
        "N_STATES",
        "STATE_NAMES",
        "OrfAnnotationHMM",
        "codon_emissions",
        "codon_hint_emission",
        "eye_emission",
        "state_start_dist",
        "state_transitions",
    ]
except ImportError:
    pass
