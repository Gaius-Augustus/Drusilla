# Drusilla: deep-learning ORF annotator for RNA-seq assembled transcripts.
#
# Includes: Drusilla with the bundled drusilla_model (no download at build or
# run time), gffread, StringTie, pysam (strand-fix-bam) and lightgbm /
# scikit-learn / pyfaidx for the LightGBM filter of Paludamentum's Drusilla flow.
#
# The image is built from the checked-out source tree, not from a git clone,
# so it contains exactly the commit you build from. Build context is the
# repository root (.dockerignore keeps it to the package inputs):
#
#   docker/build_push.sh                      # build + push, tag from pyproject
#   docker build -f docker/Dockerfile -t gaiusaugustus/drusilla:<tag> .
#
# Convert to Singularity / Apptainer:
#   singularity build drusilla.sif docker://gaiusaugustus/drusilla:latest
#
# Usage (GPU):
#   singularity run --nv -B $PWD:/work drusilla.sif annotate \
#       --assembly-gtf /work/stringtie.gtf --genome /work/genome.fa \
#       --out-dir /work/results
#
# The TensorFlow / Keras handling follows docker/drusilla/Dockerfile of
# Paludamentum, whose image runs the same code.

# The NGC TensorFlow build is the only one that runs on Blackwell GPUs (sm_120).
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Record the Python packages of the NGC base image. The check at the end of
# this file fails the build if a pip install replaced the NGC TensorFlow or
# put PyPI CUDA/cuDNN wheels next to it (which breaks Blackwell GPUs).
RUN python3 -c "import importlib.metadata as m; print('\n'.join(sorted(d.metadata['Name'].lower().replace('_', '-') + '==' + d.version for d in m.distributions())))" > /opt/ngc-base-packages.txt

# The NGC image points tf.keras at Keras 2 (tf_keras). Drusilla's HMM layer,
# hidten and bricks2marble are written against Keras 3 (tf.keras.Layer).
ENV TF_USE_LEGACY_KERAS=0
RUN python3 -m pip install --no-cache-dir --upgrade "keras>=3,<4"

# External tools Drusilla shells out to:
#   gffread   — always (transcript extraction)
#   stringtie — only for BAM input
RUN apt-get update && apt-get install -y --no-install-recommends \
        gffread \
        stringtie \
    && rm -rf /var/lib/apt/lists/*

# bricks2marble[tf] -> hidten[tensorflow] -> tensorflow[and-cuda], whose
# and-cuda extra installs PyPI CUDA wheels on top of the NGC stack. So the
# TF-dependent packages are installed without dependencies and the rest
# explicitly: numpy / pydantic (bricks2marble, hidten), pyyaml / pyfaidx /
# pysam (Drusilla), biopython / pandas, and for the LightGBM filter of
# Paludamentum's Drusilla flow lightgbm with scikit-learn (the model is a
# pickled LGBMClassifier).
RUN python3 -m pip install --no-cache-dir --no-deps bricks2marble hidten && \
    python3 -m pip install --no-cache-dir \
        numpy \
        pydantic \
        pyyaml \
        pyfaidx \
        pysam \
        biopython \
        pandas \
        lightgbm \
        scikit-learn

# Drusilla from the build context. The wheel build (hatchling force-include in
# pyproject.toml) needs model_cfg/, configs/ and models/ next to src/, so the
# repository layout is reproduced under /tmp/drusilla. --no-deps: its
# dependencies are installed above.
COPY pyproject.toml README.md LICENSE /tmp/drusilla/
COPY src       /tmp/drusilla/src
COPY model_cfg /tmp/drusilla/model_cfg
COPY configs   /tmp/drusilla/configs
COPY models    /tmp/drusilla/models
RUN python3 -m pip install --no-cache-dir --no-deps /tmp/drusilla && rm -rf /tmp/drusilla

ARG DRUSILLA_GIT_COMMIT=unknown
LABEL org.opencontainers.image.source="https://github.com/Gaius-Augustus/Drusilla" \
      org.opencontainers.image.revision="${DRUSILLA_GIT_COMMIT}"

# Smoke test: the NGC TensorFlow/CUDA stack is unmodified, tf.keras is
# Keras 3, every subcommand runs, the bundled model resolves, the HMM layer
# imports (TF / hidten / bricks2marble), the Python extras import and every
# external binary is present.
RUN python3 -c "import importlib.metadata as m, sys; \
base = set(open('/opt/ngc-base-packages.txt').read().split()); \
now = {d.metadata['Name'].lower().replace('_', '-') + '==' + d.version for d in m.distributions()}; \
added = sorted(p for p in now - base if p.startswith(('nvidia-', 'tensorflow'))); \
tf = m.version('tensorflow'); \
sys.exit(f'NGC TensorFlow stack modified: tensorflow=={tf}, added/changed: {added}' if '+nv' not in tf or added else 0)" && \
    python3 -c "import keras, sys; sys.exit(0 if keras.__version__.startswith('3.') else f'Keras {keras.__version__} is not Keras 3')" && \
    drusilla --version && \
    drusilla annotate --help > /dev/null && \
    drusilla filter-subseq --help > /dev/null && \
    drusilla strand-fix-bam --help > /dev/null && \
    drusilla models path drusilla_model && \
    python3 -c "from drusilla.hmm.annotation_hmm import OrfAnnotationHMM" && \
    python3 -c "import pysam, pyfaidx, lightgbm, sklearn, Bio, pandas, yaml, pydantic" && \
    for b in gffread stringtie; do \
        command -v "$b" > /dev/null || { echo "missing: $b"; exit 1; }; \
    done

ENTRYPOINT ["drusilla"]
CMD ["--help"]
