# Drusilla: deep-learning ORF annotator for RNA-seq assembled transcripts.
#
# Includes: Drusilla with the bundled drusilla_model, gffread, StringTie,
# pysam (strand-fix-bam) and lightgbm / scikit-learn / pyfaidx for the
# LightGBM filter of Paludamentum's Drusilla flow.
#
# The image is built from the checked-out source tree, not from a git clone,
# so it contains exactly the commit you build from. Build context is the
# repository root (.dockerignore keeps it to the package inputs):
#
#   docker/build_push.sh                      # build + push, tag from pyproject
#   docker build -f docker/Dockerfile -t larsgabriel23/drusilla:<tag> .
#
# Convert to Singularity / Apptainer:
#   singularity build drusilla.sif docker://larsgabriel23/drusilla:latest
#
# Usage (GPU):
#   singularity run --nv -B $PWD:/work drusilla.sif annotate \
#       --assembly-gtf /work/stringtie.gtf --genome /work/genome.fa \
#       --out-dir /work/results

FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# External tools Drusilla shells out to:
#   gffread   — always (transcript extraction)
#   stringtie — only for BAM input
RUN apt-get update && apt-get install -y --no-install-recommends \
        gffread \
        stringtie \
    && rm -rf /var/lib/apt/lists/*

# Python extras from the [gpu] extra, MINUS tensorflow — the NGC base image
# already ships a CUDA-enabled TensorFlow build, and installing
# tensorflow[and-cuda] on top of it would replace the NVIDIA wheel with
# an upstream one and break GPU acceleration.
RUN pip install --no-cache-dir \
        "bricks2marble[tf]" \
        hidten \
        biopython \
        pandas

# LightGBM filter of gene finder predictions (Paludamentum's Drusilla flow):
# the model is a pickled lightgbm LGBMClassifier, which needs scikit-learn to
# load; the ORF features read the genome with pyfaidx. pysam: strand-fix-bam.
RUN pip install --no-cache-dir \
        lightgbm \
        scikit-learn \
        pyfaidx \
        pysam

# Drusilla from the build context. The wheel build (hatchling force-include in
# pyproject.toml) needs model_cfg/, configs/ and models/ next to src/, so the
# repository layout is reproduced under /tmp/drusilla.
COPY pyproject.toml README.md LICENSE /tmp/drusilla/
COPY src       /tmp/drusilla/src
COPY model_cfg /tmp/drusilla/model_cfg
COPY configs   /tmp/drusilla/configs
COPY models    /tmp/drusilla/models
RUN pip install --no-cache-dir /tmp/drusilla && rm -rf /tmp/drusilla

ARG DRUSILLA_GIT_COMMIT=unknown
LABEL org.opencontainers.image.source="https://github.com/Gaius-Augustus/Drusilla" \
      org.opencontainers.image.revision="${DRUSILLA_GIT_COMMIT}"

# Cache for the legacy per-clade model archives; drusilla_model is bundled
# and needs no download. World-writable so non-root container users can fill it.
ENV DRUSILLA_CACHE_DIR=/data/drusilla_cache
RUN mkdir -p /data/drusilla_cache && chmod 777 /data/drusilla_cache

# Smoke test: entry point and every subcommand, the bundled model, the
# TF / hidten / bricks2marble stack behind the HMM layer, the Python extras
# and every external binary.
RUN drusilla --version && \
    drusilla annotate --help > /dev/null && \
    drusilla filter-subseq --help > /dev/null && \
    drusilla strand-fix-bam --help > /dev/null && \
    drusilla models path drusilla_model && \
    python3 -c "from drusilla.hmm.annotation_hmm import OrfAnnotationHMM" && \
    python3 -c "import pysam, pyfaidx, lightgbm, sklearn, Bio, pandas" && \
    for b in gffread stringtie; do \
        command -v "$b" > /dev/null || { echo "missing: $b"; exit 1; }; \
    done

ENTRYPOINT ["drusilla"]
CMD ["--help"]
