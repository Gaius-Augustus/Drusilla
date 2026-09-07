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

RUN cd /opt && \
        git clone https://github.com/Gaius-Augustus/Drusilla && \
        cd Drusilla && \
        pip install --no-cache-dir .

ENV DRUSILLA_CACHE_DIR=/data/drusilla_cache
RUN mkdir -p /data/drusilla_cache
