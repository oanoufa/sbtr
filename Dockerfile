# syntax=docker/dockerfile:1

# Build MAFFT from source
FROM python:3.11-slim AS mafft-builder
LABEL maintainer="olivier.anoufa@pasteur.fr"

ENV MAFFT_VERSION=7.526

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget gcc make libc6-dev \
    && cd /tmp \
    && wget -O mafft-src.tgz https://gitlab.com/sysimm/mafft/-/archive/v${MAFFT_VERSION}/mafft-v${MAFFT_VERSION}.tar.gz \
    && tar -xzf mafft-src.tgz \
    && cd mafft-v${MAFFT_VERSION}/core \
    && make \
    && make install \
    && rm -rf /tmp/* \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Runtime Image
FROM python:3.11-slim AS runtime

ARG TORCH_VARIANT=cpu

# Installed awk and grep/sed to support MAFFT scripts in slim environments
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 gawk \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy MAFFT binaries and binary dependencies from builder
COPY --from=mafft-builder /usr/local/bin/ /usr/local/bin/
COPY --from=mafft-builder /usr/local/libexec/mafft/ /usr/local/libexec/mafft/

WORKDIR /app

# Download repository
RUN curl -fsSL https://github.com/oanoufa/sbtr/archive/refs/heads/main.tar.gz | \
    tar -xzf - -C /app --strip-components=1

# Install Python requirements safely
RUN pip install --no-cache-dir --upgrade pip \
    && if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; fi \
    && if [ -f "requirements_${TORCH_VARIANT}.txt" ]; then pip install --no-cache-dir -r "requirements_${TORCH_VARIANT}.txt"; fi \
    && rm -f requirements*.txt

# Setup writable temporary directories in standard /tmp
ENV TMPDIR=/tmp \
    TEMP=/tmp \
    TMP=/tmp \
    MAFFT_TMPDIR=/tmp/mafft \
    MPLCONFIGDIR=/tmp/mpl_config \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface

RUN mkdir -p /tmp/mpl_config /tmp/huggingface /tmp/mafft /tmp/cache \
    && chmod -R 777 /tmp

ENTRYPOINT ["python", "python_scripts/sbtr.py"]
CMD ["--help"]