# syntax=docker/dockerfile:1

# build mafft from source
FROM python:3.11 AS mafft-builder
LABEL maintainer=olivier.anoufa@pasteur.fr

ENV MAFFT_VERSION=7.526
RUN apt-get update --fix-missing \
    && apt-get install -y --no-install-recommends wget gcc make \
    && cd /usr/local/ \
    && wget -O mafft-src.tgz https://gitlab.com/sysimm/mafft/-/archive/v${MAFFT_VERSION}/mafft-v${MAFFT_VERSION}.tar.gz \
    && tar -xzf mafft-src.tgz \
    && rm -f mafft-src.tgz \
    && cd mafft-v${MAFFT_VERSION}/core \
    && make \
    && make install \
    && rm -rf /usr/local/mafft-v${MAFFT_VERSION} \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# runtime
FROM python:3.11-slim AS runtime

# TORCH_VARIANT selects which requirements file (and torch wheel index) to install.

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# mafft binaries + support scripts from the builder stage
COPY --from=mafft-builder /usr/local/bin/mafft* /usr/local/bin/
COPY --from=mafft-builder /usr/local/libexec/mafft/ /usr/local/libexec/mafft/

# Download github repository
RUN mkdir -p /app && \
    curl -fsSL https://github.com/oanoufa/sbtr/archive/refs/heads/main.tar.gz | \
    tar -xzf - -C /app --strip-components=1

# pip install libraries
WORKDIR /app
COPY requirements.txt requirements_cpu.txt requirements_gpu.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements_${TORCH_VARIANT}.txt \
    && rm -f requirements.txt requirements_cpu.txt requirements_gpu.txt

# Persisted via a mounted volume at run time so repeat runs reuse cached
# model / reference-bank downloads instead of re-fetching from HF Hub.
ENV HF_HOME=/root/.cache/huggingface

ENTRYPOINT ["python", "python_scripts/sbtr.py"]
CMD ["--help"]