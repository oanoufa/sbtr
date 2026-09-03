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
ARG CUSTOM_TMP=/sbtr_tmp

# Installed awk and grep/sed to support MAFFT scripts in slim environments
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 gawk \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy MAFFT binaries and binaries dependencies from builder
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

# Setup writeable temporary directories (Apptainer/Singularity compatible)
ENV TMP_DIR=${CUSTOM_TMP} \
    HOME=${CUSTOM_TMP} \
    TMPDIR=${CUSTOM_TMP} \
    TEMP=${CUSTOM_TMP} \
    TMP=${CUSTOM_TMP} \
    MAFFT_TMPDIR=${CUSTOM_TMP}/mafft \
    MPLCONFIGDIR=${CUSTOM_TMP}/mpl_config \
    HF_HOME=${CUSTOM_TMP}/huggingface \
    TRANSFORMERS_CACHE=${CUSTOM_TMP}/huggingface

RUN mkdir -p ${TMP_DIR}/mpl_config ${TMP_DIR}/huggingface ${TMP_DIR}/mafft ${TMP_DIR}/cache \
    && chmod -R 777 ${TMP_DIR}

ENTRYPOINT ["python", "python_scripts/sbtr.py"]
CMD ["--help"]