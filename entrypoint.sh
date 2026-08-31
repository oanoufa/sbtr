#!/bin/bash
set -euo pipefail

# TODO: replace with the real org/repo
REPO="oanoufa/sbtr"
REPO_REF="${REPO_REF:-main}"           # override with `-e REPO_REF=v1.2.0` for a pinned/reproducible run
APP_DIR="/app"

echo "Fetching ${REPO}@${REPO_REF} ..."
rm -rf "${APP_DIR}"
mkdir -p "${APP_DIR}"
curl -fsSL "https://github.com/${REPO}/archive/refs/heads/${REPO_REF}.tar.gz" \
    -o /tmp/repo.tar.gz \
  || curl -fsSL "https://github.com/${REPO}/archive/refs/tags/${REPO_REF}.tar.gz" \
    -o /tmp/repo.tar.gz
tar -xzf /tmp/repo.tar.gz -C "${APP_DIR}" --strip-components=1
rm -f /tmp/repo.tar.gz

cd "${APP_DIR}"
exec python python_scripts/sbtr.py "$@"