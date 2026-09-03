#!/bin/bash
set -euo pipefail

cd /app
exec python python_scripts/sbtr.py "$@"