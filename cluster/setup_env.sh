#!/usr/bin/env bash
# One-time setup of the (non-conda) Python virtualenv on the cluster.
#
# pyvoro's maintained fork (pyvoro-mmalahe, what requirements.txt now installs
# -- still imports as `import pyvoro`) only ships prebuilt wheels for CPython
# 3.8-3.11. There is no 3.12 or 3.13 wheel: pip would fall back to building
# from source on those, and that build is known-broken on 3.13 (an old
# Cython-generated C call uses a signature that changed in 3.13). So this
# venv must be created with a 3.10 or 3.11 interpreter, not whatever
# `python3` happens to resolve to on the login node.
#
# If the cluster doesn't already expose a bare python3.11 (check
# `module avail python` first), use conda ONLY to obtain that interpreter --
# not to manage the actual project environment -- then run this script from
# inside it:
#
#   conda create -y -n py311-bootstrap python=3.11
#   conda activate py311-bootstrap
#   bash cluster/setup_env.sh
#
# The conda env is just a source of a python3.11 binary; the venv this
# script creates is a standard, self-contained venv with no conda
# dependency at runtime.
set -euo pipefail

PROJECT_DIR="/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder"
VENV_DIR="$PROJECT_DIR/venv"

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
if [[ "$PY_VERSION" != "3.10" && "$PY_VERSION" != "3.11" ]]; then
  echo "python3 resolves to Python $PY_VERSION -- pyvoro-mmalahe has no prebuilt wheel" >&2
  echo "for this version (only 3.8-3.11 are supported). Activate a Python 3.10" >&2
  echo "or 3.11 interpreter first (see the comment at the top of this script for" >&2
  echo "the conda-bootstrap approach), then rerun this script." >&2
  exit 1
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$PROJECT_DIR/requirements.txt"

echo "Environment ready at $VENV_DIR (Python $PY_VERSION)"
