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
# Miniconda's py311_env supplies the Python 3.11 interpreter used to build the
# project venv. The venv remains the actual project environment.
#
# The conda env is just a source of a python3.11 binary; the venv this
# script creates is a standard, self-contained venv with no conda
# dependency at runtime.
set -euo pipefail

PROJECT_DIR="/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder"
VENV_DIR="$PROJECT_DIR/venv"

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate py311_env

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
if [[ "$PY_VERSION" != "3.10" && "$PY_VERSION" != "3.11" ]]; then
  echo "python3 resolves to Python $PY_VERSION -- pyvoro-mmalahe has no prebuilt wheel" >&2
  echo "for this version (only 3.8-3.11 are supported). Activate a Python 3.10" >&2
  echo "or recreate py311_env with Python 3.11, then rerun this script." >&2
  exit 1
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$PROJECT_DIR/requirements.txt"

echo "Environment ready at $VENV_DIR (Python $PY_VERSION)"
