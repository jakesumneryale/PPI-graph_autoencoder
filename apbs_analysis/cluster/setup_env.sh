#!/usr/bin/env bash
# One-time setup of the APBS/pdb2pqr conda environment on the cluster.
#
# Unlike the Voronoi venv, this one must be conda rather than pip: APBS is a
# compiled Fortran/C++ solver with no PyPI wheel, and conda-forge is the only
# maintained binary distribution of it. pdb2pqr is pulled from the same channel
# so its Python and APBS's agree.
#
# The environment is created inside the project directory (not the user's home
# conda prefix) so the compute nodes read it over the same shared filesystem as
# the code, and so it can be deleted with the project.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/nfs/roberts/project/pi_co54/jas485/PPI-graph_autoencoder}"
ENV_PREFIX="${APBS_ENV_PREFIX:-$PROJECT_DIR/apbs_env}"

module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [[ -d "$ENV_PREFIX" ]]; then
  echo "Environment already exists at $ENV_PREFIX; delete it to rebuild."
else
  conda create -y --prefix "$ENV_PREFIX" -c conda-forge --override-channels \
    python=3.11 apbs pdb2pqr numpy scipy h5py pandas
fi

conda activate "$ENV_PREFIX"
echo
echo "apbs:     $(command -v apbs)"
echo "pdb2pqr:  $(command -v pdb2pqr)"
echo "python:   $(command -v python)"
echo
python "$PROJECT_DIR/apbs_analysis/cluster/check_apbs_environment.py"
echo "Environment ready at $ENV_PREFIX"
