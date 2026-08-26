# Sourced by the cluster scripts to put the APBS environment on PATH.
#
# The environment is created with `conda create --prefix`, which produces a
# *nameless* env: `conda activate apbs_env` only resolves names against the
# directories in `envs_dirs`, so the short name works only once setup_env.sh's
# ~/.condarc entry is in place. That entry lives in the user's home directory,
# outside this repo, so the full prefix is kept as a fallback -- an array task
# should not fail because a personal config file was reset.
#
# Expects ENV_PREFIX to be set by the caller.
module load miniconda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda activate "${APBS_ENV_NAME:-apbs_env}" 2>/dev/null; then
  echo "Activated ${APBS_ENV_NAME:-apbs_env} by name ($CONDA_PREFIX)"
elif conda activate "$ENV_PREFIX" 2>/dev/null; then
  echo "Name did not resolve; activated by prefix ($ENV_PREFIX)."
  echo "Run apbs_analysis/cluster/setup_env.sh to register the short name."
else
  echo "Could not activate the APBS environment (tried name '${APBS_ENV_NAME:-apbs_env}' and prefix '$ENV_PREFIX')." >&2
  echo "Run apbs_analysis/cluster/setup_env.sh first." >&2
  exit 1
fi
