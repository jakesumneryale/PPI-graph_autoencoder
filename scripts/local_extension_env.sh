# Source from the repository: source scripts/local_extension_env.sh
# Optional: export LOCAL_ROOT=/scratch/ppi_extension_data before sourcing.
export LOCAL_ROOT="${LOCAL_ROOT:-/scratch/ppi_extension_data/full}"
export SUBSET_DIR="$LOCAL_ROOT/subset_hdf5"
export FEATURE_DATA="$SUBSET_DIR"
export PDB_ROOT="$LOCAL_ROOT/pdb"
export APBS_DIR="$LOCAL_ROOT/apbs_model_data"
export ESM_ROOT="$LOCAL_ROOT/SS_embeds"
export OPTIONAL_NODE_FEATURES_DIR="$LOCAL_ROOT/rsasa_i"
export EXTENSION_DATA="$LOCAL_ROOT/prepared"
export EXPERIMENT_DIR="$LOCAL_ROOT/local_runs"
# Single-process local preparation. Training can separately enable loader workers.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
