README

This is the graph autoencoder model for PPIs. 

The plan will be to create a Graph Attention Autoencoder model that will learn interface features of PPIs and then use those features in a scoring function. 

I will update this plan as it matures. 

Quick start

Install dependencies in your Python environment. On the Ubuntu 24.04 / RTX 4090
machine, install the CUDA-enabled PyTorch build first from pytorch.org, then run:

pip install -r requirements.txt

Inspect the HDF5 graph data:

python protein_hdf5_dataset.py --require-target

Run a one-batch forward/backward sanity check:

python smoke_test_gate.py --device auto

Train the GATE model:

python train_gate.py --epochs 10 --batch-size 4 --device auto

Training now splits data by target HDF5 file to avoid leakage across train,
validation, and test. By default it uses an 80/20 train/test split at the
target level, then splits the remaining training targets 90/10 into
train/validation. After every epoch it reports train, validation, and test
losses, writes them to checkpoints/loss_history.csv, and writes the exact
target split to checkpoints/target_splits.json.

You can compare feature variants while keeping the exact same target split by
passing a shared --split-manifest path across runs. The training script also
writes best-checkpoint test-set DockQ predictions to test_predictions.csv for
later correlation analysis. Use --node-feature-set all for the full 22 node
features, or --node-feature-set no-aa-identity to drop the 20-dimensional
amino-acid identity features. You can also append the per-decoy average
rSASA/drSASA scalars to every node with:

python train_gate.py --use-rsasa-i --use-drsasa

or either flag by itself. The optional CSVs are loaded from
/scratch/ppi_autoencoder_code/rsasa_i_data by default, and you can override
that with --optional-node-features-dir.

If you are running on the cluster, add --cluster and run the code from:

/home/jas485/project_pi_co54/jas485/ppi_autoencoder_project

That flag switches the default graph-data directory to:

/home/jas485/project_pi_co54/jas485/ppi_processed_graphs

and the default optional rSASA/drSASA directory to:

/home/jas485/project_pi_co54/jas485/rsasa_i_graph_data

For the NVIDIA L40S nodes, a good starting point is one GPU with
--num-workers 8 so the dataloader uses 8 CPU cores on the 32-core / 4-GPU
node layout. Example:

python train_gate.py --cluster --device auto --num-workers 8 --epochs 20

If a cluster run appears to hang before the GPU gets work, try the new safer
DataLoader settings first:

python train_gate.py --cluster --device auto --num-workers 8 --worker-start-method spawn

The training script now prints progress while indexing the dataset, building
the dataloaders, and starting each epoch so it is easier to see where startup
time is going. If needed, fall back to --num-workers 0 to confirm the issue is
worker startup rather than model code.

Evaluate how well saved checkpoints reconstruct held-out test graphs:

python evaluate_gate_reconstruction.py

This writes summary and per-target CSV files under
checkpoints/reconstruction_eval/ and reports interpretable metrics such as
amino-acid identity misclassification rate, binary feature accuracy, sampled
edge recovery/precision, and C-alpha distance error in angstroms. When
training finishes, train_gate.py now also runs this reconstruction evaluation
automatically on the saved test split and writes reconstruction_summary.csv
and reconstruction_by_target.csv inside the training output directory.

The scripts first use PPI_HDF5_DATA when it is set, then
/scratch/ppi_autoencoder_code/processed_graph_data when that directory exists.
If neither is available, they fall back to data_for_testing and then
data_for_training. You can always pass --data /path/to/file_or_directory to
override this.

Unreadable or truncated HDF5 files are skipped with a warning by default. Pass
--strict-hdf5 if you want the scripts to fail immediately on a bad file.

The richer HDF5 files are expected to contain node_features/aa_type,
node_features/chain, node_features/interface_nodes, edge_features/contacts,
edge_features/interface_edges, edge_features/ca_dist, and target_scores/DockQ.

Voronoi contact-area edge features

You can now precompute residue-residue Voronoi contact areas from the existing
decoy PDB files and align them back to the graph node IDs (`aa_id`) used in
the HDF5 graph files.

First, generate the per-target model-reference lists from the graph HDF5 keys:

python precompute_voronoi_model_references.py --cluster

This writes text and CSV reference files under:

voronoi_edge_features_data/model_references

Then run the standalone wrapper for a specific target directory:

python generate_voronoi_contact_area_data.py \
  --cluster \
  /full/path/to/<target_name>

That writes:

- a target-level HDF5 file with one group per model under
  voronoi_edge_features_data/contact_area_hdf5
- a per-model summary CSV alongside it

Each model group stores the node metadata (`aa_id`, residue index, chain, amino
acid identity), all residue-residue Voronoi contact pairs, their surface
areas, and the same areas aligned to the graph's existing
edge_features/contacts ordering.

If you want to write the aligned area directly into the graph HDF5 as a new
edge feature, add:

python generate_voronoi_contact_area_data.py \
  --cluster \
  --write-graph-feature \
  --feature-name voronoi_contact_area \
  /full/path/to/<target_name>
