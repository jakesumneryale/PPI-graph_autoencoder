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

The scripts use data_for_testing when that directory exists, otherwise they use
data_for_training. You can always pass --data /path/to/file_or_directory to
override this.

The richer HDF5 files are expected to contain node_features/aa_type,
node_features/chain, node_features/interface_nodes, edge_features/contacts,
edge_features/interface_edges, edge_features/ca_dist, and target_scores/DockQ.
