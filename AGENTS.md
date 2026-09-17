# Cluster resource allocation

Persistent user requirement: do not waste cluster resources. Verify actual
execution parallelism before setting Slurm CPU counts; do not infer CPU usage
from the requested allocation or a script's name.

- Serial jobs request one CPU per task.
- Jobs requesting multiple CPUs must explicitly configure and use matching
  threads or worker processes. Account for the main process and prevent nested
  BLAS/OpenMP/PyTorch thread pools from oversubscribing the allocation.
- Distinguish parallelism across array jobs from parallelism within one job.
  An array of serial jobs still needs only one CPU per task.
- Check each stage separately: data preparation, validation, training, and
  comparison. Log the effective worker/thread configuration and verify that it
  fits the actual Slurm allocation.
- Reuse valid completed preprocessing and results when correcting resource
  requests; do not rerun expensive work solely because it requested extra CPUs.
- Do not add Slurm array concurrency caps unless the user asks for them.


# Local prototyping and data reuse

- Prefer local preflight and training on the existing 10% subset before cluster
  dispatch. Audit all local graph files separately; do not expand training to
  the full dataset merely because it is present.
- Existing inputs are under /scratch/ppi_autoencoder_code/processed_graph_data,
  /scratch/ppi_autoencoder_code/rsasa_i_data, and
  /scratch/uniformly_sampled_ppi_data. Inspect these before downloading data.
- Local graph copies may predate cluster features. Verify versions/coverage and
  content hashes; reuse identical PDB/CSV/embedding files without downloading
  them again. Never overwrite existing original datasets during local setup.
- Workstation Globus collection: 60ab259b-1d82-11f1-bd71-0e34a6ec9899.
  Bouchet collection: a2bf0df9-5633-4565-b083-b8907423bb77.
- Local transfer destination: /scratch/ppi_extension_data/full. Verify endpoint
  path permissions and actual CUDA access; do not claim GPU validation from
  CPU-only tests.
