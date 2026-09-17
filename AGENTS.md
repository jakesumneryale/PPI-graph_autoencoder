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
