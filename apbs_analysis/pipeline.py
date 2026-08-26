"""Resumable batch runner shared by the cluster and local entry points.

Same operational contract as the Voronoi workers: models are processed one at
a time into a per-target HDF5, already-complete models are skipped, SIGTERM
stops the loop cleanly so a requeued job resumes where it left off, and every
model (success or failure) is recorded in a summary CSV.
"""

from __future__ import annotations

import concurrent.futures as futures_module
import csv
from dataclasses import dataclass
from pathlib import Path
import signal
import time
import traceback
from typing import Iterator, Sequence

import h5py

from apbs_analysis.common import ModelInput, model_group_is_complete
from apbs_analysis.electrostatics import ApbsSettings, compute_surface_electrostatics
from apbs_analysis.storage import commit_model_group, discard_staging_groups


SUMMARY_FIELDS = (
    "target_name",
    "model_id",
    "pdb_path",
    "location_type",
    "status",
    "message",
    "elapsed_seconds",
    "num_atoms",
    "num_residues",
    "num_surface_points",
    "total_charge",
    "total_sasa",
    "mean_surface_potential",
)

_stop_requested = False


def _request_stop(signum, frame) -> None:  # noqa: ARG001
    global _stop_requested
    _stop_requested = True


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)


@dataclass
class RunOptions:
    output_hdf5_path: Path
    summary_csv_path: Path
    settings: ApbsSettings
    store_surface_points: bool = False
    store_grid: bool = False
    intermediates_dir: Path | None = None
    scratch_dir: Path | None = None
    overwrite: bool = False
    log_every: int = 25
    timeout: float | None = None
    workers: int = 1  # >1 solves models in parallel processes; the parent still does every HDF5 write


def _new_row(model: ModelInput) -> dict[str, object]:
    return {
        "target_name": model.target_name,
        "model_id": model.model_id,
        "pdb_path": str(model.pdb_path),
        "location_type": model.location_type,
        "status": "success",
        "message": "",
        "elapsed_seconds": 0.0,
        "num_atoms": 0,
        "num_residues": 0,
        "num_surface_points": 0,
        "total_charge": "",
        "total_sasa": "",
        "mean_surface_potential": "",
    }


def compute_one(model: ModelInput, options: RunOptions):
    """Solve one model. Runs in a worker process when workers > 1, so it must
    stay free of any open HDF5 handle -- the parent owns all writes."""
    row = _new_row(model)
    start_time = time.time()
    result = None
    try:
        result = compute_surface_electrostatics(
            pdb_path=model.pdb_path,
            model_id=model.model_id,
            settings=options.settings,
            scratch_dir=options.scratch_dir,
            keep_intermediates_dir=options.intermediates_dir,
            keep_grid=options.store_grid,
            keep_surface_points=options.store_surface_points,
            timeout=options.timeout,
        )
        row["num_atoms"] = int(len(result.structure))
        row["num_residues"] = int(len(result.residue_number))
        row["num_surface_points"] = (
            int(len(result.surface_xyz)) if result.surface_xyz is not None else ""
        )
        row["total_charge"] = round(float(result.structure.charge.sum()), 4)
        row["total_sasa"] = round(float(result.residue_sasa.sum()), 2)
        surfaced = result.residue_potential_mean[result.residue_surface_point_count > 0]
        row["mean_surface_potential"] = round(float(surfaced.mean()), 5) if surfaced.size else ""
        if result.warnings:
            row["message"] = "; ".join(result.warnings)
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["message"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:800]
        traceback.print_exc()
    row["elapsed_seconds"] = round(time.time() - start_time, 3)
    return row, result


def _select_pending(
    handle: h5py.File, models: Sequence[ModelInput], options: RunOptions
) -> tuple[list[ModelInput], list[dict[str, object]]]:
    """Split models into work to do and already-finished rows to record."""
    pending: list[ModelInput] = []
    skipped_rows: list[dict[str, object]] = []
    for model in models:
        existing = handle.get(model.model_id)
        if existing is not None and not options.overwrite and model_group_is_complete(
            existing,
            want_surface_points=options.store_surface_points,
            want_grid=options.store_grid,
        ):
            row = _new_row(model)
            row["status"] = "skipped_exists"
            row["message"] = "Complete group already present."
            skipped_rows.append(row)
        else:
            pending.append(model)
    return pending, skipped_rows


def _iter_results(pending: Sequence[ModelInput], options: RunOptions) -> Iterator[tuple]:
    """Yield (row, result) per model, sequentially or from a bounded pool.

    The pool keeps only workers + 1 models in flight: a stored potential grid
    is tens of MiB, so an unbounded queue of finished results would be the
    largest memory consumer in the run.
    """
    if options.workers <= 1:
        for model in pending:
            if _stop_requested:
                return
            yield compute_one(model, options)
        return

    queue = iter(pending)
    with futures_module.ProcessPoolExecutor(max_workers=options.workers) as pool:
        submitted: dict = {}
        broken_rows: list[dict[str, object]] = []

        def submit_next() -> bool:
            model = next(queue, None)
            if model is None:
                return False
            try:
                submitted[pool.submit(compute_one, model, options)] = model
            except Exception as exc:  # noqa: BLE001
                # The pool is broken (a worker died hard). Drain the queue into
                # error rows rather than aborting the whole target.
                for remaining in [model, *queue]:
                    row = _new_row(remaining)
                    row["status"] = "error"
                    row["message"] = f"worker pool unusable: {type(exc).__name__}: {exc}"[:800]
                    broken_rows.append(row)
                return False
            return True

        for _ in range(options.workers + 1):
            if not submit_next():
                break

        while submitted:
            done, _pending = futures_module.wait(
                set(submitted), return_when=futures_module.FIRST_COMPLETED
            )
            for future in done:
                model = submitted.pop(future)
                try:
                    yield future.result()
                except Exception as exc:  # noqa: BLE001
                    # A worker that died outright (OOM, segfault in APBS) must
                    # not take the rest of the target down with it.
                    row = _new_row(model)
                    row["status"] = "error"
                    row["message"] = f"worker process failed: {type(exc).__name__}: {exc}"[:800]
                    yield row, None
            if _stop_requested:
                for future in submitted:
                    future.cancel()
                return
            for _ in range(len(done)):
                if not submit_next():
                    break

    for row in broken_rows:
        yield row, None


def run_models(
    models: Sequence[ModelInput], options: RunOptions, run_attributes: dict[str, object] | None = None
) -> list[dict[str, object]]:
    """Process every model into options.output_hdf5_path; return summary rows."""
    options.output_hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    options.summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(options.output_hdf5_path, "a") as handle:
        discarded = discard_staging_groups(handle)
        if discarded:
            print(f"  cleared {discarded} staging group(s) left by an interrupted run")
        for key, value in (run_attributes or {}).items():
            handle.attrs[key] = value
        for key, value in options.settings.as_attributes().items():
            handle.attrs[key] = value
        handle.attrs["stores_surface_points"] = bool(options.store_surface_points)
        handle.attrs["stores_potential_grid"] = bool(options.store_grid)

        pending, summary_rows = _select_pending(handle, models, options)
        if summary_rows:
            print(f"  {len(summary_rows)} model(s) already complete; {len(pending)} to compute")

        completed = 0
        for row, result in _iter_results(pending, options):
            if result is not None:
                try:
                    commit_model_group(handle, str(row["model_id"]), result)
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "error"
                    row["message"] = f"write failed: {type(exc).__name__}: {exc}"[:800]
            summary_rows.append(row)
            completed += 1
            if options.log_every and completed % options.log_every == 0:
                print(f"  processed {completed}/{len(pending)} models", flush=True)

        if completed < len(pending):
            print(
                f"  stopped after {completed}/{len(pending)} models. Rerun the same command to resume."
            )

    write_summary(options.summary_csv_path, summary_rows)
    return summary_rows


def write_summary(summary_csv_path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Merge with any previous summary so resumed runs keep earlier rows."""
    existing: dict[str, dict[str, object]] = {}
    if summary_csv_path.is_file():
        with summary_csv_path.open("r", newline="", encoding="utf-8") as handle:
            for previous in csv.DictReader(handle):
                existing[previous["model_id"]] = previous
    for row in rows:
        model_id = str(row["model_id"])
        previous = existing.get(model_id)
        # A "skipped_exists" row carries no statistics, so never let it clobber
        # the real row a previous run recorded for the same model.
        if row["status"] == "skipped_exists" and previous is not None and previous.get("status") != "skipped_exists":
            continue
        existing[model_id] = row

    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing.values())


def report(target_name: str, rows: Sequence[dict[str, object]], summary_csv_path: Path) -> int:
    """Print a one-line outcome; return the number of failed models."""
    successes = sum(1 for row in rows if row["status"] == "success")
    errors = sum(1 for row in rows if row["status"] == "error")
    skipped = sum(1 for row in rows if row["status"] == "skipped_exists")
    print(
        f"Finished {target_name}: {successes} succeeded, {errors} failed, "
        f"{skipped} skipped. Summary -> {summary_csv_path}"
    )
    return errors
