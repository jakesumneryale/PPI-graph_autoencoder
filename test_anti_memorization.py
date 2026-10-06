import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch, Data

from anti_memorization import (
    ModelEMA,
    TargetAdversary,
    TargetBalancedSampler,
    augment_batch,
    dann_coefficient,
    gradient_reversal,
)
from GATE_model import GraphAttentionAutoencoder
from test_model_extensions import write_fixture


def toy_graph(num_nodes=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    src = [a for a, b in pairs] + [b for a, b in pairs]
    dst = [b for a, b in pairs] + [a for a, b in pairs]
    edge_attr = torch.rand((len(src), 2), generator=g) * 5
    return Data(x=torch.rand((num_nodes, 5), generator=g), edge_index=torch.tensor([src, dst]),
                edge_attr=edge_attr, y=torch.tensor([0.5]),
                interface_mask=torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.bool))


def test_gradient_reversal_flips_and_scales_gradient():
    x = torch.ones(3, requires_grad=True)
    gradient_reversal(x, 0.25).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -0.25))


def test_dann_ramp_is_monotone_from_zero_to_one():
    values = [dann_coefficient(p) for p in np.linspace(0, 1, 11)]
    assert values[0] == 0 and values[-1] == pytest.approx(1, abs=1e-4)
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_augmentation_leaves_input_untouched_and_drops_contacts_symmetrically():
    batch = Batch.from_data_list([toy_graph(seed=1), toy_graph(seed=2)])
    original = batch.clone()
    torch.manual_seed(0)
    out = augment_batch(batch, edge_dropout=0.5, node_feature_mask=0.5, ca_dist_noise=1.0, ca_dist_column=1)
    for name in ("x", "edge_index", "edge_attr"):
        assert torch.equal(getattr(batch, name), getattr(original, name))
    kept = {tuple(e) for e in out.edge_index.t().tolist()}
    assert kept and len(kept) < batch.edge_index.size(1)
    assert all((b, a) in kept for a, b in kept)
    assert out.edge_attr.size(0) == out.edge_index.size(1)
    assert (out.edge_attr[:, 1] >= 0).all()
    masked_rows = (out.x == 0).all(dim=1)
    assert masked_rows.any() and not masked_rows.all()


def test_augmentation_noise_only_touches_ca_dist_column():
    batch = Batch.from_data_list([toy_graph()])
    out = augment_batch(batch, ca_dist_noise=0.5, ca_dist_column=1)
    assert torch.equal(out.edge_attr[:, 0], batch.edge_attr[:, 0])
    assert not torch.equal(out.edge_attr[:, 1], batch.edge_attr[:, 1])


def test_balanced_sampler_draws_per_target_and_redraws_each_epoch():
    groups = [list(range(0, 10)), list(range(10, 13)), list(range(13, 33))]
    sampler = TargetBalancedSampler(groups, per_target=4, seed=7)
    first, second = list(sampler), list(sampler)
    assert len(first) == len(sampler) == 4 + 3 + 4
    for epoch in (first, second):
        assert len(set(epoch)) == len(epoch)
        assert sum(i in groups[0] for i in epoch) == 4
        assert sum(i in groups[1] for i in epoch) == 3
    assert first != second


def test_ema_tracks_average_and_restores_live_weights():
    model = torch.nn.Linear(2, 1)
    ema = ModelEMA(model, decay=0.5)
    live = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    assert not torch.equal(ema.shadow["weight"], model.weight)
    with ema.applied(model):
        assert torch.equal(model.weight, ema.shadow["weight"])
    assert torch.equal(model.weight, live["weight"] + 1.0)


def test_detached_head_sends_no_dockq_gradient_to_encoder():
    batch = Batch.from_data_list([toy_graph(seed=3), toy_graph(seed=4)])
    model = GraphAttentionAutoencoder(5, 2, hidden_dim=8, latent_dim=4, gat_heads=2, pooling="combined")
    model.detach_quality_input = True
    model(batch)["quality_pred"].sum().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.gat1.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.quality_head.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.graph_projector.parameters())

    model.zero_grad(set_to_none=True)
    model.detach_quality_input = False
    model(batch)["quality_pred"].sum().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.gat1.parameters())


def test_adversary_pushes_encoder_against_its_own_objective():
    torch.manual_seed(0)
    z = torch.randn(8, 4, requires_grad=True)
    adversary = TargetAdversary(4, 8, num_targets=3)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    torch.nn.functional.cross_entropy(adversary(z, 1.0), labels).backward()
    reversed_grad = z.grad.clone()
    z.grad = None
    torch.nn.functional.cross_entropy(adversary.net(z), labels).backward()
    assert torch.allclose(reversed_grad, -z.grad)


def make_data(tmp_path, targets=5):
    data = tmp_path / "data"; data.mkdir()
    for i in range(targets):
        write_fixture(data / f"{i}abc.hdf5")
    return data


BASE = ["--epochs", "2", "--num-workers", "0", "--cpu-threads", "1", "--device", "cpu", "--batch-size", "2",
        "--no-test-evaluation", "--checkpoint-metric", "target_mse", "--hidden-dim", "8", "--latent-dim", "4",
        "--gat-heads", "2", "--loss-weight-mode", "fixed", "--dockq-range-diagnostics",
        "--test-fraction", "0.2", "--val-fraction", "0.25", "--residual-connections", "--pooling", "combined",
        "--edge-features", "interface_edges,ca_dist"]


@pytest.mark.parametrize("arm", [
    ["--dockq-head-mode", "detached"],
    ["--target-adversary-weight", "0.1"],
    ["--edge-dropout", "0.1", "--node-feature-mask", "0.1", "--ca-dist-noise", "0.25"],
    ["--decoys-per-target", "1", "--ema-decay", "0.9"],
    ["--target-adversary-weight", "0.1", "--edge-dropout", "0.1", "--node-feature-mask", "0.1",
     "--ca-dist-noise", "0.25", "--decoys-per-target", "1", "--ema-decay", "0.9"],
])
def test_training_arms_run_and_checkpoint_the_best_validation(tmp_path, arm):
    out = tmp_path / "run"
    cmd = [sys.executable, "train_gate.py", "--data", str(make_data(tmp_path)), "--output-dir", str(out),
           "--val-every-steps", "1", *BASE, *arm]
    result = subprocess.run(cmd, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr

    steps = pd.read_csv(out / "validation_steps.csv")
    history = pd.read_csv(out / "loss_history.csv")
    predictions = pd.read_csv(out / "validation_predictions.csv")
    assert len(history) == 2 and steps.end_of_epoch.sum() <= 2 and len(steps) >= 2
    assert steps.global_step.is_monotonic_increasing and steps.global_step.is_unique
    # The exported predictions come from the checkpoint, which is the best validation seen at any step.
    mse = np.mean((predictions.true_target - predictions.predicted_target) ** 2)
    assert mse == pytest.approx(steps.val_target_mse.min(), rel=1e-5, abs=1e-7)
    checkpoint = torch.load(out / "gate_model.pt", map_location="cpu", weights_only=False)
    assert checkpoint["best_step"] == steps.loc[steps.val_target_mse.idxmin(), "global_step"]
    if "--target-adversary-weight" in arm:
        assert {"train_target_adversary_ce", "train_target_adversary_acc"} <= set(history.columns)
        assert "target_adversary_state_dict" in checkpoint
    if "--ema-decay" in arm:
        assert checkpoint["ema_weights"] and steps.ema.all()


def test_control_without_new_options_keeps_history_columns(tmp_path):
    out = tmp_path / "run"
    cmd = [sys.executable, "train_gate.py", "--data", str(make_data(tmp_path, 3)), "--output-dir", str(out),
           *[a for a in BASE if a not in ("0.2", "0.25", "--test-fraction", "--val-fraction")]]
    result = subprocess.run(cmd, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    history = pd.read_csv(out / "loss_history.csv")
    assert not any("adversary" in c for c in history.columns)


@pytest.mark.parametrize("bad", [["--edge-dropout", "1.0"], ["--ema-decay", "1.5"],
                                 ["--target-adversary-weight", "-1"],
                                 ["--val-every-steps", "-1"]])
def test_invalid_options_are_rejected(tmp_path, bad):
    cmd = [sys.executable, "train_gate.py", "--data", str(tmp_path), *bad]
    if "--loss-weight-mode" not in bad:
        cmd += ["--loss-weight-mode", "fixed", "--checkpoint-metric", "target_mse"]
    result = subprocess.run(cmd, text=True, capture_output=True)
    assert result.returncode != 0 and "train_gate.py: error:" in result.stderr


def test_schedule_epochs_decouples_cosine_length_from_training_length():
    from types import SimpleNamespace
    from train_gate import build_lr_scheduler

    def lr_at(step, **kw):
        args = SimpleNamespace(lr_schedule="cosine", epochs=20, warmup_steps=0, lr_final_fraction=0.0, **kw)
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        sched = build_lr_scheduler(opt, args, steps_per_epoch=10)
        for _ in range(step):
            opt.step(); sched.step()
        return opt.param_groups[0]["lr"]

    assert lr_at(200) == pytest.approx(0.0, abs=1e-9)                      # 20-epoch schedule fully decayed
    assert lr_at(200, lr_schedule_epochs=50) == pytest.approx(0.5 * (1 + np.cos(np.pi * 0.4)))
    assert lr_at(100, lr_schedule_epochs=20) == pytest.approx(0.5)


def test_step_validation_runs_with_adaptive_loss_shares(tmp_path):
    out = tmp_path / "run"
    base = [a for a in BASE]
    base[base.index("fixed")] = "shares"
    cmd = [sys.executable, "train_gate.py", "--data", str(make_data(tmp_path)), "--output-dir", str(out),
           "--val-every-steps", "1", "--lr-schedule", "cosine", "--warmup-steps", "1", "--lr-schedule-epochs", "5",
           *base]
    result = subprocess.run(cmd, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    steps = pd.read_csv(out / "validation_steps.csv")
    predictions = pd.read_csv(out / "validation_predictions.csv")
    history = pd.read_csv(out / "loss_history.csv")
    mse = np.mean((predictions.true_target - predictions.predicted_target) ** 2)
    assert mse == pytest.approx(steps.val_target_mse.min(), rel=1e-5, abs=1e-7)
    assert history.weight_target_mse.nunique() > 1   # the balancer actually adapted the weights
    assert "training stops after" in result.stdout and " min)" in result.stdout


@pytest.mark.parametrize("bad", [["--edge-dropout", "0.1", "--loss-weight-mode", "shares"],
                                 ["--lr-schedule-epochs", "50"],
                                 ["--val-every-steps", "5", "--checkpoint-metric", "loss"]])
def test_intervention_and_schedule_guards(tmp_path, bad):
    cmd = [sys.executable, "train_gate.py", "--data", str(tmp_path), "--checkpoint-metric", "target_mse", *bad]
    result = subprocess.run(cmd, text=True, capture_output=True)
    assert result.returncode != 0 and "train_gate.py: error:" in result.stderr
