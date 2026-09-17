"""Check CUDA gradients, interface pooling, EGNN symmetry and tiny-batch fitting.

Consumes only outputs of smoke_test_model_extensions.py. This is a numerical
software diagnostic, not training for an accuracy comparison.
"""
import argparse
import json
from pathlib import Path

import torch
from torch_geometric.data import Batch
from EGNN_model import build_graph_model
from protein_hdf5_dataset import ProteinGraphHDF5Dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('These checks require CUDA')
    results = {}
    for architecture in ('gat', 'egnn'):
        torch.manual_seed(7)
        checkpoint = torch.load(args.smoke / 'runs' / f'{architecture}_supervised_combined_seed7' /
                                'gate_model.pt', map_location='cpu', weights_only=False)
        a = checkpoint['args']
        dataset = ProteinGraphHDF5Dataset(args.smoke / 'software_only_data' / '1acb.hdf5',
            node_features=checkpoint['node_features'], edge_features=checkpoint['edge_features'],
            optional_node_features_dir=a['optional_node_features_dir'], use_esm=True, require_pos=True,
            edge_feature_transforms=checkpoint['edge_feature_transforms'],
            edge_feature_stats=checkpoint['edge_feature_stats'])
        graphs = sorted(list(dataset), key=lambda g: float(g.y))
        batch = Batch.from_data_list([graphs[0], graphs[-1]]).to('cuda')
        model = build_graph_model(architecture, pooling='interface', esm_dim=checkpoint['esm_dim'],
            esm_projection_dim=a['esm_projection_dim'], in_node_feats=checkpoint['in_node_feats'],
            in_edge_feats=checkpoint['in_edge_feats'], hidden_dim=a['hidden_dim'],
            latent_dim=a['latent_dim'], gat_heads=a['gat_heads'], dropout=0,
            residual_connections=True, out_edge_feats=len(checkpoint['edge_recon_columns'])).cuda()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        result = {'nodes': batch.num_nodes, 'directed_edges': batch.num_edges}
        with torch.no_grad():
            z = torch.randn(batch.num_nodes, a['latent_dim'], device='cuda')
            expected = torch.stack([torch.cat((z[(batch.batch == i) & batch.interface_mask].mean(0),
                z[(batch.batch == i) & batch.interface_mask].max(0).values)) for i in range(2)])
            pooled = model.pool_nodes(z, batch.batch, batch)
            torch.testing.assert_close(pooled, expected)
            z[~batch.interface_mask] += 10000
            torch.testing.assert_close(model.pool_nodes(z, batch.batch, batch), pooled)
            result['interface_only_pooling'] = True
            if architecture == 'egnn':
                rotation, _ = torch.linalg.qr(torch.randn(3, 3, device='cuda'))
                translation = torch.tensor([10., -5., 3.], device='cuda')
                moved = batch.clone()
                moved.pos = batch.pos @ rotation + translation
                h, pos = model.encode_geometry(batch)
                moved_h, moved_pos = model.encode_geometry(moved)
                torch.testing.assert_close(h, moved_h, atol=2e-5, rtol=2e-4)
                torch.testing.assert_close(pos @ rotation + translation, moved_pos, atol=2e-5, rtol=2e-4)
                pred = model(batch)['quality_pred']
                moved_pred = model(moved)['quality_pred']
                torch.testing.assert_close(pred, moved_pred, atol=2e-5, rtol=2e-4)
                result['rigid_motion_prediction_max_error'] = float((pred-moved_pred).abs().max())
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(60):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(batch)['quality_pred'] - batch.y.view(-1)).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError('Nonfinite gradient')
            if step == 0:
                grad = float(model.esm_projector[1].weight.grad.abs().sum())
                if grad <= 0:
                    raise ValueError('No gradient to ESM projection')
                result['esm_projection_gradient_l1'] = grad
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            final = float((model(batch)['quality_pred'] - batch.y.view(-1)).square().mean())
        result.update(initial_target_mse=losses[0], final_target_mse=final,
                      peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                      steps=len(losses), finite_gradients=True)
        if final >= losses[0] * .5:
            raise ValueError(f'{architecture}: tiny-batch loss did not fall by 50%: {result}')
        results[architecture] = result
        print(architecture, result, flush=True)
    (args.smoke / 'directed_checks.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
