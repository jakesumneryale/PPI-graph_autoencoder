"""Summarize completed sweep checkpoints without accessing test predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dockq_objectives import prediction_report


def summarize(root):
    matrix=json.loads((root/'matrix.json').read_text()); rows=[]; missing=[]; bin_rows=[]
    reference=None
    for run in matrix['runs']:
        # Portable after copying results from cluster to workstation.
        path=root/Path(run['output']).name
        if not (path/'validation_predictions.csv').exists():
            missing.append(path.name); continue
        h=pd.read_csv(path/'loss_history.csv'); f=pd.read_csv(path/'validation_predictions.csv')
        f=f.sort_values(['target_id','graph_name']).reset_index(drop=True)
        if f.duplicated(['target_id','graph_name']).any(): raise ValueError(f'Duplicate prediction keys: {path}')
        identity=f[['target_id','graph_name','true_target']]
        if reference is None: reference=identity
        else: pd.testing.assert_frame_equal(reference,identity)
        best=h.loc[h.val_target_mse.idxmin()]
        report=prediction_report(f.true_target,f.predicted_target,f.target_id)
        if not np.isclose(report['pooled_mse'],best.val_target_mse,rtol=1e-4,atol=1e-7):
            raise ValueError(f'Prediction/checkpoint mismatch: {path}')
        rows.append(dict(run=path.name,seed=run['seed'],**run['dockq_sweep'],
            best_epoch=int(best.epoch),final_mse=h.iloc[-1].val_target_mse,
            **{k:v for k,v in report.items() if k!='bins'}))
        bin_rows.extend(dict(run=path.name,seed=run['seed'],**run['dockq_sweep'],**b) for b in report['bins'])
    out=root/'analysis'; out.mkdir(exist_ok=True)
    (out/'missing_runs.json').write_text(json.dumps(missing,indent=2)+'\n')
    if not rows: raise ValueError('No completed runs')
    table=pd.DataFrame(rows); table.to_csv(out/'per_run.csv',index=False)
    pd.DataFrame(bin_rows).to_csv(out/'per_bin.csv',index=False)
    summary=table.groupby(['reconstruction_lambda','range_weighting']).agg(
        seeds=('seed','nunique'),mean_mse=('pooled_mse','mean'),std_mse=('pooled_mse','std'),
        mean_target_mse=('macro_target_mse','mean'),mean_bin_mse=('macro_bin_mse','mean'))
    summary['complete']=summary.seeds==len({r['seed'] for r in matrix['runs']})
    summary.to_csv(out/'configuration_summary.csv')
    print(summary.sort_values('mean_mse').to_string())
    print(f'{len(rows)} completed, {len(missing)} missing. Compare complete seed sets; standard deviations are across seeds, not confidence intervals.')
    return table

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path)
    summarize(p.parse_args().root)
