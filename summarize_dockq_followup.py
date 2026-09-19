"""Validation-only summary including reused controls and training diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dockq_objectives import prediction_report


def summarize(root):
    matrix=json.loads((root/'matrix.json').read_text()); rows=[]; bins=[]; missing=[]; reference=None
    controls={}
    entries=[]
    for c in matrix['reused_controls']:
        original=Path(c['output']); local=root.parent/original.parent.name/original.name
        path=original if original.exists() else local
        entries.append((path,dict(stage='pooling',pooling='all',exponent=.5,reconstruction_lambda=1),c['seed'],True))
    for r in matrix['runs']:
        entries.append((root/Path(r['output']).name,r['followup'],r['seed'],False))
    for path,cfg,seed,reused in entries:
        if not (path/'validation_predictions.csv').exists(): missing.append(str(path));continue
        f=pd.read_csv(path/'validation_predictions.csv').sort_values(['target_id','graph_name']).reset_index(drop=True)
        if f.duplicated(['target_id','graph_name']).any():raise ValueError(f'Duplicate graphs: {path}')
        identity=f[['target_id','graph_name','true_target']]
        if reference is None: reference=identity
        else:pd.testing.assert_frame_equal(reference,identity)
        report=prediction_report(f.true_target,f.predicted_target,f.target_id)
        base={k:cfg[k] for k in ('stage','pooling','exponent','reconstruction_lambda')}
        row=dict(**base,seed=seed,run=path.name,reused=reused,validation_mse=report['pooled_mse'],
                 macro_target_mse=report['macro_target_mse'],macro_bin_mse=report['macro_bin_mse'])
        if reused:controls[seed]=report['pooled_mse']
        if (path/'initial_validation.json').exists():
            row['initial_mse']=json.loads((path/'initial_validation.json').read_text())['pooled_mse']
        if (path/'loss_history.csv').exists():
            h=pd.read_csv(path/'loss_history.csv'); candidates=[h.val_target_mse.min()]
            if 'initial_mse' in row:candidates.append(row['initial_mse'])
            if not np.isclose(min(candidates),report['pooled_mse'],rtol=1e-4,atol=1e-7):raise ValueError(f'Checkpoint MSE mismatch: {path}')
            row['epochs']=len(h);row['final_mse']=h.iloc[-1].val_target_mse
        bins.extend(dict(**base,seed=seed,split='validation',**b) for b in report['bins'])
        if (path/'training_predictions.csv').exists():
            t=pd.read_csv(path/'training_predictions.csv')
            if set(t.target_id)&set(f.target_id):raise ValueError('Training/validation overlap')
            tr=prediction_report(t.true_target,t.predicted_target,t.target_id)
            row['training_eval_mse']=tr['pooled_mse']
            bins.extend(dict(**base,seed=seed,split='training',**b) for b in tr['bins'])
        rows.append(row)
    out=root/'analysis';out.mkdir(exist_ok=True)
    (out/'missing_runs.json').write_text(json.dumps(missing,indent=2)+'\n')
    if not rows:raise ValueError('No results found')
    frame=pd.DataFrame(rows);frame['delta_control']=frame.validation_mse-frame.seed.map(controls)
    frame.to_csv(out/'per_run.csv',index=False);pd.DataFrame(bins).to_csv(out/'per_bin.csv',index=False)
    summary=frame.groupby(['stage','pooling','exponent','reconstruction_lambda']).agg(
        seeds=('seed','nunique'),mean_mse=('validation_mse','mean'),seed_sd=('validation_mse','std'),
        macro_target_mse=('macro_target_mse','mean'),macro_bin_mse=('macro_bin_mse','mean'),delta_control=('delta_control','mean'))
    summary.to_csv(out/'configuration_summary.csv')
    print(summary.to_string());print(f'{len(missing)} missing entries. Compare complete matched seeds; test predictions are never read.')
    return frame

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    summarize(p.parse_args().root)
