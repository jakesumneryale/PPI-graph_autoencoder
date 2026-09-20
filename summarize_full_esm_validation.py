"""Paired ESM/control validation metrics; a negative MSE delta favors ESM."""
import argparse
import json
from pathlib import Path
import pandas as pd
from summarize_full_pooling_validation import summarize


def compare(parent,root):
    controls=summarize(parent);esm=summarize(root)
    matrix=json.loads((root/'matrix.json').read_text())
    completed=set(esm.run);rows=[]
    for r in matrix['runs']:
        name=Path(r['output']).name;control=Path(r['matched_control']).name
        if name not in completed or control not in set(controls.run):continue
        a=pd.read_csv(root/name/'validation_predictions.csv').sort_values(['target_id','graph_name']).reset_index(drop=True)
        b=pd.read_csv(parent/control/'validation_predictions.csv').sort_values(['target_id','graph_name']).reset_index(drop=True)
        pd.testing.assert_frame_equal(a[['target_id','graph_name','true_target']],b[['target_id','graph_name','true_target']])
        x=esm.set_index('run').loc[name];y=controls.set_index('run').loc[control]
        row=dict(pooling=x.pooling,seed=int(x.seed),esm_run=name,control_run=control)
        for metric in ('mse','macro_mse','macro_rho'):
            row[metric+'_esm']=x[metric];row[metric+'_control']=y[metric];row[metric+'_delta']=x[metric]-y[metric]
        rows.append(row)
    if not rows:raise ValueError('No completed matched pairs')
    result=pd.DataFrame(rows);result.to_csv(root/'analysis/paired_esm_deltas.csv',index=False)
    print(result.to_string(index=False))
    print('MSE deltas: negative favors ESM. Rho delta: positive favors ESM. Completed pairs:',len(rows),'of 6.')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--esm',type=Path,required=True)
    a=p.parse_args();compare(a.parent,a.esm)
