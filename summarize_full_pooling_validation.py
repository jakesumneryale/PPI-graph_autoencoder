"""Compare all/combined pooling and weighting on the full frozen validation split."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def summarize(root):
    m=json.loads((root/'matrix.json').read_text());split=json.loads((root/'target_splits.json').read_text())['splits']['val']
    rows=[];bins=[];targets=[];missing=[];reference=None
    for r in m['runs']:
        p=root/Path(r['output']).name
        if not (p/'validation_predictions.csv').exists():missing.append(p.name);continue
        f=pd.read_csv(p/'validation_predictions.csv').sort_values(['target_id','graph_name']).reset_index(drop=True)
        assert len(f)==split['sample_count'] and set(f.target_id)==set(split['targets'])
        assert not f.duplicated(['target_id','graph_name']).any()
        assert np.isfinite(f[['true_target','predicted_target']]).all().all()
        identity=f[['target_id','graph_name','true_target']]
        if reference is None:reference=identity
        else:pd.testing.assert_frame_equal(reference,identity)
        h=pd.read_csv(p/'loss_history.csv');expected=int(r['argv'][r['argv'].index('--epochs')+1])
        assert h.epoch.tolist()==list(range(1,expected+1))
        mse=np.mean((f.true_target-f.predicted_target)**2)
        assert np.isclose(mse,h.val_target_mse.min(),rtol=1e-4,atol=1e-7)
        base=dict(run=p.name,seed=r['seed'],pooling=r['full_validation']['pooling'],exponent=r['full_validation']['exponent'])
        rr=[]
        for target,g in f.groupby('target_id'):
            rr.append(dict(**base,target=target,n=len(g),mse=np.mean((g.true_target-g.predicted_target)**2),rho=spearmanr(g.true_target,g.predicted_target).statistic if g.true_target.nunique()>1 and g.predicted_target.nunique()>1 else np.nan))
        targets.extend(rr)
        rows.append(dict(**base,mse=mse,macro_mse=np.mean([v['mse'] for v in rr]),macro_rho=np.nanmean([v['rho'] for v in rr]),best_epoch=int(h.loc[h.val_target_mse.idxmin()].epoch)))
        index=np.searchsorted(np.array([.2,.4,.6,.8],dtype=np.float32),f.true_target.to_numpy(dtype=np.float32),side='right')
        for b in range(5):
            g=f.loc[index==b];e=g.predicted_target-g.true_target
            bins.append(dict(**base,bin=b,n=len(g),mse=np.mean(e**2),bias=e.mean()))
    out=root/'analysis';out.mkdir(exist_ok=True);(out/'missing_runs.json').write_text(json.dumps(missing,indent=2)+'\n')
    if not rows:raise ValueError('No completed results')
    table=pd.DataFrame(rows);table.to_csv(out/'per_run.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'per_bin.csv',index=False);pd.DataFrame(targets).to_csv(out/'per_target.csv',index=False)
    summary=table.groupby(['pooling','exponent']).agg(seeds=('seed','nunique'),mean_mse=('mse','mean'),seed_sd=('mse','std'),macro_mse=('macro_mse','mean'),macro_rho=('macro_rho','mean'))
    summary.to_csv(out/'summary.csv');print(summary.to_string());print(f'{len(missing)} missing runs; compare matching seed sets.')
    return table
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);summarize(p.parse_args().root)
