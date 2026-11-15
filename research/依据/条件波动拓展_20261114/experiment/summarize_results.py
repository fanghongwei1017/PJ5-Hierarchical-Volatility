"""Aggregate only already computed and validated extension results."""
from pathlib import Path
import hashlib
import json
import pandas as pd

OUT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text('utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    evaluations=sorted(OUT.glob('*.evaluation.json'));assert len(evaluations)==8
    rows=[];checks=[];sources={};failed_starts=[]
    for path in evaluations:
        e=load(path);vpath=OUT/f"{e['symbol']}.validation.json";v=load(vpath)
        assert e['status']==v['status']=='passed' and v['evaluation_sha256']==sha(path)
        assert not v['new_parameter_boundaries'] and v['all_new_optimizer_success']
        sources[path.name]=sha(path);sources[vpath.name]=sha(vpath);checks.extend(v['density_checks'])
        for r in e['paired_intervals']:
            rows.append({'symbol':e['symbol'],'block':r['block'],'a':r['a'],'b':r['b'],'returns':r['n'],'days':r['days'],
                         'mean_difference':r['mean'],'ci5_lower':r['ci95_by_block_days']['5'][0],'ci5_upper':r['ci95_by_block_days']['5'][1],
                         'passes_sample_gate':e['counts'][r['block']]['meets_original_gate']})
    for p in sorted((OUT/'training').glob('*.json')):
        t=load(p)
        for i,s in enumerate(t['parameter']['starts']):
            if not s['success']:failed_starts.append({'training_case':t['name'],'start_index':i,**s})
    pd.DataFrame(rows).to_csv(OUT/'paired_score_summary.csv',index=False,encoding='utf-8-sig')
    summary={'status':'passed','stocks':len(evaluations),'selected_h':load(OUT/'selection.json')['selected_h'],
             'new_selected_parameters_on_boundaries':0,'unused_failed_optimizer_starts':failed_starts,
             'max_probe_log_density_error':max(r['max_probe_log_density_error'] for r in checks),
             'max_point_log_density_error':max(r['max_point_log_density_error'] for r in checks),
             'max_mean_log_score_error':max(x for r in checks for x in r['mean_log_score_error_by_block'].values()),
             'source_sha256':sources,'summary_script_sha256':sha(Path(__file__)),
             'scope':'Own high-precision density/state validation; independent validation is reported separately.'}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_sha256','unused_failed_optimizer_starts']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
