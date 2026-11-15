"""Independent provenance, block-boundary and result checks for the 8:1:1 refit."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import fit_runner as run

def main():
    protocol=run.protocol()
    split=run.load(run.DATA/'split_manifest.json')
    days=split['days']
    assert len(days)==731 and days==sorted(days)
    expected={}
    for block,seg in [('train',days[:584]),('valid',days[584:657]),('test',days[657:])]:
        expected.update({d:('purge' if d in seg[-5:] else block) for d in seg})
    assert sum(x=='train' for x in expected.values())==579
    assert sum(x=='valid' for x in expected.values())==68
    assert sum(x=='test' for x in expected.values())==69
    results=[];summary=[]
    for case in protocol['cases']:
        k=run.key(case)
        base=run.load(run.OUT/(k+'.base.json'));depth=run.load(run.OUT/(k+'.depth.json'))
        evaluation=run.load(run.OUT/(k+'.evaluation.json'));numerics=run.load(run.OUT/(k+'.validation.json'))
        assert base['status']==evaluation['status']==numerics['status']=='passed'
        assert base['input_sha256']['protocol']==run.sha(run.OUT/'protocol.json')
        assert depth['base_sha256']==run.sha(run.OUT/(k+'.base.json'))
        assert evaluation['input_sha256']['base']==run.sha(run.OUT/(k+'.base.json'))
        assert evaluation['input_sha256']['depth']==run.sha(run.OUT/(k+'.depth.json'))
        assert numerics['evaluation_sha256']==run.sha(run.OUT/(k+'.evaluation.json'))
        for refined in [numerics['final_refined_fit'],*numerics['interior_refined_neighbors']]:
            assert refined['optimizer_success'] and not refined['parameter_boundary'],(k,'high precision optimization')
        assert abs(numerics['final_refined_error_delta'])<2e-6
        assert abs(numerics['final_characteristic_error_delta'])<2e-6
        assert all(r['max_log_density_error']<2e-6 for r in numerics['observed_range_density_checks'])
        raw=pd.read_csv(run.DATA/'historical'/f"returns_{case['symbol']}_{case['frequency_minutes']}min.csv.gz",
                        dtype={'trading_date':str},float_precision='round_trip')
        assert all(expected[d]==b for d,b in zip(raw.trading_date,raw.block))
        parts,scale,seasonality,hashes=run.prepare(case,True)
        assert scale==base['scale']
        train_raw=raw.loc[raw.block.eq('train')].copy()
        if case['variant']=='raw_nonzero':train_raw=train_raw.loc[train_raw.raw_log_return.ne(0.)]
        independent_returns=train_raw.raw_log_return.to_numpy()
        if case['variant']=='deseasonalized':
            mapping=base['seasonality']['scales']['s_by_tod']
            tod=train_raw.tod.astype(str) if 'tod' in train_raw else pd.to_datetime(train_raw.bar_start).dt.strftime('%H:%M')
            independent_returns=independent_returns/tod.map(mapping).to_numpy()
            assert base['seasonality']['scales']['n_train_returns']==len(train_raw)
            assert base['seasonality']['scales']['train_day_last']==train_raw.trading_date.max()
        independent_scale=float(np.std(independent_returns,ddof=1))
        scale_error=abs(independent_scale-scale)
        assert scale_error<1e-15,(k,scale_error)
        for name,digest in hashes.items():
            assert digest==evaluation['input_sha256'][name]
        tr=parts['train'];M=base['M_star']
        v,close_indices=run.rolling_variance(tr['x'],M,tr['sessions'])
        independent=[]
        for ident in np.unique(tr['sessions']):
            x=tr['x'][tr['sessions']==ident]
            if len(x)>=M:
                independent.extend(np.var(np.lib.stride_tricks.sliding_window_view(x,M),axis=1,ddof=1))
        assert len(v)==len(independent)==base['n_windows_train']
        var_error=float(np.max(np.abs(v-np.asarray(independent))))
        assert var_error<1e-10,(k,var_error)
        for i in close_indices:
            assert tr['sessions'][i]==tr['sessions'][i-M+1]
        rows=depth['fits'];assert [r['N'] for r in rows]==run.N_GRID[:len(rows)]
        n=depth['last_N'];by={r['N']:r for r in rows}
        changes=[abs(by[a]['error']-by[b]['error']) for a,b in [(n//4,n//2),(n//2,n)]]
        gap=abs(rows[-1]['error']-depth['limit']['error'])/depth['limit']['error']
        plateau=all(d<1e-4 and d/by[b]['error']<.001 for d,b in zip(changes,[n//2,n])) and gap<.001
        assert plateau==depth['practical_plateau']
        assert plateau,(k,'ceiling without practical plateau')
        assert all(r['optimizer_success'] and not r['parameter_boundary'] for r in rows)
        point=pd.read_csv(run.OUT/(k+'.point_scores.csv.gz'),dtype={'trading_date':str},float_precision='round_trip')
        assert run.sha(run.OUT/(k+'.point_scores.csv.gz'))==evaluation['point_scores_sha256']
        errors=[];period_gates={}
        for block,p in parts.items():
            q=point.loc[point.block.eq(block)]
            assert list(q.trading_date)==list(p['dates'])
            ndays=int(q.trading_date.nunique());nreturns=len(q)
            gate=bool(ndays>=60 and nreturns>=5000)
            assert evaluation['counts'][block]['meets_observed_day_and_return_gate']==gate
            period_gates[block]={'n_days':ndays,'n_returns':nreturns,'meets_60_day_5000_return_gate':gate}
            normalization_error=float(np.max(np.abs(q.normalized_return.to_numpy()-p['x'])))
            assert normalization_error<1e-14
            for r in evaluation['scores']:
                if r['block']==block:
                    error=abs(q['log_score_'+r['model']].mean()-r['mean_log_score'])
                    errors.append(float(error));assert error<1e-12
        for ci in evaluation['paired_intervals']:
            block=ci['block']
            if block.startswith('2026_'):
                q=point.loc[point.block.eq('2026')]
                q=q.loc[q.trading_date.le('20260430') if block=='2026_Jan_Apr' else q.trading_date.ge('20260501')]
            else:q=point.loc[point.block.eq(block)]
            assert len(q)==ci['n']
            period_gates[block]={'n_days':int(q.trading_date.nunique()),'n_returns':len(q),
                'meets_60_day_5000_return_gate':bool(q.trading_date.nunique()>=60 and len(q)>=5000)}
            d=q['log_score_'+ci['a']]-q['log_score_'+ci['b']]
            assert abs(d.mean()-ci['mean'])<1e-12
            for bounds in ci['ci95_by_block_days'].values():
                assert np.isfinite(bounds).all() and bounds[0]<=bounds[1]
        results.append({'key':k,'status':'passed','scale':scale,'M':M,'windows':len(v),
            'training_scale_independent_error':scale_error,
            'rolling_variance_max_error':var_error,'max_score_summary_error':max(errors),
            'last_N':n,'best_N':depth['best_N_in_tested_grid'],'plateau':plateau,
            'doubling_absolute_changes':changes,'relative_gap_to_limit':gap,
            'counts':evaluation['counts'],'period_sample_gates':period_gates})
        summary.append({'key':k,'symbol':case['symbol'],'frequency_minutes':case['frequency_minutes'],'variant':case['variant'],
            'scale':scale,'M':M,'base_selected_N':base['selected']['N'],'base_selected_error':base['selected']['error'],
            'last_N':n,'best_N':depth['best_N_in_tested_grid'],'best_beta':depth['best']['beta'],
            'best_error':depth['best']['error'],'last_error':rows[-1]['error'],
            'limit_error':depth['limit']['error'],'tau':depth['limit']['tau'],'plateau':plateau})
    run.dump(run.OUT/'validation_summary.json',{'status':'passed','protocol_sha256':run.sha(run.OUT/'protocol.json'),
        'historical_split_nominal':[584,73,74],'historical_split_retained':[579,68,69],
        'cases':results,'validator_sha256':run.sha(__file__)})
    run.dump(run.OUT/'summary.json',summary)
    pd.DataFrame(summary).to_csv(run.OUT/'summary.csv',index=False,float_format='%.17g')
    print('PASSED',len(results),'refitted cases; dates, scales, windows, scores, plateaus, numerical checks')
if __name__=='__main__':main()

