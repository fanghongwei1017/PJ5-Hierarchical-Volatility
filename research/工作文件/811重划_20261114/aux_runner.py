"""Refit-sensitive auxiliary statistics for the 8:1:1 chronological split."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import fit_runner as fit
DATA = ROOT/'依据/811重划实验_20261114/data'
OUT = ROOT/'依据/811重划实验_20261114/aux'
SEED_RESELECT = 20261107
SEED_CLOCK = 20261110
N_RESELECT = 20
N_CLOCK = 2000
BLOCKS = (1,5,10,20)
ENDPOINTS = ('lunch_before_am_last','lunch_after_pm_first',
             'night_before_prev_pm_last','night_after_am_first')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.writing')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def source():
    fit.init()
    selected=pd.read_csv(DATA/'selected_8.csv')
    primary=selected.sort_values('eligible_rank').iloc[0].symbol
    p=DATA/'historical'/f'returns_{primary}_5min.csv.gz'
    df=pd.read_csv(p,dtype={'trading_date':str},float_precision='round_trip')
    df=fit.ensure_tod(df)
    if 'slot_index' not in df:
        # The time-of-day labels are fixed by the session grid. The first available
        # return in a session starts at 09:35/13:05, not at the session open.
        labels=sorted(df.tod.unique())
        df['slot_index']=df.tod.map({t:i for i,t in enumerate(labels)}).astype(int)
    assert np.isfinite(df.raw_log_return).all()
    assert not df.duplicated(['trading_date','session_id','bar_start']).any()
    tr=df.loc[df.block.eq('train')].copy()
    scale=float(tr.raw_log_return.std(ddof=1))
    assert len(tr)>0 and scale>0
    session=pd.factorize(tr.trading_date+'|'+tr.session_id.astype(str))[0]
    evidence={'source_file':p.relative_to(ROOT).as_posix(),'source_sha256':sha(p),
              'split_sha256':sha(DATA/'split_manifest.json'),'selected_sha256':sha(DATA/'selected_8.csv'),
              'runtime_sha256':sha(fit.RUNTIME/'manifest.json'),'script_sha256':sha(__file__)}
    return primary,df,tr,scale,session,evidence


def seasonality(df,tr,evidence):
    scales,diag=fit.fit_seasonality(tr,5)
    processed=fit.apply_seasonality(df,scales)
    training=processed.loc[processed.block.eq('train')]
    g=training.groupby(['slot_index','tod','session_id']).agg(
        n=('raw_log_return','size'),
        mean_abs_raw=('raw_log_return',lambda x:float(np.mean(np.abs(x)))),
        mean_abs_deseasonalized=('deseasonalized_log_return',lambda x:float(np.mean(np.abs(x))))
    ).reset_index().sort_values(['slot_index','tod','session_id'])
    raw_ratio=float(g.mean_abs_raw.max()/g.mean_abs_raw.min())
    deseas_ratio=float(g.mean_abs_deseasonalized.max()/g.mean_abs_deseasonalized.min())
    assert len(g)==46
    assert np.max(np.abs(processed.raw_log_return.to_numpy()/processed.seasonality_scale.to_numpy()
                         -processed.deseasonalized_log_return.to_numpy()))<1e-15
    OUT.mkdir(parents=True,exist_ok=True)
    g.to_csv(OUT/'intraday_mean_absolute_returns.csv',index=False)
    result={'status':'passed','definition':'max/min across time-of-day groups of mean(abs(return)); not MAD-scale amplitude',
        'train_days':int(tr.trading_date.nunique()),'train_returns':len(tr),
        'training_start':tr.trading_date.min(),'training_end':tr.trading_date.max(),
        'mean_absolute_amplitude_before':raw_ratio,'mean_absolute_amplitude_after':deseas_ratio,
        'mad_scale_amplitude_separate':diag['amplitude'],
        'scale_by_tod':scales,'diagnostics':diag,'inputs':evidence}
    dump(OUT/'seasonality_check.json',result)
    print('SEASONALITY',raw_ratio,deseas_ratio,flush=True)
    return processed,result


def correlation_rows(x,y,eligible):
    counts=eligible.sum(axis=1);denom=np.maximum(counts,1)
    xm=np.where(eligible,x,0).sum(axis=1)/denom
    ym=np.where(eligible,y,0).sum(axis=1)/denom
    dx=np.where(eligible,x-xm[:,None],0.)
    dy=np.where(eligible,y-ym[:,None],0.)
    xx=(dx*dx).sum(axis=1);yy=(dy*dy).sum(axis=1)
    zero=(xx<=0)|(yy<=0);valid=(counts>=2)&~zero
    corr=np.full(len(counts),np.nan)
    corr[valid]=(dx*dy).sum(axis=1)[valid]/np.sqrt(xx[valid]*yy[valid])
    return corr,counts,zero


def point(values,eligible):
    mask=eligible[None,:]
    a,n,_=correlation_rows(values[:,0][None,:],values[:,1][None,:],mask)
    b,_,_=correlation_rows(values[:,2][None,:],values[:,3][None,:],mask)
    return {'n_common_dates':int(n[0]),'lunch_pearson':float(a[0]),'night_pearson':float(b[0]),
            'difference_night_minus_lunch':float(b[0]-a[0])}


def boot(values,indices,eligible):
    mask=eligible[indices]
    a,n,az=correlation_rows(values[:,0][indices],values[:,1][indices],mask)
    b,_,bz=correlation_rows(values[:,2][indices],values[:,3][indices],mask)
    delta=b-a;valid=np.isfinite(delta)
    assert valid.any()
    return {'n_boot_requested':N_CLOCK,'n_boot_valid_difference':int(valid.sum()),
        'n_boot_invalid_difference':int((~valid).sum()),'n_boot_insufficient_pairs':int((n<2).sum()),
        'n_boot_zero_variance_lunch':int(az.sum()),'n_boot_zero_variance_night':int(bz.sum()),
        'resampled_pair_count_min':int(n.min()),'resampled_pair_count_max':int(n.max()),
        'difference_ci95':np.quantile(delta[valid],[.025,.975]).tolist(),
        'difference_bootstrap_se':float(delta[valid].std(ddof=1))},delta


def clock_check(primary,processed,evidence):
    endpoints={}
    for day,d in processed.groupby('trading_date',sort=True):
        assert set(d.session_id)=={1,2}
        am=d.loc[d.session_id.eq(1)].sort_values('bar_start')
        pm=d.loc[d.session_id.eq(2)].sort_values('bar_start')
        assert len(am)==len(pm)==23,day
        assert d.block.nunique()==1
        endpoints[day]={'am_first':am.iloc[0],'am_last':am.iloc[-1],
                        'pm_first':pm.iloc[0],'pm_last':pm.iloc[-1]}
    days=sorted(endpoints);pairs=[]
    for prev,cur in zip(days[:-1],days[1:]):
        before,after=endpoints[prev],endpoints[cur]
        gap=(datetime.strptime(cur,'%Y%m%d')-datetime.strptime(prev,'%Y%m%d')).days
        row={'date':cur,'previous_trading_date':prev,'block':after['am_first'].block,
             'previous_block':before['pm_last'].block,'calendar_day_gap':gap,
             'calendar_close_to_next_open_minutes':1110+1440*(gap-1)}
        for name,v in zip(ENDPOINTS,[after['am_last'],after['pm_first'],before['pm_last'],after['am_first']]):
            row[name+'_raw_abs']=abs(float(v.raw_log_return))
            row[name+'_deseasonalized_abs']=abs(float(v.deseasonalized_log_return))
            row[name+'_seasonality_scale']=float(v.seasonality_scale)
            row[name+'_interval_start']=str(v.bar_start)
            row[name+'_interval_end']=str(v.bar_end) if 'bar_end' in v else str(v.interval_end)
        pairs.append(row)
    pd.DataFrame(pairs).to_csv(OUT/'paired_dates.csv',index=False)
    scopes={'all_sample_exploratory':pairs,
            'test_with_both_days_in_test':[q for q in pairs if q['block']==q['previous_block']=='test']}
    results={};invariance=[];summary=[]
    for scope,rows in scopes.items():
        n=len(rows);gaps=np.array([q['calendar_day_gap'] for q in rows])
        assert n>0
        values={v:np.array([[q[name+'_'+v+'_abs'] for name in ENDPOINTS] for q in rows])
                for v in ('raw','deseasonalized')}
        subsets={'all_calendar_gaps':np.ones(n,dtype=bool),'calendar_day_gap_eq_1':gaps==1}
        result={'n_base_paired_dates':n,'first_current_date':rows[0]['date'],'last_current_date':rows[-1]['date'],
            'calendar_day_gap_counts':dict(sorted(Counter(map(str,gaps)).items())),
            'calendar_close_to_next_open_minutes_counts':dict(sorted(Counter(str(q['calendar_close_to_next_open_minutes']) for q in rows).items())),
            'endpoint_time_of_day':{k:sorted({q[k+'_interval_start'].split(' ')[-1] for q in rows}) for k in ENDPOINTS},
            'endpoint_scale_values':{k:sorted({q[k+'_seasonality_scale'] for q in rows}) for k in ENDPOINTS},'subsets':{}}
        for subset,eligible in subsets.items():
            result['subsets'][subset]={v:{'point':point(x,eligible),'bootstrap':[]} for v,x in values.items()}
            for k in ('lunch_pearson','night_pearson','difference_night_minus_lunch'):
                invariance.append(abs(result['subsets'][subset]['raw']['point'][k]-result['subsets'][subset]['deseasonalized']['point'][k]))
        for length in BLOCKS:
            rng=np.random.default_rng(SEED_CLOCK)
            starts=rng.integers(0,n,(N_CLOCK,math.ceil(n/length)))
            indices=((starts[:,:,None]+np.arange(length))%n).reshape(N_CLOCK,-1)[:,:n]
            for subset,eligible in subsets.items():
                deltas={}
                for variant,x in values.items():
                    stats,delta=boot(x,indices,eligible)
                    stats.update(block_trading_days=length,seed=SEED_CLOCK)
                    result['subsets'][subset][variant]['bootstrap'].append(stats)
                    deltas[variant]=delta
                valid=np.isfinite(deltas['raw'])&np.isfinite(deltas['deseasonalized'])
                invariance.append(float(np.max(np.abs(deltas['raw'][valid]-deltas['deseasonalized'][valid]))))
                b=result['subsets'][subset]['raw']['bootstrap'][-1]
                summary.append({'scope':scope,'subset':subset,**result['subsets'][subset]['raw']['point'],
                    'block_trading_days':length,'ci95_lower':b['difference_ci95'][0],'ci95_upper':b['difference_ci95'][1],
                    'valid_replicates':b['n_boot_valid_difference'],'invalid_replicates':b['n_boot_invalid_difference']})
        results[scope]=result
    maxerr=max(invariance);assert maxerr<1e-12
    trdays=processed.loc[processed.block.eq('test'),'trading_date'].unique()
    assert results['test_with_both_days_in_test']['n_base_paired_dates']==len(trdays)-1
    out={'status':'passed','symbol':primary,'frequency_minutes':5,'source_rows':len(processed),
        'source_trading_days':len(days),'inputs':evidence,'new_parameter_fits':0,
        'design':'Retrospective paired-date check after the 8:1:1 re-split; not a new blind test',
        'difference_definition':'corr(abs(previous PM last),abs(current AM first)) - corr(abs(current AM last),abs(current PM first))',
        'bootstrap_design':'Circular moving blocks on full scope date grid; common draws for both correlations; gap==1 selected after resampling',
        'seed':SEED_CLOCK,'bootstrap_replicates_per_length':N_CLOCK,'block_lengths_trading_days':list(BLOCKS),
        'max_raw_vs_deseasonalized_point_or_bootstrap_difference':maxerr,'scopes':results}
    dump(OUT/'clock_check.json',out)
    pd.DataFrame(summary).to_csv(OUT/'clock_summary.csv',index=False)
    print('CLOCK',len(pairs),'all pairs;',len(trdays)-1,'test pairs; invariance',maxerr,flush=True)
    return out


def train_bootstrap(primary,tr,scale,sessions,evidence):
    target=OUT/'training_reselection.json'
    dates=tr.trading_date.to_numpy();days=np.unique(dates);x=tr.raw_log_return.to_numpy()/scale
    assert abs(x.std(ddof=1)-1)<1e-12
    day_index={d:np.flatnonzero(dates==d) for d in days}
    rng=np.random.default_rng(SEED_RESELECT)
    draws=[rng.choice(days,size=len(days),replace=True) for _ in range(N_RESELECT)]
    metadata={'symbol':primary,'seed':SEED_RESELECT,'n_replicates':N_RESELECT,'train_scale_fixed':scale,
        'train_days':len(days),'train_returns':len(x),'M_grid':[3,5,8,12,20,30],'N_grid':list(range(1,9)),
        'beta_bounds':[.05,20.],'log_beta_tolerance':.002,
        'design':'Whole training-day resampling with a new session id for every AM/PM occurrence; fixed original new-training scale; reselect M/bins/beta/N each draw',
        'inputs':evidence}
    reps=[]
    if target.exists():
        prior=json.loads(target.read_text('utf-8'))
        for k,v in metadata.items():assert prior[k]==v,('resume mismatch',k)
        reps=prior['replicates']
    start=time.monotonic()
    for rep,draw in enumerate(draws):
        if rep<len(reps):continue
        chunks=[];labels=[];offset=0;expected_lengths=[]
        for d in draw:
            ix=day_index[d];original=sessions[ix]
            _,codes=np.unique(original,return_inverse=True)
            chunks.append(x[ix]);labels.append(codes+offset)
            expected_lengths.extend(np.bincount(codes).tolist())
            offset+=int(codes.max())+1
        xb=np.concatenate(chunks);sb=np.concatenate(labels)
        boundaries=np.r_[0,np.flatnonzero(np.diff(sb)!=0)+1,len(sb)]
        assert np.array_equal(np.diff(boundaries),expected_lengths)
        assert len(np.unique(sb))==len(expected_lengths)==offset
        m,window_rows=fit.select_window(xb,sb,fit.build_binning,metadata['M_grid'])
        assert m is not None
        v,_=fit.rolling_variance(xb,m,sb)
        expected_windows=sum(max(0,n-m+1) for n in expected_lengths)
        assert len(v)==expected_windows
        spec=fit.build_binning(xb,v)
        best,table=fit.fit_hierarchy(xb,v,spec,metadata['N_grid'],beta_bounds=(.05,20.))
        assert best is not None and all(q['status']=='ok' for q in table),table
        row={'rep':rep,'M':m,'N':best['N'],'beta':best['beta'],'error':best['error'],
             'draw_sha256':hashlib.sha256('|'.join(draw).encode()).hexdigest(),
             'sample_rows':len(xb),'session_occurrences':offset,'n_windows':len(v),
             'selection_table':table,'window_rows':window_rows}
        reps.append(row)
        dump(target,{**metadata,'status':'running','replicates':reps})
        print('RESELECT',rep+1,N_RESELECT,'M',m,'N',best['N'],'elapsed',round(time.monotonic()-start,1),flush=True)
    result={**metadata,'status':'passed','replicates':reps,
        'M_counts':dict(sorted(Counter(str(q['M']) for q in reps).items())),
        'N_counts':dict(sorted(Counter(str(q['N']) for q in reps).items())),
        'all_session_boundary_checks_passed':True}
    assert len(reps)==N_RESELECT
    dump(target,result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['stats','bootstrap','all'],default='all')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    primary,df,tr,scale,sessions,evidence=source()
    if args.stage in ['stats','all']:
        processed,_=seasonality(df,tr,evidence)
        clock_check(primary,processed,evidence)
    if args.stage in ['bootstrap','all']:
        train_bootstrap(primary,tr,scale,sessions,evidence)
    print('AUX COMPLETE',args.stage,primary,flush=True)


if __name__=='__main__':main()

