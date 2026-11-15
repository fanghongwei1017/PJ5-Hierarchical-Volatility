"""8:1:1 retrospective refit on archived real returns; no old parameter JSON is read."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib
import json
import math
import platform
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
from scipy import special, stats
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'依据/811重划实验_20261114/data'
OUT = ROOT/'依据/811重划实验_20261114/fits'
RUNTIME = OUT/'runtime'
VERSION = 1
M_GRID = [3,5,8,12,20,30]
N_GRID = list(range(1,33))+[64,128,256,512,1024,2048,4096,8192]
SEED = 20261114
REPS = 2000

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)

def load(path):
    return json.loads(Path(path).read_text('utf-8'))

def ensure_runtime():
    """Freeze only reusable numerical code, without old dataset or output paths."""
    manifest=RUNTIME/'manifest.json'
    if manifest.exists():
        saved=load(manifest)
        for name,digest in saved['files'].items():
            assert sha(RUNTIME/name)==digest, name
        return saved
    RUNTIME.mkdir(parents=True,exist_ok=True)
    oldsource=ROOT/'工作文件/depth_extension_20261114.py'
    source=oldsource.read_text('utf-8'); lines=source.splitlines(keepends=True)
    needed={'grid','limit_grid','logpdf','Objective','fit'}
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in needed]
    assert {n.name for n in nodes}==needed
    pieces=['import math\nimport numpy as np\nfrom scipy import special\nfrom scipy.optimize import minimize_scalar\n\n']
    for node in nodes:pieces.append(''.join(lines[node.lineno-1:node.end_lineno])+'\n\n')
    (RUNTIME/'fit_engine.py').write_text(''.join(pieces),encoding='utf-8')
    oldruntime=ROOT/'依据/2026时间外验证/runtime/pj5_h'
    for p in oldruntime.rglob('*.py'):
        target=RUNTIME/'pj5_h'/p.relative_to(oldruntime)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    saved={'source':str(oldsource),'source_sha256':sha(oldsource),
           'frozen_function_names':sorted(needed),
           'files':{p.relative_to(RUNTIME).as_posix():sha(p) for p in RUNTIME.rglob('*.py')}}
    dump(manifest,saved)
    return saved

def init():
    global engine,rolling_variance,select_window,build_binning,fit_hierarchy,cascade,fit_seasonality,apply_seasonality
    runtime=ensure_runtime()
    sys.path.insert(0,str(RUNTIME))
    engine=importlib.import_module('fit_engine')
    from pj5_h.inference.fit import rolling_variance,select_window,fit_hierarchy
    from pj5_h.inference.binning import build_binning
    from pj5_h.models import cascade
    from pj5_h.data.seasonality import fit_seasonality,apply_seasonality
    cascade.set_grid(65536)
    return runtime

def cases():
    selected=pd.read_csv(DATA/'selected_8.csv',dtype={'symbol':str})
    order='eligible_rank' if 'eligible_rank' in selected else 'rank'
    if order in selected:selected=selected.sort_values(order)
    symbols=selected.symbol.tolist()
    assert len(symbols)==len(set(symbols))==8
    primary=symbols[0]
    return [dict(symbol=s,frequency_minutes=5,variant='raw',base_n_max=8 if s==primary else 5,primary=s==primary)
            for s in symbols]+[
        dict(symbol=primary,frequency_minutes=f,variant=v,base_n_max=n,primary=False)
        for f,v,n in [(5,'deseasonalized',5),(5,'raw_nonzero',5),(1,'raw',5),(15,'raw',5)]]

def key(case):return f"{case['symbol']}_{case['frequency_minutes']}min_{case['variant']}"

def protocol():
    runtime=init()
    value={'version':VERSION,'runner_sha256':sha(__file__),'split':'8:1:1 chronological; each nominal block drops its final 5 days',
        'split_sha256':sha(DATA/'split_manifest.json'),'selected_sha256':sha(DATA/'selected_8.csv'),
        'design':'Retrospective re-split and refit; all historical and 2026 outcomes were previously viewed; not a new blind test.',
        'cases':cases(),'M_grid':M_GRID,'N_grid':N_GRID,
        'base_fit':{'beta_bounds':[.05,20.],'log_beta_tolerance':.002,'optimizer':'original bounded one-dimensional search'},
        'mle_fit':{'beta_bounds':[.05,20.],'log_beta_tolerance':1e-5},
        'depth_fit':{'beta_bounds':'[0.05,max(20,4*N)]','log_grid':49,'tolerance':1e-7,
                     'plateau':'2 consecutive doubling changes each absolute<1e-4 and relative<0.001; relative distance to independently fitted limit<0.001'},
        'limit_fit':{'tau_bounds':[.0001,100.]},
        'numerical_check':{'grid':262144,'integration_points':16384,'domain_multiplier':2.,
                           'log_parameter_grid_points':97,'max_error':2e-6,'independent':'analytic characteristic inversion'},
        'bootstrap':{'unit':'observed whole trading day','method':'paired circular moving blocks; return-count weighting',
                     'block_days':[1,5,10],'replicates':REPS,'seed':SEED},
        'sample_gate':{'days':60,'returns':5000,'not_relaxed_for_811':True},
        'raw_nonzero_window':'remove exactly zero raw returns, then use retained observations within each original session; timestamps retained, no cross-session windows',
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__},
        'runtime_manifest_sha256':sha(RUNTIME/'manifest.json')}
    path=OUT/'protocol.json'
    if path.exists():
        old=load(path)
        assert {k:old[k] for k in value}==value,'Protocol mismatch'
        return old
    value['created_utc']=datetime.now(timezone.utc).isoformat()
    dump(path,value)
    return value

def ensure_tod(df):
    df=df.copy()
    df['trading_date']=df.trading_date.astype(str)
    if 'tod' not in df:
        df['tod']=pd.to_datetime(df.bar_start).dt.strftime('%H:%M')
    return df.sort_values(['trading_date','session_id','bar_start']).reset_index(drop=True)

def prepare(case, with_oos=True):
    p=DATA/'historical'/f"returns_{case['symbol']}_{case['frequency_minutes']}min.csv.gz"
    df=ensure_tod(pd.read_csv(p,dtype={'trading_date':str},float_precision='round_trip'))
    df['original_row']=np.arange(len(df))
    assert set(df.block.unique()) <= {'train','valid','test','purge'}
    assert np.isfinite(df.raw_log_return).all()
    if 'cross_session' in df:assert not df.cross_session.astype(bool).any()
    if 'close' in df and 'prev_close' in df:
        assert np.max(np.abs(np.log(df.close/df.prev_close)-df.raw_log_return))<1e-12
    seasonality=None
    if case['variant']=='deseasonalized':
        scales,diag=fit_seasonality(df.loc[df.block.eq('train')],case['frequency_minutes'])
        df=apply_seasonality(df,scales)
        value='deseasonalized_log_return'
        train=df.loc[df.block.eq('train')]
        amp=lambda c:float(train.groupby('tod')[c].apply(lambda x:np.abs(x).mean()).max()/train.groupby('tod')[c].apply(lambda x:np.abs(x).mean()).min())
        seasonality={'scales':scales,'diagnostics':diag,'mean_absolute_amplitude_before':amp('raw_log_return'),
                     'mean_absolute_amplitude_after':amp(value)}
    else:value='raw_log_return'
    if case['variant']=='raw_nonzero':df=df.loc[df.raw_log_return.ne(0.)].copy()
    scale=float(df.loc[df.block.eq('train'),value].std(ddof=1))
    assert np.isfinite(scale) and scale>0
    parts={}
    for block in ['train','valid','test']:
        d=df.loc[df.block.eq(block)].copy()
        assert len(d)>0
        parts[block]={'x':d[value].to_numpy()/scale,'dates':d.trading_date.to_numpy(),
                      'sessions':pd.factorize(d.trading_date+'|'+d.session_id.astype(str))[0],
                      'frame':d}
    assert abs(parts['train']['x'].std(ddof=1)-1)<1e-12
    hashes={'historical':sha(p),'split':sha(DATA/'split_manifest.json'),'protocol':sha(OUT/'protocol.json')}
    if with_oos and case['frequency_minutes']==5 and case['variant']=='raw':
        op=DATA/'oos2026'/f"returns_{case['symbol']}_5min.csv.gz"
        d=ensure_tod(pd.read_csv(op,dtype={'trading_date':str},float_precision='round_trip'))
        parts['2026']={'x':d.raw_log_return.to_numpy()/scale,'dates':d.trading_date.to_numpy(),
                       'sessions':pd.factorize(d.trading_date+'|'+d.session_id.astype(str))[0],'frame':d}
        hashes['oos2026']=sha(op)
    return parts,scale,seasonality,hashes

def counts(parts,frequency):
    rows={}
    for block,p in parts.items():
        daycounts=pd.Series(p['dates']).value_counts()
        complete=int((daycounts==46).sum()) if frequency==5 else None
        days=len(daycounts)
        rows[block]={'n_returns':len(p['x']),'n_days':days,'n_complete_5min_days':complete,
                     'first_date':str(min(p['dates'])),'last_date':str(max(p['dates'])),
                     'meets_observed_day_and_return_gate':bool(days>=60 and len(p['x'])>=5000),
                     'meets_complete_day_and_return_gate':bool(complete>=60 and len(p['x'])>=5000) if complete is not None else None,
                     'zero_fraction':float(np.mean(p['x']==0)),'normalized_sd':float(np.std(p['x'],ddof=1))}
    return rows

def base_fit(case):
    path=OUT/(key(case)+'.base.json')
    parts,scale,seasonality,hashes=prepare(case,False)
    if path.exists():
        saved=load(path);assert saved['input_sha256']==hashes
        return saved
    started=time.perf_counter();tr=parts['train']
    M,windows=select_window(tr['x'],tr['sessions'],build_binning,M_GRID)
    assert M is not None
    v,_=rolling_variance(tr['x'],M,tr['sessions']);spec=build_binning(tr['x'],v)
    best,table=fit_hierarchy(tr['x'],v,spec,range(1,case['base_n_max']+1))
    assert best is not None and all(r['status']=='ok' for r in table),table
    stable=[{'N':r['N'], 'stable_error':engine.Objective(tr['x'],v,spec).evaluate(engine.grid(r['N'],r['beta']))['error'],
             'base_error':r['error']} for r in table]
    assert max(abs(r['stable_error']-r['base_error']) for r in stable)<2e-5,stable
    result={'status':'passed','key':key(case),**case,'scale':scale,'M_star':M,'window_rows':windows,
        'n_windows_train':len(v),'bins':spec.to_dict(),'selection_table':table,'selected':best,
        'counts':counts(parts,case['frequency_minutes']),'seasonality':seasonality,
        'gaussian':{'mean':float(tr['x'].mean()),'sd':float(tr['x'].std(ddof=1))},
        'input_sha256':hashes,'base_stable_engine_check':stable,
        'elapsed_seconds':time.perf_counter()-started,'script_sha256':sha(__file__)}
    dump(path,result);print('BASE',key(case),'M',M,'N',best['N'],'E',best['error'],'seconds',result['elapsed_seconds'],flush=True)
    return result

def objective(case,base):
    parts,scale,seasonality,hashes=prepare(case,False)
    assert hashes==base['input_sha256'] and scale==base['scale']
    tr=parts['train'];v,_=rolling_variance(tr['x'],base['M_star'],tr['sessions'])
    spec=build_binning(tr['x'],v)
    assert spec.to_dict()==base['bins']
    return parts,engine.Objective(tr['x'],v,spec)

def depth_fit(case):
    base=base_fit(case);parts,obj=objective(case,base)
    path=OUT/(key(case)+'.depth.json');partial=OUT/(key(case)+'.depth.partial.json')
    basehash=sha(OUT/(key(case)+'.base.json'))
    if path.exists():
        r=load(path);assert r['base_sha256']==basehash
        return r
    previous=load(partial) if partial.exists() else {}
    if previous:assert previous['base_sha256']==basehash
    started=time.perf_counter()
    lim=previous.get('limit') or engine.fit(obj)
    assert lim['optimizer_success'] and not lim['parameter_boundary'],lim
    prior={r['N']:r for r in previous.get('fits',[])};rows=[];practical=False
    for n in N_GRID:
        ts=time.perf_counter()
        row=prior.get(n) or engine.fit(obj,n)
        assert row['optimizer_success'] and not row['parameter_boundary'],(key(case),n,row)
        if n not in prior:row['elapsed_seconds']=time.perf_counter()-ts
        rows.append(row)
        dump(partial,{'base_sha256':basehash,'fits':rows,'limit':lim,'last_completed_N':n})
        print('DEPTH',key(case),n,'E',row['error'],'beta',row['beta'],'seconds',row.get('elapsed_seconds'),flush=True)
        if n>=128:
            by={f['N']:f for f in rows}
            pairs=[(n//4,n//2),(n//2,n)]
            changes=[abs(by[a]['error']-by[b]['error']) for a,b in pairs]
            practical=all(d<1e-4 and d/by[b]['error']<.001 for d,(_,b) in zip(changes,pairs)) and abs(row['error']-lim['error'])/lim['error']<.001
            if practical:break
    best=min(rows,key=lambda f:f['error']);last=rows[-1]['N']
    result={'status':'passed' if practical else 'ceiling_reached_without_plateau','key':key(case),**case,
        'base_sha256':basehash,'fits':rows,'limit':lim,'last_N':last,
        'practical_plateau':practical,'best_N_in_tested_grid':best['N'],
        'finite_minimum_in_tested_grid':best['N']!=last,'best':best,
        'elapsed_seconds_this_invocation':time.perf_counter()-started}
    dump(path,result);return result

def evaluate_grid(x,grid,target=4096):
    # A symmetric density has equal values at +x and -x; deduplication is exact.
    unique,inverse=np.unique(np.abs(x),return_inverse=True)
    return engine.logpdf(unique,grid,target)[inverse]

def mle_fit(parts,N):
    x=parts['train']['x'];u,weights=np.unique(np.abs(x),return_counts=True)
    def nll(logb):
        b=math.exp(logb)
        lp=stats.t.logpdf(u,df=2*b+2,scale=math.sqrt(b/(b+1))) if N==1 else engine.logpdf(u,engine.grid(N,b))
        return -float(np.sum(lp*weights)/len(x))
    fit=minimize_scalar(nll,bounds=(math.log(.05),math.log(20.)),method='bounded',options={'xatol':1e-5})
    b=math.exp(fit.x)
    return {'N':N,'beta':b,'negative_mean_log_likelihood':float(fit.fun),'success':bool(fit.success),
            'beta_bounds':[.05,20.],'log_beta_tolerance':1e-5,'evaluations':int(fit.nfev)}

def paired_ci(dates,difference,L):
    days,inverse=np.unique(dates,return_inverse=True)
    sums=np.bincount(inverse,weights=difference);count=np.bincount(inverse)
    rng=np.random.default_rng(SEED)
    starts=rng.integers(0,len(days),(REPS,math.ceil(len(days)/L)))
    ix=((starts[:,:,None]+np.arange(L))%len(days)).reshape(REPS,-1)[:,:len(days)]
    means=sums[ix].sum(axis=1)/count[ix].sum(axis=1)
    return [float(x) for x in np.quantile(means,[.025,.975])]

def evaluate_case(case):
    base=base_fit(case);depth=depth_fit(case)
    path=OUT/(key(case)+'.evaluation.json')
    parts,scale,seasonality,hashes=prepare(case,True)
    hashes.update(base=sha(OUT/(key(case)+'.base.json')),depth=sha(OUT/(key(case)+'.depth.json')))
    if path.exists():
        old=load(path);assert old['input_sha256']==hashes;return old
    started=time.perf_counter()
    models={'Gaussian':None}
    for r in base['selection_table']:
        if r['N'] in {1,5,8,base['selected']['N']}:
            models['base_N'+str(r['N'])]={'N':r['N'],'beta':r['beta']}
    for r in depth['fits']:
        if r['N'] in {1,5,8,depth['last_N'],depth['best_N_in_tested_grid']}:
            models['depth_N'+str(r['N'])]={'N':r['N'],'beta':r['beta']}
    models['limit']={'N':None,'tau':depth['limit']['tau']}
    if case['primary']:
        for n in [1,8]:
            mpath=OUT/(key(case)+f'.mle_N{n}.json')
            if mpath.exists():
                mle=load(mpath);assert mle['base_sha256']==hashes['base']
            else:
                ts=time.perf_counter();mle={**mle_fit(parts,n),'base_sha256':hashes['base'],
                                           'elapsed_seconds':time.perf_counter()-ts}
                dump(mpath,mle)
            assert mle['success'];models[f'mle_N{n}']=mle
    lp={};scores=[];scoreframes=[]
    for name,param in models.items():
        if name!='Gaussian':
            g=engine.limit_grid(param['tau']) if param['N'] is None else engine.grid(param['N'],param['beta'])
        lp[name]={}
        for block,p in parts.items():
            if name=='Gaussian':value=stats.norm.logpdf(p['x'],base['gaussian']['mean'],base['gaussian']['sd'])
            else:value=evaluate_grid(p['x'],g)
            assert np.isfinite(value).all()
            lp[name][block]=value
            scores.append({'model':name,'block':block,'n':len(value),'mean_log_score':float(value.mean())})
    for block,p in parts.items():
        frame=p['frame'][['trading_date','session_id','bar_start','raw_log_return']].copy()
        frame['normalized_return']=p['x'];frame['block']=block
        for name in models:frame['log_score_'+name]=lp[name][block]
        scoreframes.append(frame)
    pd.concat(scoreframes,ignore_index=True).to_csv(OUT/(key(case)+'.point_scores.csv.gz'),index=False,
        float_format='%.17g',compression={'method':'gzip','mtime':0})
    pairs=[('base_N1','Gaussian'),('base_N1','base_N5'),('base_N5','Gaussian'),
           ('base_N1','depth_N'+str(depth['last_N'])),('base_N1','depth_N'+str(depth['best_N_in_tested_grid'])),
           ('limit','depth_N'+str(depth['last_N']))]
    if 'base_N8' in models:pairs += [('base_N1','base_N8'),('base_N8','Gaussian'),
                                    ('depth_N'+str(depth['last_N']),'base_N8')]
    pairs=list(dict.fromkeys(pairs))
    intervals=[]
    for block,p in parts.items():
        if block=='train':continue
        periods=[(block,np.ones(len(p['x']),dtype=bool))]
        if block=='2026' and case['primary']:
            periods += [('2026_Jan_Apr',p['dates']<='20260430'),('2026_May_Aug',p['dates']>='20260501')]
        for period,mask in periods:
            if not mask.any():continue
            for a,b in pairs:
                difference=(lp[a][block]-lp[b][block])[mask]
                intervals.append({'a':a,'b':b,'block':period,'n':int(mask.sum()),
                    'days':int(len(np.unique(p['dates'][mask]))),'mean':float(difference.mean()),
                    'ci95_by_block_days':{str(L):paired_ci(p['dates'][mask],difference,L) for L in [1,5,10]}})
    result={'status':'passed','key':key(case),**case,'input_sha256':hashes,
        'counts':counts(parts,case['frequency_minutes']),'models':models,'scores':scores,
        'paired_intervals':intervals,'point_scores_sha256':sha(OUT/(key(case)+'.point_scores.csv.gz')),
        'elapsed_seconds':time.perf_counter()-started}
    dump(path,result);print('SCORED',key(case),'seconds',result['elapsed_seconds'],flush=True)
    return result

def numerical_check(case):
    base=base_fit(case);depth=depth_fit(case);evaluation=evaluate_case(case)
    path=OUT/(key(case)+'.validation.json')
    if path.exists():
        r=load(path);assert r['evaluation_sha256']==sha(OUT/(key(case)+'.evaluation.json'));return r
    started=time.perf_counter();parts,obj=objective(case,base)
    allparts,_,_,_=prepare(case,True)
    probes=np.unique(np.concatenate([np.array([0.])]+[np.quantile(p['x'],np.linspace(0,1,41)) for p in allparts.values()]))
    final=depth['fits'][-1]
    refined=engine.fit(obj,final['N'],size=262144,target=16384,width_factor=2.,coarse_points=97)
    char=obj.evaluate(engine.grid(final['N'],final['beta'],262144,2.,'characteristic'),16384)
    assert abs(refined['error']-final['error'])<2e-6
    assert abs(char['error']-final['error'])<2e-6
    interior=[]
    if depth['finite_minimum_in_tested_grid']:
        n=depth['best_N_in_tested_grid']
        for nn in [max(1,n-1),n,n+1]:
            interior.append(engine.fit(obj,nn,size=262144,target=16384,width_factor=2.,coarse_points=97))
        assert min(interior,key=lambda r:r['error'])['N']==n
    checks=[]
    for name,param in evaluation['models'].items():
        if name=='Gaussian':continue
        if param['N'] is None:
            got=engine.logpdf(probes,engine.limit_grid(param['tau']))
            reference=engine.logpdf(probes,engine.limit_grid(param['tau'],262144,2.),16384)
        else:
            n,b=param['N'],param['beta']
            got=engine.logpdf(probes,engine.grid(n,b))
            reference=stats.t.logpdf(probes,df=2*b+2,scale=math.sqrt(b/(b+1))) if n==1 else engine.logpdf(probes,engine.grid(n,b,262144,2.,'characteristic'),16384)
        error=float(np.max(np.abs(got-reference)))
        assert error<2e-6,(key(case),name,error)
        checks.append({'model':name,'max_log_density_error':error,'probe_count':len(probes)})
    result={'status':'passed','key':key(case),'evaluation_sha256':sha(OUT/(key(case)+'.evaluation.json')),
        'final_refined_fit':refined,'final_refined_error_delta':refined['error']-final['error'],
        'final_characteristic_error_delta':char['error']-final['error'],'interior_refined_neighbors':interior,
        'observed_range_density_checks':checks,'elapsed_seconds':time.perf_counter()-started}
    dump(path,result);print('VERIFIED',key(case),'seconds',result['elapsed_seconds'],flush=True)
    return result

def pilot(case):
    base=base_fit(case);_,obj=objective(case,base);rows=[]
    for n in [1,8,32]:
        t=time.perf_counter();r=engine.fit(obj,n);rows.append({'N':n,'seconds':time.perf_counter()-t,'error':r['error'],'beta':r['beta']})
    dump(OUT/(key(case)+'.pilot.json'),rows);print('PILOT',rows,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['base','depth','evaluate','verify','all','pilot','protocol'],default='all')
    ap.add_argument('--case-indices');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    proto=protocol();selected=proto['cases']
    if args.case_indices:selected=[selected[int(i)] for i in args.case_indices.split(',')]
    if args.phase=='protocol':print(json.dumps(proto,ensure_ascii=False));return
    for case in selected:
        try:
            if args.phase=='pilot':pilot(case)
            else:
                base_fit(case)
                if args.phase in ['depth','evaluate','verify','all']:depth_fit(case)
                if args.phase in ['evaluate','verify','all']:evaluate_case(case)
                if args.phase in ['verify','all']:numerical_check(case)
        except Exception as error:
            dump(OUT/(key(case)+'.failure.json'),{'status':'failed','phase':args.phase,'error':repr(error),'utc':datetime.now(timezone.utc).isoformat()})
            raise
    print('COMPLETE',args.phase,len(selected),flush=True)

if __name__=='__main__':main()

