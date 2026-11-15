"""Causal daily-background extension of the frozen hierarchy density model.

Only new directories are written. The protocol is retrospective: old holdout
outcomes were viewed before this extension was proposed. Training and validation
select the new settings before new test/2026 scoring is executed.
"""
from pathlib import Path
from functools import lru_cache
import argparse
import hashlib
import importlib.util
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
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'依据/811重划实验_20261114'
DATA=OLD/'data'
OUT=ROOT/'依据/条件波动拓展_20261114/experiment'
RUNTIME=OUT/'runtime'
PRIMARY='601318.SH'
H_GRID=[5,10,20,40,60]
N_GRID=[1,8]
WARMUP=60
REPS=2000
SEED=20261114
BOUNDS=np.log([[.05,64.],[.02,50.]])
LOG_X_GRID=np.linspace(math.log(1e-8),math.log(1e4),2049)
ENGINE=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text('utf-8'))
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.writing')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)

def init():
    global ENGINE
    OUT.mkdir(parents=True,exist_ok=True);RUNTIME.mkdir(exist_ok=True)
    source=OLD/'fits/runtime/fit_engine.py';target=RUNTIME/'fit_engine.py'
    if target.exists():assert sha(source)==sha(target)
    else:shutil.copy2(source,target)
    spec=importlib.util.spec_from_file_location('extension_frozen_engine',target)
    ENGINE=importlib.util.module_from_spec(spec);spec.loader.exec_module(ENGINE)

def symbols():return pd.read_csv(DATA/'selected_8.csv').sort_values('eligible_rank').symbol.tolist()

def identity():
    return {'script_sha256':sha(__file__),'runtime_sha256':sha(RUNTIME/'fit_engine.py'),
            'split_sha256':sha(DATA/'split_manifest.json'),'selection_sha256':sha(DATA/'selected_8.csv'),
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__}}

def protocol():
    init()
    p={'version':1,**identity(),'design':'Retrospective extension; old historical and 2026 results were already viewed; not a blind test.',
       'primary':PRIMARY,'symbols':symbols(),'h_grid_observed_update_days':H_GRID,'N_grid':N_GRID,
       'update':'q[d+1]=2**(-1/h)*q[d]+(1-2**(-1/h))*mean(x[d]**2); next retained observed day only',
       'purge':'not fitted, not scored, not used to update q; missing days do not decay q',
       'initial_q':1.,'2026_initial_q':1.,'training_warmup_observed_days':WARMUP,
       'training_mask':'same first 60 retained observed training days excluded for every new static/dynamic estimator',
       'parameter_selection':'primary conditional N8 training likelihood; choose h using valid log score only; transfer same h to all stocks and all dynamic families',
       'shape_bounds':[.05,64.],'background_multiplier_bounds':[.02,50.],
       'MLE':'zero location; estimate beta and c from training residuals; Gaussian c is mean squared residual',
       'optimizer':'three log-parameter starts, L-BFGS-B ftol=1e-11 gtol=1e-6 maxiter=300; boundaries recorded and notified',
       'shape_starts':{'1':[.3,2.,16.],'8':[1.,8.,32.]},
       'logabs_interpolation':{'size':2049,'range':[1e-8,1e4],'method':'PCHIP log-density'},
       'density_engine':{'fft_points':65536,'quadrature_points':4096,'verification_fft':262144,'verification_quadrature':16384},
       'score':'log p_N(x/sqrt(c*q); beta,v0=1)-0.5*log(c*q)',
       'evaluation':'train_fit,valid,test,2026_all,2026_after20; reference old joint uses original full training data',
       'CI':{'repetitions':REPS,'seed':SEED,'block_days':[1,5,10]},
       'coverage_levels':[.005,.01,.025,.05,.95,.975,.99,.995]}
    path=OUT/'protocol.json'
    if path.exists():
        old=load(path);assert {k:old[k] for k in p}==p,'Protocol mismatch';return old
    p['created_utc']=datetime.now(timezone.utc).isoformat();dump(path,p);return p

def frame(symbol,period):
    path=DATA/('historical' if period=='historical' else 'oos2026')/f'returns_{symbol}_5min.csv.gz'
    df=pd.read_csv(path,dtype={'trading_date':str},float_precision='round_trip')
    df=df.sort_values(['trading_date','session_id','bar_start']).reset_index(drop=True)
    assert not df.duplicated(['trading_date','session_id','bar_start']).any()
    return df,path

def historical_training(symbol):
    df,path=frame(symbol,'historical')
    scale=float(df.loc[df.block.eq('train'),'raw_log_return'].std(ddof=1))
    assert abs(scale-load(OLD/f'fits/{symbol}_5min_raw.base.json')['scale'])<1e-14
    df['x']=df.raw_log_return/scale
    return df,scale,path

def causal_q(df,h):
    """Return only retained observations; purge never affects state or masks."""
    d=df.loc[df.block.ne('purge')].copy().reset_index(drop=True)
    q=np.empty(len(d));ordinal=np.empty(len(d),dtype=np.int64)
    state=1.;lam=2.**(-1./h);audit=[]
    for j,(day,indices) in enumerate(d.groupby('trading_date',sort=True).indices.items()):
        ix=np.asarray(indices);q[ix]=state;ordinal[ix]=j
        values=d.iloc[ix].x.to_numpy();moment=float(np.mean(values*values))
        next_state=max(lam*state+(1.-lam)*moment,1e-12)
        audit.append({'trading_date':day,'q_before':state,'mean_x2_after_observation':moment,'q_next':next_state,'n':len(ix)})
        state=next_state
    d['q']=q;d['observed_day_ordinal']=ordinal
    return d,audit

def fit_mask(d):
    days=np.sort(d.loc[d.block.eq('train'),'trading_date'].unique())
    assert len(days)>WARMUP
    return d.block.eq('train') & d.trading_date.isin(days[WARMUP:])

@lru_cache(maxsize=12)
def density_table(n,beta):
    g=ENGINE.grid(n,float(beta))
    lp=ENGINE.logpdf(np.exp(LOG_X_GRID),g)
    return PchipInterpolator(LOG_X_GRID,lp,extrapolate=False),float(ENGINE.logpdf(np.array([0.]),g)[0])

def log_unit(z,n,beta):
    z=np.asarray(z,dtype=float)
    if n==1:return stats.t.logpdf(z,df=2*beta+2,scale=math.sqrt(beta/(beta+1)))
    interp,zero=density_table(n,float(beta));out=np.full(len(z),zero)
    nz=np.abs(z)>1e-8
    assert not np.any(np.abs(z)>1e4),'Interpolation range exceeded'
    out[nz]=interp(np.log(np.abs(z[nz])))
    assert np.isfinite(out).all()
    return out

def model_logpdf(x,q,param):
    scale=np.sqrt(param['c']*np.asarray(q))
    if param['family']=='Gaussian':return stats.norm.logpdf(x,loc=0,scale=scale)
    return log_unit(np.asarray(x)/scale,param['N'],param['beta'])-np.log(scale)

def fit_mle(x,q,n):
    start=time.perf_counter();x=np.asarray(x);q=np.asarray(q);z=x/np.sqrt(q)
    c0=float(np.mean(z*z))
    if n==0:
        return {'family':'Gaussian','N':None,'beta':None,'c':c0,'mean_train_log_score':float(np.mean(stats.norm.logpdf(z,scale=math.sqrt(c0))-.5*np.log(q))),
                'optimizer_success':True,'parameter_boundary':False,'starts':[],'elapsed_seconds':time.perf_counter()-start}
    qterm=.5*np.log(q)
    def loss(v):
        beta,c=np.exp(v)
        return -float(np.mean(log_unit(z/math.sqrt(c),n,float(beta))-.5*math.log(c)-qterm))
    results=[]
    for beta0 in ([.3,2.,16.] if n==1 else [1.,8.,32.]):
        v0=np.log([beta0,np.clip(c0,.025,45)])
        opt=minimize(loss,v0,method='L-BFGS-B',bounds=BOUNDS,options={'ftol':1e-11,'gtol':1e-6,'maxiter':300,'maxls':30})
        results.append({'success':bool(opt.success),'message':str(opt.message),'iterations':int(opt.nit),'evaluations':int(opt.nfev),'beta':float(np.exp(opt.x[0])),'c':float(np.exp(opt.x[1])),'negative_mean_log_likelihood':float(opt.fun)})
    good=[r for r in results if r['success']]
    assert good,results
    best=min(good,key=lambda r:r['negative_mean_log_likelihood'])
    beta,c=best['beta'],best['c'];boundary=beta<.0505 or beta>63.36 or c<.0202 or c>49.5
    return {'family':'hierarchy','N':n,'beta':beta,'c':c,'mean_train_log_score':-best['negative_mean_log_likelihood'],
            'optimizer_success':True,'parameter_boundary':boundary,'starts':results,'elapsed_seconds':time.perf_counter()-start}

def train_candidate(symbol,h,n,dynamic):
    name=f'{symbol}_'+('dynamic' if dynamic else 'static')+f'_N{n}'+(f'_h{h}' if dynamic else '')
    path=OUT/'training'/f'{name}.json'
    hist,scale,source=historical_training(symbol)
    d,audit=causal_q(hist,h);mask=fit_mask(d)
    inputs={'protocol':sha(OUT/'protocol.json'),'historical':sha(source)}
    if path.exists():
        saved=load(path);assert saved['input_sha256']==inputs;return saved
    x=d.x.to_numpy();q=d.q.to_numpy() if dynamic else np.ones(len(d));start=time.perf_counter()
    param=fit_mle(x[mask],q[mask],n)
    valid=d.block.eq('valid').to_numpy();valid_score=float(model_logpdf(x[valid],q[valid],param).mean())
    result={'status':'passed','symbol':symbol,'dynamic':dynamic,'h':h if dynamic else None,'name':name,'scale':scale,
            'training_days':int(d.loc[mask].trading_date.nunique()),'training_returns':int(mask.sum()),
            'valid_returns':int(valid.sum()),'valid_mean_log_score':valid_score,'parameter':param,'input_sha256':inputs,
            'training_start':str(d.loc[mask].trading_date.min()),'training_end':str(d.loc[mask].trading_date.max()),
            'elapsed_seconds':time.perf_counter()-start}
    dump(path,result)
    print('TRAINED',name,'beta',param['beta'],'c',param['c'],'valid',valid_score,'boundary',param['parameter_boundary'],'seconds',result['elapsed_seconds'],flush=True)
    return result

def choose_h():
    rows=[train_candidate(PRIMARY,h,8,True) for h in H_GRID]
    assert not any(r['parameter']['parameter_boundary'] for r in rows),'Candidate boundary hit: report before proceeding'
    best=max(rows,key=lambda r:r['valid_mean_log_score']);result={'status':'passed','selected_h':best['h'],
          'selection_source':'primary valid only; largest mean log score; fixed candidate order breaks exact ties',
          'candidates':[{'h':r['h'],'valid_mean_log_score':r['valid_mean_log_score'],'beta':r['parameter']['beta'],'c':r['parameter']['c']} for r in rows],
          'candidate_sha256':{r['name']:sha(OUT/'training'/f"{r['name']}.json") for r in rows},'protocol_sha256':sha(OUT/'protocol.json')}
    path=OUT/'selection.json'
    if path.exists():assert load(path)==result
    else:dump(path,result)
    return result

def train_symbol(symbol,h):
    results={}
    for dynamic in [False,True]:
        for n in [0,1,8]:
            model=('dynamic' if dynamic else 'static')+('_Gaussian' if n==0 else f'_N{n}')
            results[model]=train_candidate(symbol,h,n,dynamic)
    return results

def tests():
    """Small meaningful checks use train observations only, not generated markets."""
    d,_,_=historical_training(PRIMARY);d=d.loc[d.block.eq('train')].copy()
    d=d.loc[d.trading_date.isin(sorted(d.trading_date.unique())[:35])].reset_index(drop=True)
    base,a=causal_q(d,10);changed=d.copy();last=str(d.trading_date.max());changed.loc[changed.trading_date.eq(last),'x']*=3
    other,_=causal_q(changed,10);assert np.array_equal(base.q,other.q)
    future=d.copy();middle=sorted(d.trading_date.unique())[20];future.loc[future.trading_date.ge(middle),'x']*=2
    altered,_=causal_q(future,10);assert np.array_equal(base.loc[base.trading_date.le(middle),'q'],altered.loc[altered.trading_date.le(middle),'q'])
    purge=d.copy();purge.loc[purge.trading_date.eq(middle),'block']='purge';pa,_=causal_q(purge,10)
    purge.loc[purge.block.eq('purge'),'x']*=100;pb,_=causal_q(purge,10);assert np.array_equal(pa.q,pb.q)
    assert base.groupby('trading_date').q.nunique().eq(1).all()
    assert np.allclose([r['q_next'] for r in a[:-1]],[r['q_before'] for r in a[1:]],rtol=0,atol=0)
    probes=np.unique(np.r_[0,np.quantile(d.x,np.linspace(0,1,41))])
    cases=[]
    for n,beta in [(1,.5),(8,1.),(8,8.),(8,32.)]:
        approx=log_unit(probes,n,beta)
        exact=stats.t.logpdf(probes,df=2*beta+2,scale=math.sqrt(beta/(beta+1))) if n==1 else ENGINE.logpdf(probes,ENGINE.grid(n,beta,262144,2.,'characteristic'),16384)
        error=float(np.max(np.abs(approx-exact)));assert error<2e-6,(n,beta,error)
        cases.append({'N':n,'beta':beta,'max_log_density_error':error})
    result={'status':'passed','causality_future_change':True,'causality_same_day_change':True,'purge_ignored':True,'daily_q_constant':True,
            'scope':'first 35 real training days only','numerical_checks':cases}
    dump(OUT/'preflight_tests.json',result);print('PREFLIGHT PASSED',cases,flush=True)
    return result

def paired_ci(dates,difference,L):
    days,inverse=np.unique(dates,return_inverse=True);sums=np.bincount(inverse,weights=difference);counts=np.bincount(inverse)
    rng=np.random.default_rng(SEED);starts=rng.integers(0,len(days),(REPS,math.ceil(len(days)/L)))
    ix=((starts[:,:,None]+np.arange(L))%len(days)).reshape(REPS,-1)[:,:len(days)]
    means=sums[ix].sum(axis=1)/counts[ix].sum(axis=1)
    return [float(x) for x in np.quantile(means,[.025,.975])]

def cdf_unit(z,n,beta):
    z=np.asarray(z)
    if n==1:return stats.t.cdf(z,df=2*beta+2,scale=math.sqrt(beta/(beta+1)))
    g=ENGINE.grid(n,beta);y,dens=g[:2];stride=len(y)//4096;yy=y[::stride];weights=dens[::stride]*(y[1]-y[0])*stride
    good=weights>0;yy=yy[good];weights=weights[good];weights/=weights.sum()
    zz=np.exp(LOG_X_GRID);f=np.empty(len(zz))
    invsd=np.exp(-.5*yy)
    for i in range(0,len(zz),128):f[i:i+128]=special.ndtr(zz[i:i+128,None]*invsd).dot(weights)
    interp=PchipInterpolator(LOG_X_GRID,f,extrapolate=False)
    absval=np.abs(z);out=np.full(len(z),.5);nz=absval>1e-8
    out[nz]=.5+np.sign(z[nz])*(interp(np.log(absval[nz]))-.5)
    assert np.isfinite(out).all() and out.min()>=0 and out.max()<=1
    return out

def baseline_models(symbol):
    b=load(OLD/f'fits/{symbol}_5min_raw.base.json');one=next(r for r in b['selection_table'] if r['N']==1)
    models={'oldjoint_N1':{'family':'hierarchy','N':1,'beta':one['beta'],'c':1.},
            'old_Gaussian':{'family':'old_Gaussian','mean':b['gaussian']['mean'],'sd':b['gaussian']['sd'],'c':b['gaussian']['sd']**2}}
    if symbol==PRIMARY:
        eight=next(r for r in b['selection_table'] if r['N']==8)
        models['oldjoint_N8']={'family':'hierarchy','N':8,'beta':eight['beta'],'c':1.}
    return models

def parameters(symbol,h):
    result={}
    for dynamic in [False,True]:
        for n in [0,1,8]:
            name=f'{symbol}_'+('dynamic' if dynamic else 'static')+f'_N{n}'+(f'_h{h}' if dynamic else '')
            saved=load(OUT/'training'/f'{name}.json')
            assert saved['input_sha256']['protocol']==sha(OUT/'protocol.json')
            model=('dynamic' if dynamic else 'static')+('_Gaussian' if n==0 else f'_N{n}')
            result[model]=saved['parameter']
    result.update(baseline_models(symbol));return result

def evaluate(symbol):
    selection=load(OUT/'selection.json');h=selection['selected_h'];params=parameters(symbol,h)
    hist,scale,source=historical_training(symbol);old,audit_hist=causal_q(hist,h)
    later,later_source=frame(symbol,'2026');later['x']=later.raw_log_return/scale;later['block']='2026'
    new,audit_new=causal_q(later,h)
    d=pd.concat([old,new],ignore_index=True);input_hashes={'historical':sha(source),'2026':sha(later_source),'selection':sha(OUT/'selection.json'),'protocol':sha(OUT/'protocol.json')}
    path=OUT/f'{symbol}.evaluation.json'
    if path.exists():assert load(path)['input_sha256']==input_hashes;return load(path)
    start=time.perf_counter();x=d.x.to_numpy();q=d.q.to_numpy()
    for model,p in params.items():
        qm=q if model.startswith('dynamic') else np.ones(len(d));z=x/np.sqrt(qm*p['c'])
        if model=='old_Gaussian':
            lp=stats.norm.logpdf(x,loc=p['mean'],scale=p['sd']);pit=stats.norm.cdf(x,loc=p['mean'],scale=p['sd'])
        elif p['family']=='Gaussian':lp=model_logpdf(x,qm,p);pit=stats.norm.cdf(z)
        else:lp=model_logpdf(x,qm,p);pit=cdf_unit(z,p['N'],p['beta'])
        d['log_score_'+model]=lp;d['pit_'+model]=pit;d['predictive_variance_'+model]=qm*p['c']
    masks={'train_fit':fit_mask(d).to_numpy(),'valid':d.block.eq('valid').to_numpy(),'test':d.block.eq('test').to_numpy(),
           '2026_all':d.block.eq('2026').to_numpy(),'2026_after20':(d.block.eq('2026')&d.observed_day_ordinal.ge(20)).to_numpy()}
    scores=[];intervals=[];coverage=[];countrows={}
    comparisons=[('dynamic_N8','static_N8'),('dynamic_N8','dynamic_Gaussian'),('dynamic_N8','dynamic_N1'),('dynamic_N8','oldjoint_N1')]
    if symbol==PRIMARY:comparisons+=[('dynamic_N8','oldjoint_N8'),('static_N8','oldjoint_N8')]
    for block,mask in masks.items():
        part=d.loc[mask];days=part.trading_date.nunique();n=len(part);counts=part.groupby('trading_date').size()
        countrows[block]={'returns':n,'days':days,'complete_5min_days':int(counts.eq(46).sum()),'meets_original_gate':bool(n>=5000 and counts.eq(46).sum()>=60),
                          'first_date':str(part.trading_date.min()),'last_date':str(part.trading_date.max())}
        for model in params:
            lp=part['log_score_'+model].to_numpy();pit=part['pit_'+model].to_numpy();sortedpit=np.sort(pit)
            ks=float(max(np.max(np.arange(1,n+1)/n-sortedpit),np.max(sortedpit-np.arange(n)/n)))
            scores.append({'block':block,'model':model,'n':n,'mean_log_score':float(lp.mean()),'pit_uniform_distance':ks})
            coverage.append({'block':block,'model':model,'cdf_threshold_frequencies':{str(a):float(np.mean(pit<=a)) for a in [.005,.01,.025,.05,.95,.975,.99,.995]}})
        if block=='train_fit':continue
        for a,b in comparisons:
            difference=(part['log_score_'+a]-part['log_score_'+b]).to_numpy();dates=part.trading_date.to_numpy()
            intervals.append({'a':a,'b':b,'block':block,'n':n,'days':days,'mean':float(difference.mean()),
                              'ci95_by_block_days':{str(L):paired_ci(dates,difference,L) for L in [1,5,10]}})
    keep=['trading_date','session_id','bar_start','raw_log_return','block','x','q','observed_day_ordinal']+[c for c in d if c.startswith(('log_score_','pit_','predictive_variance_'))]
    scorepath=OUT/f'{symbol}.point_scores.csv.gz';d[keep].to_csv(scorepath,index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    dump(OUT/f'{symbol}.states.json',{'historical':audit_hist,'2026':audit_new})
    result={'status':'passed','symbol':symbol,'selected_h':h,'scale':scale,'input_sha256':input_hashes,'models':params,'counts':countrows,'scores':scores,
            'paired_intervals':intervals,'coverage':coverage,'point_scores_sha256':sha(scorepath),'elapsed_seconds':time.perf_counter()-start}
    dump(path,result);print('EVALUATED',symbol,'seconds',result['elapsed_seconds'],flush=True);return result

def validate(symbol):
    path=OUT/f'{symbol}.evaluation.json';ev=load(path);p=OUT/f'{symbol}.point_scores.csv.gz';assert sha(p)==ev['point_scores_sha256']
    d=pd.read_csv(p,dtype={'trading_date':str},float_precision='round_trip');checks=[]
    for model,par in ev['models'].items():
        if par['family']!='hierarchy':continue
        q=d.q.to_numpy() if model.startswith('dynamic') else np.ones(len(d));x=d.x.to_numpy();z=x/np.sqrt(q*par['c']);n=par['N'];b=par['beta']
        probes=np.unique(np.r_[0,np.quantile(z,np.linspace(0,1,41))])
        approx=log_unit(probes,n,b)
        if n==1:reference=stats.t.logpdf(probes,df=2*b+2,scale=math.sqrt(b/(b+1)));full=stats.t.logpdf(z,df=2*b+2,scale=math.sqrt(b/(b+1)))
        else:
            g=ENGINE.grid(n,b,262144,2.,'characteristic');reference=ENGINE.logpdf(probes,g,16384)
            # Direct evaluation of all observations verifies mean score, independent of interpolation.
            full=ENGINE.logpdf(z,g,16384)
        maxerror=float(np.max(np.abs(approx-reference)));assert maxerror<2e-6,(symbol,model,maxerror)
        density_log=full-.5*np.log(q*par['c']);saved=d['log_score_'+model].to_numpy();errors={}
        for block in ['train','valid','test','2026']:
            mask=d.block.eq(block).to_numpy();err=float(abs(density_log[mask].mean()-saved[mask].mean()));assert err<2e-6,(symbol,model,block,err);errors[block]=err
        checks.append({'model':model,'max_probe_log_density_error':maxerror,'mean_log_score_error_by_block':errors,'max_point_log_density_error':float(np.max(np.abs(density_log-saved)))})
    # State predictions are independently reconstructed using only past retained observations.
    state_errors=[]
    lam=2**(-1/ev['selected_h'])
    for period in ['historical','2026']:
        v=d.loc[d.block.eq('2026') if period=='2026' else d.block.ne('2026')];state=1.
        for day,part in v.groupby('trading_date',sort=True):
            state_errors.append(float(np.max(np.abs(part.q-state))));state=lam*state+(1-lam)*float(np.dot(part.x,part.x)/len(part))
    assert max(state_errors)<1e-12
    result={'status':'passed','symbol':symbol,'evaluation_sha256':sha(path),'density_checks':checks,'state_max_abs_error':max(state_errors),
            'all_new_optimizer_success':all(p['optimizer_success'] for k,p in ev['models'].items() if k.startswith(('static','dynamic'))),
            'new_parameter_boundaries':[k for k,p in ev['models'].items() if k.startswith(('static','dynamic')) and p['parameter_boundary']]}
    assert result['all_new_optimizer_success']
    dump(OUT/f'{symbol}.validation.json',result);print('VALIDATED',symbol,'boundaries',result['new_parameter_boundaries'],flush=True);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['tests','select','train','evaluate','validate','all'],default='tests');p.add_argument('--symbols');args=p.parse_args()
    init()
    if args.stage=='tests':tests();return
    proto=protocol()
    if args.stage=='select':tests();choose_h();return
    selection=load(OUT/'selection.json');assert selection['protocol_sha256']==sha(OUT/'protocol.json')
    syms=args.symbols.split(',') if args.symbols else symbols();assert set(syms)<=set(symbols())
    for sym in syms:
        if args.stage in ['train','all']:train_symbol(sym,selection['selected_h'])
        if args.stage in ['evaluate','all']:evaluate(sym)
        if args.stage in ['validate','all']:validate(sym)
    print('COMPLETE',args.stage,syms,flush=True)

if __name__=='__main__':main()
