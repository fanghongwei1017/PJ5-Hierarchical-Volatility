"""Independent scientific verification of the causal-background extension.

This validator never changes fitted models or frozen input data. Its preflight
checks operate on real training rows and include deliberate negative controls.
The final audit recomputes state paths, scores, PIT diagnostics and moving-block
intervals from the delivered point-level output and checks density references.
"""
from pathlib import Path
from functools import lru_cache
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import platform
import time

import numpy as np
import pandas as pd
import scipy
from scipy import integrate, special, stats

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OLD=ROOT/'依据/811重划实验_20261114'
DATA=OLD/'data'
EXP=ROOT/'依据/条件波动拓展_20261114/experiment'
OUT=ROOT/'依据/条件波动拓展_20261114/validation'
FIT_PATH=HERE/'extension_fit.py'
REPS=2000
SEED=20261114
LEVELS=[.005,.01,.025,.05,.95,.975,.99,.995]
FIT=None
ENGINE=None

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text('utf-8'))
def save(path,data):
    OUT.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def init():
    global FIT,ENGINE
    FIT=module('extension_under_test',FIT_PATH)
    ENGINE=module('independent_frozen_engine',EXP/'runtime/fit_engine.py')
    FIT.ENGINE=ENGINE

def source(symbol,period):
    path=DATA/('historical' if period=='historical' else 'oos2026')/f'returns_{symbol}_5min.csv.gz'
    d=pd.read_csv(path,dtype={'trading_date':str,'block':str,'session_id':str},float_precision='round_trip',low_memory=False)
    return d.sort_values(['trading_date','session_id','bar_start']).reset_index(drop=True),path

def independent_state(raw,h,scale):
    """Scalar recurrence in explicit date order, separate from worker group-index code."""
    rows=raw.loc[raw.block.ne('purge')].copy().reset_index(drop=True)
    values=rows.raw_log_return.to_numpy()/scale
    dates=rows.trading_date.to_numpy();state=1.;lam=math.exp(-math.log(2)/h)
    states=np.empty(len(rows));ordinal=np.empty(len(rows),dtype=int);summaries=[]
    dates_unique=sorted(set(dates))
    for j,day in enumerate(dates_unique):
        mask=dates==day;states[mask]=state;ordinal[mask]=j
        moment=math.fsum(float(v)*float(v) for v in values[mask])/int(mask.sum())
        updated=lam*state+(1-lam)*moment
        summaries.append(dict(trading_date=day,q_before=state,mean_x2_after_observation=moment,q_next=updated,n=int(mask.sum())))
        state=updated
    rows['x']=values;rows['q']=states;rows['observed_day_ordinal']=ordinal
    return rows,summaries

def independent_ci(dates,delta,L):
    """Explicit block construction and list aggregation; no worker CI helper."""
    days=sorted(set(dates));daily=[]
    for day in days:
        part=delta[dates==day];daily.append((math.fsum(float(v) for v in part),len(part)))
    rng=np.random.default_rng(SEED)
    seeds=rng.integers(len(days),size=(REPS,math.ceil(len(days)/L)))
    means=[];totals=np.array([a[0] for a in daily]);sizes=np.array([a[1] for a in daily])
    for starts in seeds:
        # Count how many times each complete day occurs, then weight its total.
        # This is independent of the worker's direct gathered-array summation.
        indices=np.concatenate([np.arange(int(start),int(start)+L)%len(days) for start in starts])[:len(days)]
        multiplicities=np.bincount(indices,minlength=len(days))
        numerator=float(np.dot(multiplicities,totals));denominator=int(np.dot(multiplicities,sizes))
        means.append(numerator/denominator)
    return np.quantile(means,[.025,.975])

def analytic_one_logpdf(z,beta):
    z=np.asarray(z)
    return (special.gammaln(beta+1.5)-special.gammaln(beta+1)
            -.5*math.log(2*math.pi*beta)-(beta+1.5)*np.log1p(z*z/(2*beta)))

@lru_cache(maxsize=32)
def reference_background(n,beta):
    g=ENGINE.grid(n,beta,262144,2.,'characteristic')
    y,density=g[:2];dy=y[1]-y[0]
    ix=density>0
    weights=density[ix]*dy
    assert abs(weights.sum()-1)<1e-10
    return g,y[ix],weights

def reference_cdf(z,n,beta):
    z=np.asarray(z)
    if n==1:
        # Incomplete-beta expression rather than scipy.stats.t.cdf used by worker.
        nu=2*beta+2;y=np.asarray(z)/math.sqrt(beta/(beta+1))
        tail=.5*special.betainc(nu/2,.5,nu/(nu+y*y))
        return np.where(y>=0,1-tail,tail)
    _,y,w=reference_background(n,float(beta));answer=np.empty(len(z))
    invsd=np.exp(-.5*y)
    for i in range(0,len(z),64):answer[i:i+64]=special.ndtr(z[i:i+64,None]*invsd).dot(w)
    return answer

def preflight():
    d,path=source('601318.SH','historical')
    scale=float(d.loc[d.block.eq('train'),'raw_log_return'].std(ddof=1));d['x']=d.raw_log_return/scale
    first=sorted(d.loc[d.block.eq('train'),'trading_date'].unique())[:90]
    d=d.loc[d.trading_date.isin(first)].copy().reset_index(drop=True)
    assert FIT.WARMUP==60,'Frozen training warmup must be 60 days'
    assert FIT.H_GRID==[5,10,20,40,60]
    original,_=FIT.causal_q(d,20)
    target=first[65]
    changed=d.copy();changed.loc[changed.trading_date.ge(target),'x']*=3.7
    mutated,_=FIT.causal_q(changed,20)
    before=original.trading_date.le(target)
    np.testing.assert_array_equal(original.loc[before,'q'],mutated.loc[before,'q'])
    assert np.max(np.abs(original.loc[~before,'q']-mutated.loc[~before,'q']))>1e-4
    same=d.copy();same.loc[same.trading_date.eq(target),'x']+=5
    other,_=FIT.causal_q(same,20)
    np.testing.assert_array_equal(original.loc[before,'q'],other.loc[before,'q'])
    assert np.max(np.abs(original.loc[~before,'q']-other.loc[~before,'q']))>1e-4
    # Negative control: a contemporaneous estimator must be rejected by this contract.
    leaky=d.groupby('trading_date').x.apply(lambda a:float(np.mean(a*a)))
    leaky_mutated=same.groupby('trading_date').x.apply(lambda a:float(np.mean(a*a)))
    assert leaky.loc[target]!=leaky_mutated.loc[target]
    # Five dates as a boundary gap. Their magnitudes cannot affect any retained path.
    purged=d.copy();gap=first[70:75];purged.loc[purged.trading_date.isin(gap),'block']='purge'
    p0,_=FIT.causal_q(purged,20);purged.loc[purged.block.eq('purge'),'x']=1e8
    p1,_=FIT.causal_q(purged,20);np.testing.assert_array_equal(p0.q,p1.q)
    assert not set(gap)&set(p0.trading_date)
    # Continuous state through valid/test labels; only a new episode resets state.
    labels=d.copy();labels.loc[labels.trading_date.isin(first[75:82]),'block']='valid';labels.loc[labels.trading_date.isin(first[82:]),'block']='test'
    relabeled,_=FIT.causal_q(labels,20);np.testing.assert_array_equal(relabeled.q,original.q)
    assert relabeled.loc[relabeled.trading_date.eq(first[75]),'q'].iloc[0]!=1.
    assert original.groupby('trading_date').q.nunique().eq(1).all()
    fitted=FIT.fit_mask(original)
    assert set(original.loc[fitted].trading_date)==set(first[60:])
    ref,_=independent_state(d,20,scale)
    err=float(np.max(np.abs(ref.q-original.q)));assert err<1e-12
    density_checks=[]
    probes=np.unique(np.r_[0.,np.quantile(original.x,np.linspace(0,1,41))])
    for n,beta in [(1,.3),(1,8.),(8,1.),(8,8.),(8,32.)]:
        p={'family':'hierarchy','N':n,'beta':beta,'c':1.}
        got=FIT.model_logpdf(probes,np.ones(len(probes)),p)
        expected=analytic_one_logpdf(probes,beta) if n==1 else ENGINE.logpdf(probes,reference_background(n,beta)[0],16384)
        density_error=float(np.max(np.abs(got-expected)));assert density_error<2e-6
        q=2.3;c=.7;p['c']=c;a=math.sqrt(q*c)
        transformed=FIT.model_logpdf(a*probes,np.full(len(probes),q),p)+math.log(a)
        transform_error=float(np.max(np.abs(got-transformed)));assert transform_error<1e-12
        # The incorrect transformation without the Jacobian differs by log(a).
        assert abs(math.log(a))>0.1
        cdf=FIT.cdf_unit(probes,n,beta);cdf_ref=reference_cdf(probes,n,beta)
        cdf_error=float(np.max(np.abs(cdf-cdf_ref)));assert cdf_error<2e-6
        integrand=lambda u:math.exp(float(FIT.model_logpdf(np.array([a*math.exp(u)]),np.array([q]),p)[0])+math.log(a)+u)
        integral=2*integrate.quad(integrand,math.log(1e-8),math.log(5000.),epsabs=3e-7,epsrel=3e-7,limit=200)[0]
        expected_mass=float(2*(reference_cdf(np.array([5000.]),n,beta)[0]-reference_cdf(np.array([1e-8]),n,beta)[0]))
        mass_error=abs(integral-expected_mass);assert mass_error<2e-6
        density_checks.append({'N':n,'beta':beta,'logpdf_error':density_error,'scale_transform_error':transform_error,'cdf_error':cdf_error,'normalization_truncated_integral_error':mass_error})
    gaussian={'family':'Gaussian','N':None,'beta':None,'c':.8}
    q=np.linspace(.4,2.,len(probes))
    np.testing.assert_allclose(FIT.model_logpdf(probes,q,gaussian),-.5*probes**2/(.8*q)-.5*np.log(2*math.pi*.8*q),rtol=0,atol=1e-14)
    result={'status':'passed','script_sha256':sha(__file__),'tested_script_sha256':sha(FIT_PATH),
        'source_sha256':sha(path),'scope':'first 90 actual training days; perturbation checks are algorithm diagnostics, not market evidence',
        'training_warmup':60,'causality_future_and_same_day':True,'negative_controls_detect_leakage_and_missing_jacobian':True,
        'purge_excluded_from_state':True,'historical_state_continuous_across_block_labels':True,
        'independent_state_max_error':err,'density_checks':density_checks}
    save(OUT/'preflight.json',result)
    print('INDEPENDENT PREFLIGHT PASSED',flush=True)
    return result

def masks(d):
    td=sorted(d.loc[d.block.eq('train'),'trading_date'].unique())
    return {'train_fit':d.block.eq('train')&d.trading_date.isin(td[60:]),'valid':d.block.eq('valid'),'test':d.block.eq('test'),
        '2026_all':d.block.eq('2026'),'2026_after20':d.block.eq('2026')&d.observed_day_ordinal.ge(20)}

def protocol_checks():
    p=load(EXP/'protocol.json')
    assert p['script_sha256']==sha(FIT_PATH)
    assert p['runtime_sha256']==sha(EXP/'runtime/fit_engine.py')==sha(OLD/'fits/runtime/fit_engine.py')
    assert p['split_sha256']==sha(DATA/'split_manifest.json')
    assert p['selection_sha256']==sha(DATA/'selected_8.csv')
    assert p['training_warmup_observed_days']==60 and p['2026_initial_q']==1.
    assert p['h_grid_observed_update_days']==[5,10,20,40,60]
    assert p['versions']=={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__}
    astroot=ast.parse(FIT_PATH.read_text('utf-8'))
    trainfun=next(n for n in astroot.body if isinstance(n,ast.FunctionDef) and n.name=='train_candidate')
    values=[n.value for n in ast.walk(trainfun) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    assert 'test' not in values and '2026' not in values
    files=[]
    for record in load(DATA/'final_manifest.json')['outputs']:
        path=Path(record['path']);assert sha(path)==record['sha256'],path
        files.append({'path':str(path.relative_to(ROOT)),'sha256':record['sha256']})
    selection=load(EXP/'selection.json');assert selection['protocol_sha256']==sha(EXP/'protocol.json')
    candidates=[]
    for h in p['h_grid_observed_update_days']:
        path=EXP/'training'/f'601318.SH_dynamic_N8_h{h}.json';a=load(path)
        assert sha(path)==selection['candidate_sha256'][a['name']]
        assert a['training_days']==519 and a['training_returns']==23874
        assert a['input_sha256']['protocol']==sha(EXP/'protocol.json')
        hist,_=source('601318.SH','historical');scale=float(hist.loc[hist.block.eq('train'),'raw_log_return'].std(ddof=1));d,_=independent_state(hist,h,scale)
        v=d.loc[d.block.eq('valid')]
        score=float(FIT.model_logpdf(v.x.to_numpy(),v.q.to_numpy(),a['parameter']).mean())
        assert abs(score-a['valid_mean_log_score'])<2e-11
        candidates.append((h,score))
    assert selection['selected_h']==max(candidates,key=lambda x:x[1])[0]
    return p,selection,files

def model_reference(x,q,param):
    if param['family']=='old_Gaussian':return stats.norm.logpdf(x,param['mean'],param['sd']),stats.norm.cdf(x,param['mean'],param['sd'])
    z=x/np.sqrt(q*param['c'])
    if param['family']=='Gaussian':return -.5*z*z-.5*np.log(2*math.pi*q*param['c']),special.ndtr(z)
    n,beta=param['N'],param['beta']
    lp=analytic_one_logpdf(z,beta) if n==1 else FIT.log_unit(z,n,beta)
    return lp-.5*np.log(q*param['c']),None

def verify_symbol(symbol,selected_h):
    started=time.perf_counter();evpath=EXP/f'{symbol}.evaluation.json';ev=load(evpath);pointpath=EXP/f'{symbol}.point_scores.csv.gz'
    assert ev['status']=='passed' and ev['selected_h']==selected_h
    assert ev['point_scores_sha256']==sha(pointpath)
    d=pd.read_csv(pointpath,dtype={'trading_date':str,'block':str,'session_id':str},float_precision='round_trip',low_memory=False)
    hist,hp=source(symbol,'historical');later,lp=source(symbol,'2026');later['block']='2026'
    assert ev['input_sha256']=={'historical':sha(hp),'2026':sha(lp),'selection':sha(EXP/'selection.json'),'protocol':sha(EXP/'protocol.json')}
    scale=float(hist.loc[hist.block.eq('train'),'raw_log_return'].std(ddof=1))
    assert abs(scale-ev['scale'])<1e-14
    h,ha=independent_state(hist,selected_h,scale);o,oa=independent_state(later,selected_h,scale)
    ref=pd.concat([h,o],ignore_index=True)
    for col in ['trading_date','session_id','bar_start','block','observed_day_ordinal']:
        assert d[col].astype(str).equals(ref[col].astype(str)),(symbol,col)
    np.testing.assert_array_equal(d.raw_log_return,ref.raw_log_return)
    state_error=float(np.max(np.abs(d.q-ref.q)));assert state_error<2e-12
    x_error=float(np.max(np.abs(d.x-ref.x)));assert x_error<2e-12
    assert d.loc[d.block.eq('2026')].q.iloc[0]==1.
    saved_states=load(EXP/f'{symbol}.states.json')
    for key,expected in [('historical',ha),('2026',oa)]:
        assert len(saved_states[key])==len(expected)
        for got,r in zip(saved_states[key],expected):
            for name in ['q_before','mean_x2_after_observation','q_next']:assert abs(got[name]-r[name])<2e-12
    mm=masks(d);results=[];optimization=[];parameter_sources={}
    oldbasepath=OLD/f'fits/{symbol}_5min_raw.base.json';oldbase=load(oldbasepath)
    for model,param in ev['models'].items():
        if model.startswith(('static','dynamic')):
            n=0 if param['family']=='Gaussian' else param['N']
            name=f"{symbol}_{'dynamic' if model.startswith('dynamic') else 'static'}_N{n}"+(f'_h{selected_h}' if model.startswith('dynamic') else '')
            trainpath=EXP/'training'/f'{name}.json';trained=load(trainpath)
            assert trained['parameter']==param,(symbol,model,'parameter provenance mismatch')
            assert trained['input_sha256']=={'protocol':sha(EXP/'protocol.json'),'historical':sha(hp)}
            assert trained['training_returns']==int(mm['train_fit'].sum())
            assert trained['training_days']==d.loc[mm['train_fit']].trading_date.nunique()
            parameter_sources[model]={'path':str(trainpath.relative_to(ROOT)),'sha256':sha(trainpath)}
        else:
            if model=='old_Gaussian':
                expected={'family':'old_Gaussian','mean':oldbase['gaussian']['mean'],'sd':oldbase['gaussian']['sd'],'c':oldbase['gaussian']['sd']**2}
            else:
                n=int(model.removeprefix('oldjoint_N'))
                beta=next(r['beta'] for r in oldbase['selection_table'] if r['N']==n)
                expected={'family':'hierarchy','N':n,'beta':beta,'c':1.}
            assert expected==param,(symbol,model,'archived baseline mismatch')
            parameter_sources[model]={'path':str(oldbasepath.relative_to(ROOT)),'sha256':sha(oldbasepath)}
        q=d.q.to_numpy() if model.startswith('dynamic') else np.ones(len(d));x=d.x.to_numpy()
        predicted=q*param['c'];assert np.max(np.abs(d['predictive_variance_'+model]-predicted))<1e-12
        score_ref,pit_ref=model_reference(x,q,param)
        score_error=float(np.max(np.abs(d['log_score_'+model]-score_ref)));assert score_error<2e-10
        pit=d['pit_'+model].to_numpy();assert np.isfinite(pit).all() and np.all((pit>=0)&(pit<=1))
        if param['family']=='hierarchy':
            z=x/np.sqrt(q*param['c']);probes=np.unique(np.r_[0,np.quantile(z,np.linspace(0,1,41))])
            got=FIT.cdf_unit(probes,param['N'],param['beta']);exact=reference_cdf(probes,param['N'],param['beta'])
            cdf_error=float(np.max(np.abs(got-exact)));assert cdf_error<2e-6
            # Stored PIT values are evaluated on observations, including zero and endpoints.
            ix=np.unique(np.r_[np.linspace(0,len(d)-1,257,dtype=int),np.argmin(z),np.argmax(z),np.flatnonzero(z==0)[:1]])
            pit_error=float(np.max(np.abs(pit[ix]-reference_cdf(z[ix],param['N'],param['beta']))));assert pit_error<2e-6
        else:cdf_error=pit_error=float(np.max(np.abs(pit-pit_ref)));assert pit_error<1e-13
        for block,mask in mm.items():
            rows=d.loc[mask];n=len(rows);daycounts=rows.groupby('trading_date').size()
            count=ev['counts'][block]
            assert count['returns']==n and count['days']==len(daycounts) and count['complete_5min_days']==int(daycounts.eq(46).sum())
            assert count['meets_original_gate']==bool(n>=5000 and daycounts.eq(46).sum()>=60)
            line=next(v for v in ev['scores'] if v['model']==model and v['block']==block)
            mean=math.fsum(float(v) for v in rows['log_score_'+model])/n
            assert abs(mean-line['mean_log_score'])<1e-12
            u=rows['pit_'+model].to_numpy();u.sort();cdfplus=np.arange(1,n+1)/n;cdfminus=np.arange(n)/n
            discrepancy=float(max(np.max(np.abs(cdfplus-u)),np.max(np.abs(cdfminus-u))))
            assert abs(discrepancy-line['pit_uniform_distance'])<1e-12
            coverage=next(v for v in ev['coverage'] if v['model']==model and v['block']==block)
            for level in LEVELS:assert abs(float(np.count_nonzero(u<=level)/n)-coverage['cdf_threshold_frequencies'][str(level)])<1e-15
        if model.startswith(('static','dynamic')):
            assert param['optimizer_success']
            if param['family']=='hierarchy':
                assert len(param['starts'])==3
                successful=[v for v in param['starts'] if v['success']]
                assert len(successful)>=2,(symbol,model,'fewer than two successful starts')
                best=min(v['negative_mean_log_likelihood'] for v in successful)
                all_best=min(v['negative_mean_log_likelihood'] for v in param['starts'])
                assert best-all_best<2e-6,(symbol,model,'failed start suggests a materially better objective')
                assert abs(best+param['mean_train_log_score'])<1e-12
                assert .05<=param['beta']<=64. and .02<=param['c']<=50.
            fit_score=next(v['mean_log_score'] for v in ev['scores'] if v['model']==model and v['block']=='train_fit')
            assert abs(fit_score-param['mean_train_log_score'])<2e-10
            optimization.append({'model':model,'parameter_boundary':param['parameter_boundary'],
                'all_starts_success':all(v['success'] for v in param['starts']),
                'failed_starts':[v for v in param['starts'] if not v['success']]})
        results.append({'model':model,'score_recomputation_max_error':score_error,'cdf_reference_max_error':cdf_error,'point_pit_reference_max_error':pit_error})
    intervals=[]
    for row in ev['paired_intervals']:
        part=d.loc[mm[row['block']]];delta=(part['log_score_'+row['a']]-part['log_score_'+row['b']]).to_numpy();dates=part.trading_date.to_numpy()
        assert abs(math.fsum(float(v) for v in delta)/len(delta)-row['mean'])<1e-12
        errors=[]
        for L in [1,5,10]:
            value=independent_ci(dates,delta,L);errors.append(float(np.max(np.abs(value-row['ci95_by_block_days'][str(L)]))))
        assert max(errors)<2e-12
        intervals.append({'a':row['a'],'b':row['b'],'block':row['block'],'max_interval_error':max(errors)})
    worker=load(EXP/f'{symbol}.validation.json');assert worker['status']=='passed' and worker['evaluation_sha256']==sha(evpath)
    result={'status':'passed','symbol':symbol,'state_error':state_error,'normalization_error':x_error,'models':results,
        'minimum_q':float(d.q.min()),'q_numerical_floor_triggered':bool(np.any(d.q<=1e-12) or min(r['q_next'] for r in ha+oa)<=1e-12),
        'parameter_sources':parameter_sources,'script_sha256':sha(__file__),
        'optimization':optimization,'paired_interval_checks':intervals,'boundary_models':[r['model'] for r in optimization if r['parameter_boundary']],
        'evaluation_sha256':sha(evpath),'point_scores_sha256':sha(pointpath),'worker_numerical_validation_sha256':sha(EXP/f'{symbol}.validation.json'),
        'elapsed_seconds':time.perf_counter()-started}
    save(OUT/f'{symbol}.json',result);print('INDEPENDENT VERIFIED',symbol,flush=True);return result

def final(symbols=None):
    preflight_result=load(OUT/'preflight.json')
    assert preflight_result['tested_script_sha256']==sha(FIT_PATH)
    p,selection,inputs=protocol_checks()
    syms=symbols or p['symbols'];assert set(syms)<=set(p['symbols'])
    checks=[verify_symbol(sym,selection['selected_h']) for sym in syms]
    for entry in inputs:assert sha(ROOT/entry['path'])==entry['sha256']
    result={'status':'passed','scope':'independent audit of all completed cases' if len(syms)==8 else 'partial selected symbols',
        'symbols':syms,'selected_h':selection['selected_h'],'protocol_sha256':sha(EXP/'protocol.json'),'script_sha256':sha(__file__),
        'input_files_unchanged':len(inputs),'causality_tests':preflight_result,'cases':checks,
        'boundaries':[{'symbol':r['symbol'],'models':r['boundary_models']} for r in checks if r['boundary_models']],
        'CDF_reference_scope':'zero, 41 actual transformed-return quantiles including endpoints, plus 257 regularly spaced actual observations per model',
        'log_density_full_observation_reference':'Worker checks all pointwise log densities against high-resolution characteristic inversion; audit checks the worker validation hash and all score/interval summaries independently.'}
    save(OUT/('independent_validation.json' if len(syms)==8 else 'partial_validation.json'),result)
    print('INDEPENDENT FINAL PASSED',len(syms),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['preflight','final'],default='preflight');ap.add_argument('--symbols')
    args=ap.parse_args();init()
    preflight() if args.stage=='preflight' else final(args.symbols.split(',') if args.symbols else None)
