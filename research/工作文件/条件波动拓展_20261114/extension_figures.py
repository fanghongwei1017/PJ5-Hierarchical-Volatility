"""Vector figures from frozen causal forecasts; no evaluation-period rescaling."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'依据/条件波动拓展_20261114/experiment'
OUT=ROOT/'依据/条件波动拓展_20261114/report_metrics'
FIG=ROOT/'20261115_条件波动拓展版/figs'
OLD=ROOT/'依据/811重划实验_20261114'
PRIMARY='601318.SH'
MODELS=['oldjoint_N8','static_N8','dynamic_N8','dynamic_Gaussian']

def load(p):return json.loads(p.read_text('utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(name,df):df.to_csv(FIG/name,index=False,float_format='%.17g',na_rep='nan')
def tex(name,s):(FIG/name).write_text(s,encoding='utf-8')

def point_frame(symbol):
    path=EXP/f'{symbol}.point_scores.csv.gz';ev=load(EXP/f'{symbol}.evaluation.json')
    assert sha(path)==ev['point_scores_sha256']
    df=pd.read_csv(path,dtype={'trading_date':str,'block':str,'session_id':str},float_precision='round_trip',low_memory=False)
    assert np.isfinite(df.x).all() and np.isfinite(df.q).all()
    return df,ev,path

def mean_prediction(grid,daily,param,dynamic,fit):
    weights=daily['n'].to_numpy()/daily['n'].sum()
    state=daily.q.to_numpy() if dynamic else np.ones(len(daily))
    densities=np.vstack([np.exp(fit.model_logpdf(grid,np.full(len(grid),q),param)) for q in state])
    return np.sum(weights[:,None]*densities,axis=0)

def direct_high_precision(grid,daily,param,dynamic,fit):
    weights=daily['n'].to_numpy()/daily['n'].sum();state=daily.q.to_numpy() if dynamic else np.ones(len(daily))
    if param['family']=='Gaussian':
        return sum(w*stats.norm.pdf(grid,scale=math.sqrt(param['c']*q)) for w,q in zip(weights,state))
    g=fit.ENGINE.grid(param['N'],param['beta'],262144,2.,'characteristic')
    return sum(w*np.exp(fit.ENGINE.logpdf(grid/math.sqrt(param['c']*q),g,16384))/math.sqrt(param['c']*q) for w,q in zip(weights,state))

def figures(fit):
    frame,evaluation,source=point_frame(PRIMARY)
    base=load(OLD/f'fits/{PRIMARY}_5min_raw.base.json')
    edges=np.asarray(base['bins']['x_edges']);centers=(edges[:-1]+edges[1:])/2
    grid=np.linspace(edges[0],edges[-1],401);grid[200]=0.
    audit={};daily_frames={};curve_frames={}
    for block,mask in [('test',frame.block.eq('test')),('2026',frame.block.eq('2026'))]:
        d=frame.loc[mask].copy();assert d.groupby('trading_date').q.nunique().eq(1).all()
        daily=d.groupby('trading_date',sort=True).agg(n=('x','size'),q=('q','first'),R_d=('x',lambda v:float(np.mean(v.to_numpy()**2))))
        daily['predicted_variance']=evaluation['models']['dynamic_N8']['c']*daily.q
        daily['day_index']=np.arange(len(daily));daily['date_label']=[f'{s[4:6]}-{s[6:8]}' for s in daily.index]
        daily=daily.reset_index();daily_frames[block]=daily
        csv(f'conditional_variance_{block}.csv',daily[['day_index','trading_date','date_label','n','q','R_d','predicted_variance']])
        counts=np.histogram(d.x,edges)[0];empirical=counts/(len(d)*np.diff(edges))
        hist=pd.DataFrame({'x':centers,'count':counts,'empirical_density':empirical,'plot_density':np.where(counts>0,empirical,np.nan)})
        csv(f'conditional_hist_{block}.csv',hist)
        curves=pd.DataFrame({'x':grid});checks={}
        for model in MODELS:
            p=evaluation['models'][model];dynamic=model.startswith('dynamic')
            values=mean_prediction(grid,daily,p,dynamic,fit);assert np.isfinite(values).all() and np.all(values>0)
            curves[model]=values
            # Daily sample-count weights must agree with explicitly repeating a forecast for every return.
            ix=np.array([0,100,200,300,400]);per_point=[]
            q=d.q.to_numpy() if dynamic else np.ones(len(d))
            for x in grid[ix]:per_point.append(float(np.exp(fit.model_logpdf(np.full(len(d),x),q,p)).mean()))
            repeat_error=float(np.max(np.abs(values[ix]-per_point)));assert repeat_error<1e-12
            probe_ix=np.unique(np.r_[np.linspace(0,400,41,dtype=int),200])
            reference=direct_high_precision(grid[probe_ix],daily,p,dynamic,fit)
            numerical_error=float(np.max(np.abs(np.log(values[probe_ix])-np.log(reference))));assert numerical_error<2e-6
            point_lp=fit.model_logpdf(d.x.to_numpy(),q,p)
            score_error=float(np.max(np.abs(point_lp-d['log_score_'+model].to_numpy())));assert score_error<1e-12
            checks[model]={'sample_weight_sum':float(np.sum(daily.n/len(d))),'repeated_observation_mixture_abs_error':repeat_error,
                           'high_precision_mixture_max_log_error':numerical_error,'saved_point_log_score_max_error':score_error,
                           'plotted_probability_mass_trapezoid':float(np.trapezoid(values,grid)),
                           'mixture_second_moment':float(p['c']*np.average(q))}
        curve_frames[block]=curves;csv(f'conditional_density_{block}.csv',curves)
        audit[block]={'n':len(d),'days':len(daily),'empirical_inside_bins':int(counts.sum()),'empirical_outside_bins':int(len(d)-counts.sum()),
                      'empirical_mass_in_displayed_bins':float(np.sum(empirical*np.diff(edges))),
                      'daily_min_q':float(daily.q.min()),'daily_max_q':float(daily.q.max()),'models':checks}

    density=r'''\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.25cm},
width=.45\linewidth,height=5.8cm,ymode=log,ymin=0.00001,ymax=2.5,
xlabel={归一化收益 $x$},ylabel={概率密度},
grid=major,grid style={gray!15},unbounded coords=jump,
tick label style={font=\small},title style={font=\small},
legend style={font=\scriptsize,draw=none,at={(1.15,1.17)},anchor=south,legend columns=3,column sep=.6em},
legend cell align=left,
'''+f'xmin={edges[0]},xmax={edges[-1]}'+']\n'
    series=[('oldjoint_N8','gray!75!black,dashed,thick','联合八层'),('static_N8','orange!85!black,thick','似然八层'),
            ('dynamic_N8','blue!70!black,thick','条件八层'),('dynamic_Gaussian','violet!85!black,dashdotted,thick','同规则高斯')]
    for block,title in [('test','(a) 2019年测试段'),('2026','(b) 2026年全期')]:
        density+=r'\nextgroupplot[title={'+title+'}'+(',ylabel={}' if block=='2026' else '')+']\n'
        density+=r'\addplot[black!70,only marks,mark=*,mark size=1pt] table[x=x,y=plot_density,col sep=comma]{conditional_hist_'+block+'.csv};\n'
        if block=='test':density+=r'\addlegendentry{经验密度}'+'\n'
        for column,style,label in series:
            density+=r'\addplot['+style+r'] table[x=x,y='+column+r',col sep=comma]{conditional_density_'+block+'.csv};\n'
            if block=='test':density+=r'\addlegendentry{'+label+'}\n'
    density+=r'\end{groupplot}\end{tikzpicture}'+'\n'
    tex('conditional_density.tex',density)

    variance=r'''\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.25cm},
width=.45\linewidth,height=5.8cm,ymin=0,
xlabel={观测交易日},ylabel={波动强度（训练尺度）},
grid=major,grid style={gray!15},
tick label style={font=\small},title style={font=\small},
legend style={font=\scriptsize,draw=none,at={(1.15,1.17)},anchor=south,legend columns=3,column sep=.6em},
legend cell align=left]
'''
    for block,title in [('test','(a) 2019年测试段'),('2026','(b) 2026年全期')]:
        day=daily_frames[block];indices=np.unique(np.linspace(0,len(day)-1,5,dtype=int));ticks=','.join(str(x) for x in indices)
        labels=','.join(day.iloc[indices].date_label)
        ymax=max(1.,day.R_d.max(),day.predicted_variance.max())*1.1
        variance+=r'\nextgroupplot[title={'+title+'}'+f',xmin=0,xmax={len(day)-1},ymax={ymax},xtick={{{ticks}}},xticklabels={{{labels}}}'+(',ylabel={}' if block=='2026' else '')+']\n'
        variance+=r'\addplot[gray!70!black,line width=.55pt,no marks] table[x=day_index,y=R_d,col sep=comma]{conditional_variance_'+block+'.csv};\n'
        if block=='test':variance+=r'\addlegendentry{收盘后观测 $R_d$}'+'\n'
        variance+=r'\addplot[blue!70!black,thick,no marks] table[x=day_index,y=predicted_variance,col sep=comma]{conditional_variance_'+block+'.csv};\n'
        if block=='test':variance+=r'\addlegendentry{开盘前预报 $c q_d$}'+'\n'
        variance+=r'\addplot[orange!80!black,dashed,thick] coordinates {(0,1) ('+str(len(day)-1)+',1)};\n'
        if block=='test':variance+=r'\addlegendentry{固定背景 $v_0=1$}'+'\n'
    variance+=r'\end{groupplot}\end{tikzpicture}'+'\n'
    tex('conditional_scale.tex',variance)
    return audit

def calibration():
    rows=[];sources={}
    for path in sorted(EXP.glob('*.evaluation.json')):
        e=load(path);d,_,scorepath=point_frame(e['symbol']);sources[path.name]=sha(path);sources[scorepath.name]=sha(scorepath)
        for block in ['valid','test','2026_all','2026_after20']:
            mask=d.block.eq('2026') if block.startswith('2026') else d.block.eq(block)
            if block=='2026_after20':mask&=d.observed_day_ordinal.ge(20)
            part=d.loc[mask];n=len(part);assert n==e['counts'][block]['returns']
            for model in e['models']:
                u=part['pit_'+model].to_numpy();s=np.sort(u)
                distance=float(max(np.max(np.arange(1,n+1)/n-s),np.max(s-np.arange(n)/n)))
                expected=next(r['pit_uniform_distance'] for r in e['scores'] if r['model']==model and r['block']==block)
                assert abs(distance-expected)<1e-14
                for level in [.90,.95,.99]:
                    lower=(1-level)/2;upper=1-lower
                    rows.append({'symbol':e['symbol'],'block':block,'model':model,'n':n,'days':part.trading_date.nunique(),
                                 'nominal_central_coverage':level,'observed_central_coverage':float(np.mean((u>=lower)&(u<=upper))),
                                 'lower_tail_frequency':float(np.mean(u<lower)),'upper_tail_frequency':float(np.mean(u>upper)),
                                 'pit_uniform_distance':distance,'meets_sample_gate':e['counts'][block]['meets_original_gate']})
    pd.DataFrame(rows).to_csv(OUT/'calibration_summary.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    return sources

def main():
    FIG.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    module_path=ROOT/'工作文件/条件波动拓展_20261114/extension_fit.py'
    spec=importlib.util.spec_from_file_location('frozen_conditional_fit',module_path)
    fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit);fit.init()
    assert load(EXP/'summary.json')['status']=='passed'
    sources=calibration();audit=figures(fit)
    primary=load(EXP/f'{PRIMARY}.evaluation.json')
    summary={'status':'passed','primary_parameters':primary['models'],'primary_counts':primary['counts'],
             'primary_scores':primary['scores'],'primary_intervals':primary['paired_intervals'],
             'selected_h':load(EXP/'selection.json'),'source_sha256':sources,
             'calibration_sha256':sha(OUT/'calibration_summary.csv')}
    dump(OUT/'summary.json',summary)
    outputs={p.name:sha(p) for p in sorted(FIG.glob('conditional_*')) if p.suffix in ['.tex','.csv']}
    sources.update({'selection.json':sha(EXP/'selection.json'),'frozen_fit.py':sha(module_path),
                    'original_80_bins_base.json':sha(OLD/f'fits/{PRIMARY}_5min_raw.base.json')})
    dump(OUT/'figure_audit.json',{'status':'passed','periods':audit,'source_sha256':sources,'output_sha256':outputs,
         'script_sha256':sha(Path(__file__)),'mean_prediction_definition':'sum_d(n_d/n) p(x|q_d); density display only; scores use the forecast known before each day',
         'variance_definition':'R_d is mean(x^2) observed after day d, not the latent variance; c*q_d is known before day d',
         'empirical_bins':'original 80 training bins; density denominator includes all period observations; empty plotted bins are NaN'})
    print('FIGURES PASSED',len(outputs),'artifacts; calibration rows',len(pd.read_csv(OUT/'calibration_summary.csv')),flush=True)

if __name__=='__main__':main()
