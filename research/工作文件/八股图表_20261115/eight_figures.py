"""All-eight-stock figures from unchanged, independently audited forecasts."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'依据/条件波动拓展_20261114/experiment'
OLD=ROOT/'依据/811重划实验_20261114'
OUT=ROOT/'依据/八股图表_20261115'
FIG=ROOT/'20261115_八股对照版/figs'
NAMES={'601318.SH':'中国平安','600519.SH':'贵州茅台','600030.SH':'中信证券','600887.SH':'伊利股份',
       '600036.SH':'招商银行','603799.SH':'华友钴业','600516.SH':'方大炭素','601668.SH':'中国建筑'}
MODELS=['static_N8','dynamic_N8','dynamic_Gaussian']

def load(p):return json.loads(p.read_text('utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def write_csv(name,df):df.to_csv(FIG/name,index=False,float_format='%.17g',na_rep='nan')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result);return result

def daily_data(d,param):
    assert d.groupby('trading_date').q.nunique().eq(1).all()
    daily=d.groupby('trading_date',sort=True).agg(n=('x','size'),q=('q','first'),
            R_d=('x',lambda v:float(np.mean(v.to_numpy()**2))))
    daily['predicted_variance']=param['c']*daily.q
    daily['day_index']=np.arange(len(daily))
    daily['date_label']=[f'{s[4:6]}-{s[6:8]}' for s in daily.index]
    return daily.reset_index()

def mean_prediction(grid,daily,param,dynamic,fit):
    weights=daily.n.to_numpy()/daily.n.sum()
    states=daily.q.to_numpy() if dynamic else np.ones(len(daily))
    return sum(w*np.exp(fit.model_logpdf(grid,np.full(len(grid),q),param)) for w,q in zip(weights,states))

def reference_prediction(grid,daily,param,dynamic,fit,reference):
    weights=daily.n.to_numpy()/daily.n.sum()
    states=daily.q.to_numpy() if dynamic else np.ones(len(daily))
    if param['family']=='Gaussian':
        return sum(w*stats.norm.pdf(grid,scale=math.sqrt(param['c']*q)) for w,q in zip(weights,states))
    return sum(w*np.exp(fit.ENGINE.logpdf(grid/math.sqrt(param['c']*q),reference,16384))/math.sqrt(param['c']*q)
               for w,q in zip(weights,states))

def pair(e,other,block):
    return next(r for r in e['paired_intervals'] if (r['a'],r['b'],r['block'])==('dynamic_N8',other,block))

def title(index,symbol,block):
    short=r'（45日）' if symbol=='603799.SH' and block=='2026' else ''
    return f'({chr(97+index)}) {NAMES[symbol]} {symbol}{short}'

def group_start(kind):
    common=r'''\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 4,horizontal sep=1.05cm,vertical sep=1.45cm},
width=7.45cm,height=4.45cm,
grid=major,grid style={gray!15},unbounded coords=jump,
tick label style={font=\scriptsize},title style={font=\small},label style={font=\small},
legend style={font=\footnotesize,draw=none,at={(1.12,1.23)},anchor=south,legend columns=4,column sep=.6em},
legend cell align=left,
'''
    if kind=='scale':return common+r'ymin=0,ylabel={波动强度},scaled y ticks=false]'+'\n'
    return common+r'ymode=log,ymin=0.00001,ymax=5,ylabel={概率密度},ytick={0.00001,0.001,0.1,1}]'+'\n'

def write_plots(block,panels):
    scale=group_start('scale');density=group_start('density')
    for index,symbol in enumerate(NAMES):
        data=panels[(symbol,block)];day=data['daily'];stem=f'eight_conditional_{symbol[:6]}_{block}'
        ticks=np.unique(np.linspace(0,len(day)-1,5,dtype=int));tickstr=','.join(map(str,ticks))
        labels=','.join(day.iloc[ticks].date_label)
        ymax=max(1.,day.R_d.max(),day.predicted_variance.max())*1.08
        label='' if index%2==0 else ',ylabel={}'
        xlabel=',xlabel={观测交易日}' if index>=6 else ''
        scale+=r'\nextgroupplot[title={'+title(index,symbol,block)+'}'+f',xmin=0,xmax={len(day)-1},ymax={ymax:.17g},xtick={{{tickstr}}},xticklabels={{{labels}}}'+label+xlabel+']\n'
        for col,style,legend in [('R_d','gray!70!black,line width=.5pt','收盘后观测 $R_d$'),
                                  ('predicted_variance','blue!70!black,line width=.8pt','开盘前预报 $cq_d$')]:
            scale+=r'\addplot['+style+r',no marks] table[x=day_index,y='+col+r',col sep=comma]{'+stem+'_scale.csv};\n'
            if index==0:scale+=r'\addlegendentry{'+legend+'}\n'
        scale+=r'\addplot[orange!85!black,dashed,line width=.8pt] coordinates {(0,1) ('+str(len(day)-1)+',1)};\n'
        if index==0:scale+=r'\addlegendentry{固定背景 $v_0=1$}'+'\n'
        edges=data['edges'];xlabel=',xlabel={归一化收益 $x$}' if index>=6 else ''
        density+=r'\nextgroupplot[title={'+title(index,symbol,block)+'}'+f',xmin={edges[0]},xmax={edges[-1]}'+label+xlabel+']\n'
        density+=r'\addplot[black!70,only marks,mark=*,mark size=.85pt] table[x=x,y=plot_density,col sep=comma]{'+stem+'_hist.csv};\n'
        if index==0:density+=r'\addlegendentry{经验密度}'+'\n'
        for col,style,legend in [('static_N8','orange!85!black,line width=.8pt','静态似然八层'),
                                  ('dynamic_N8','blue!70!black,line width=.8pt','条件八层'),
                                  ('dynamic_Gaussian','violet!85!black,dashdotted,line width=.8pt','同规则高斯')]:
            density+=r'\addplot['+style+r'] table[x=x,y='+col+r',col sep=comma]{'+stem+'_density.csv};\n'
            if index==0:density+=r'\addlegendentry{'+legend+'}\n'
    suffix=r'\end{groupplot}\end{tikzpicture}'+'\n'
    (FIG/f'eight_conditional_scale_{block}.tex').write_text(scale+suffix,encoding='utf-8')
    (FIG/f'eight_conditional_density_{block}.tex').write_text(density+suffix,encoding='utf-8')

def main():
    assert FIG.is_dir() and OUT.is_dir()
    independent_path=ROOT/'依据/条件波动拓展_20261114/validation/independent_validation.json'
    independent=load(independent_path);assert independent['status']=='passed'
    cases={c['symbol']:c for c in independent['cases']}
    fit_path=ROOT/'工作文件/条件波动拓展_20261114/extension_fit.py'
    engine_path=EXP/'runtime/fit_engine.py'
    assert sha(fit_path)==independent['causality_tests']['tested_script_sha256']
    fit=module('audited_conditional_fit',fit_path)
    fit.ENGINE=module('audited_density_engine',engine_path)
    sources={str(p.relative_to(ROOT)):sha(p) for p in [independent_path,fit_path,engine_path,EXP/'selection.json',EXP/'protocol.json']}
    assert sources[str((EXP/'protocol.json').relative_to(ROOT))]==independent['protocol_sha256']
    protocol=load(EXP/'protocol.json');assert sha(engine_path)==protocol['runtime_sha256']
    assert load(EXP/'selection.json')['selected_h']==5
    panels={};audits={};summary=[]
    for symbol in NAMES:
        ep=EXP/f'{symbol}.evaluation.json';vp=EXP/f'{symbol}.validation.json';pp=EXP/f'{symbol}.point_scores.csv.gz'
        bp=OLD/f'fits/{symbol}_5min_raw.base.json'
        e=load(ep);v=load(vp);base=load(bp);case=cases[symbol]
        assert e['status']==v['status']=='passed' and e['selected_h']==5
        assert sha(ep)==v['evaluation_sha256']==case['evaluation_sha256']
        assert sha(pp)==e['point_scores_sha256']==case['point_scores_sha256']
        assert sha(vp)==case['worker_numerical_validation_sha256']
        for p in [ep,vp,pp,bp]:sources[str(p.relative_to(ROOT))]=sha(p)
        d=pd.read_csv(pp,dtype={'trading_date':str,'block':str,'session_id':str},float_precision='round_trip',low_memory=False)
        assert np.isfinite(d.x).all() and np.isfinite(d.q).all()
        edges=np.asarray(base['bins']['x_edges']);assert len(edges)==81 and edges[0]<0<edges[-1]
        centers=(edges[:-1]+edges[1:])/2
        grid=np.unique(np.r_[np.linspace(edges[0],edges[-1],401),0.])
        # One enlarged characteristic grid per fitted family, reused for both periods.
        references={model:fit.ENGINE.grid(e['models'][model]['N'],e['models'][model]['beta'],262144,2.,'characteristic')
                    for model in MODELS if e['models'][model]['family']!='Gaussian'}
        for block,score_block in [('test','test'),('2026','2026_all')]:
            part=d.loc[d.block.eq(block)].copy();n=len(part)
            assert n==e['counts'][score_block]['returns']
            daily=daily_data(part,e['models']['dynamic_N8']);stem=f'eight_conditional_{symbol[:6]}_{block}'
            write_csv(stem+'_scale.csv',daily[['day_index','trading_date','date_label','n','q','R_d','predicted_variance']])
            counts=np.histogram(part.x,edges)[0];empirical=counts/(n*np.diff(edges))
            hist=pd.DataFrame({'x':centers,'count':counts,'empirical_density':empirical,
                               'plot_density':np.where(counts>0,empirical,np.nan)})
            write_csv(stem+'_hist.csv',hist)
            curves=pd.DataFrame({'x':grid});checks={};calibration={}
            for model in MODELS:
                p=e['models'][model];dynamic=model.startswith('dynamic');q=part.q.to_numpy() if dynamic else np.ones(n)
                prediction=mean_prediction(grid,daily,p,dynamic,fit)
                assert np.isfinite(prediction).all() and np.all(prediction>0)
                curves[model]=prediction
                ix=np.unique(np.linspace(0,len(grid)-1,5,dtype=int))
                repeated=np.array([np.exp(fit.model_logpdf(np.full(n,x),q,p)).mean() for x in grid[ix]])
                weight_error=float(np.max(abs(repeated-prediction[ix])));assert weight_error<1e-12
                probe_ix=np.unique(np.r_[np.linspace(0,len(grid)-1,41,dtype=int),np.argmin(abs(grid))])
                reference=reference_prediction(grid[probe_ix],daily,p,dynamic,fit,references.get(model))
                high_error=float(np.max(abs(np.log(reference)-np.log(prediction[probe_ix]))));assert high_error<2e-6
                scores=fit.model_logpdf(part.x.to_numpy(),q,p)
                point_error=float(np.max(abs(scores-part['log_score_'+model])));assert point_error<1e-12
                stored=next(r for r in e['scores'] if r['model']==model and r['block']==score_block)
                assert abs(np.mean(scores)-stored['mean_log_score'])<1e-12
                u=part['pit_'+model].to_numpy();ordered=np.sort(u)
                pit=float(max(np.max(np.arange(1,n+1)/n-ordered),np.max(ordered-np.arange(n)/n)))
                assert abs(pit-stored['pit_uniform_distance'])<1e-14
                calibration[model]={'mean_log_score':float(np.mean(scores)),'pit_uniform_distance':pit,
                                    'central_95_coverage':float(np.mean((u>=.025)&(u<=.975)))}
                checks[model]={'repeated_weight_abs_error':weight_error,'high_precision_max_log_error':high_error,
                               'saved_point_log_score_max_error':point_error,'average_predictive_variance':float(p['c']*np.mean(q))}
            write_csv(stem+'_density.csv',curves)
            pair_static=pair(e,'static_N8',score_block);pair_gauss=pair(e,'dynamic_Gaussian',score_block)
            pair_one=pair(e,'dynamic_N1',score_block)
            row={'symbol':symbol,'name':NAMES[symbol],'block':score_block,'n':n,'days':len(daily),
                 'observed_second_moment':float(np.mean(part.x.to_numpy()**2)),
                 'average_predicted_variance':float(np.average(daily.predicted_variance,weights=daily.n)),
                 'dynamic_N8_minus_static_N8':pair_static['mean'],'static_ci95_low':pair_static['ci95_by_block_days']['5'][0],
                 'static_ci95_high':pair_static['ci95_by_block_days']['5'][1],
                 'dynamic_N8_minus_dynamic_Gaussian':pair_gauss['mean'],'gaussian_ci95_low':pair_gauss['ci95_by_block_days']['5'][0],
                 'gaussian_ci95_high':pair_gauss['ci95_by_block_days']['5'][1],
                 'dynamic_N8_minus_dynamic_N1':pair_one['mean'],'single_ci95_low':pair_one['ci95_by_block_days']['5'][0],
                 'single_ci95_high':pair_one['ci95_by_block_days']['5'][1]}
            for model in MODELS:
                row.update({model+'_'+key:value for key,value in calibration[model].items()})
            summary.append(row)
            audits[symbol+'/'+block]={'n':n,'days':len(daily),'empirical_inside_bins':int(counts.sum()),
                'empirical_outside_bins':int(n-counts.sum()),'displayed_empirical_mass':float(sum(empirical*np.diff(edges))),
                'sample_weight_sum':float(sum(daily.n/n)),'calibration':calibration,'models':checks,
                'paired_intervals_5day':{other:pair(e,other,score_block) for other in ['static_N8','dynamic_Gaussian','dynamic_N1']}}
            panels[(symbol,block)]={'daily':daily,'edges':edges}
            print(symbol,block,'passed',n,len(daily),flush=True)
    for block in ['test','2026']:write_plots(block,panels)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    dump(OUT/'summary.json',{'status':'passed','rows':summary,'source_sha256':sources,'selected_h':5,
                            'stock_order':list(NAMES),'retrospective':True})
    outputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(FIG.glob('eight_conditional_*'))}
    for p in [OUT/'summary.csv',OUT/'summary.json']:outputs[str(p.relative_to(ROOT))]=sha(p)
    # Recheck all frozen sources after computation; no old result has been overwritten.
    for relative,digest in sources.items():assert sha(ROOT/relative)==digest,relative
    dump(OUT/'figure_audit.json',{'status':'passed','source_sha256':sources,'output_sha256':outputs,'panels':audits,
          'script_sha256':sha(Path(__file__)),'new_fit_performed':False,
          'prediction':'sum_d (n_d/n) p(x|q_d); point scores use each actual prior-day forecast',
          'empirical':'each stock uses its original 80 training bins; denominator includes all period observations',
          'axes':'same training-standardization definition across stocks; variance y-axis limits differ by stock and period',
          'scope':'all 8 stocks, test and 2026 full period; Huayou 2026 has only 45 observed days'})
    print('PASSED',len(outputs),'outputs',flush=True)

def restyle():
    """Regenerate TeX from verified frozen CSV, retaining numerical audit evidence."""
    audit_path=OUT/'figure_audit.json';audit=load(audit_path)
    assert audit['status']=='passed'
    old_audit_sha=sha(audit_path);old_script_sha=audit['script_sha256']
    for relative,digest in audit['source_sha256'].items():assert sha(ROOT/relative)==digest,relative
    for relative,digest in audit['output_sha256'].items():assert sha(ROOT/relative)==digest,relative
    panels={}
    for symbol in NAMES:
        edges=np.asarray(load(OLD/f'fits/{symbol}_5min_raw.base.json')['bins']['x_edges'])
        for block in ['test','2026']:
            daily=pd.read_csv(FIG/f'eight_conditional_{symbol[:6]}_{block}_scale.csv',
                              dtype={'trading_date':str,'date_label':str},float_precision='round_trip')
            panels[(symbol,block)]={'daily':daily,'edges':edges}
    for block in ['test','2026']:write_plots(block,panels)
    changed=[]
    for relative,digest in list(audit['output_sha256'].items()):
        current=sha(ROOT/relative)
        if current!=digest:
            assert Path(relative).suffix=='.tex',relative
            changed.append(relative);audit['output_sha256'][relative]=current
    assert len(changed)==4
    audit.setdefault('numerical_validation_script_sha256',old_script_sha)
    audit['script_sha256']=sha(Path(__file__))
    audit.setdefault('style_revisions',[]).append({'previous_audit_sha256':old_audit_sha,
            'previous_script_sha256':old_script_sha,'changed_outputs':changed,
            'vertical_sep_cm':1.45,'reason':'Separate each row date ticks from the next row titles.',
            'numerical_checks_reused':True,'all_CSV_and_frozen_sources_verified_unchanged':True})
    dump(audit_path,audit)
    print('RESTYLE PASSED',len(changed),'TeX files; numerical evidence and CSV unchanged',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--restyle',action='store_true');args=parser.parse_args()
    restyle() if args.restyle else main()
