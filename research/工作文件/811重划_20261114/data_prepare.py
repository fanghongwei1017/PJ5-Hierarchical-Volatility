"""Rebuild training-only selection and raw returns for an 80/10/10 chronological split."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'依据/811重划实验_20261114/data'
OLD=ROOT/'依据/完整股票数据_20261114'
OOS=ROOT/'依据/2026时间外验证'
REPO=Path('E:/Users/lenovo/Downloads/PJ5-research-main/PJ5-research-main/h-theory-reproduction')
RAW=Path('E:/data/A股分钟数据')
sys.path.insert(0,str(OOS/'runtime'))
from pj5_h.data.bars import load_day_symbol
from pj5_h.data.build import build_returns
from pj5_h.data.sessions import SESSION_RULES_ID
CORE={h*10000000+m*100000 for h,m in
      [(9+i//60,i%60) for i in range(30,150)]+[(13+i//60,i%60) for i in range(117)]}
assert len(CORE)==237

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def read(p):return json.loads(p.read_text('utf-8'))

def scan_day(job):
    day,names=job
    rows=[]
    for name in names:
        path=RAW/'sh'/day/name
        raw=path.read_bytes()
        labels=set();amounts=[];n=0
        for line in raw.splitlines()[1:]:
            if not line.strip():continue
            f=line.split(b',')
            assert len(f)==10 and f[0].decode()==name[6:-4] and f[2].decode()==day,path
            labels.add(int(f[3]));amt=float(f[9]);assert math.isfinite(amt),path
            amounts.append(amt);n+=1
        rows.append([name[6:-4],day,n,int(CORE.issubset(labels)),int(150000000 in labels),
                     math.fsum(amounts),len(raw),hashlib.sha256(raw).hexdigest()])
    return rows

def selection():
    OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    old_summary=read(OLD/'summary.json');old_scan=OLD/'training_stock_days.csv.gz'
    old_hash=next(v for k,v in old_summary['sources_sha256'].items() if k.endswith('training_stock_days.csv.gz'))
    assert sha(old_scan)==old_hash
    folders={p.name:sorted(e.name for e in os.scandir(p) if e.is_file() and e.name.startswith('kline_') and e.name.endswith('.SH.csv'))
             for p in sorted((RAW/'sh').iterdir()) if p.is_dir() and p.name.isdigit() and '20170103'<=p.name<='20191231'}
    folders={d:names for d,names in folders.items() if names}
    days=sorted(folders);assert days==read(OLD/'directory_inventory.json')['days']
    assert len(days)==731
    bounds=[0,int(.8*len(days)),int(.9*len(days)),len(days)]
    assert bounds==[0,584,657,731]
    blocks={};groups={}
    for i,label in enumerate(['train','valid','test']):
        nominal=days[bounds[i]:bounds[i+1]];retained=nominal[:-5];purged=nominal[-5:]
        groups[label]=dict(nominal=nominal,retained=retained,purged=purged,
                           nominal_days=len(nominal),retained_days=len(retained),purged_days=len(purged))
        blocks.update({d:label for d in retained});blocks.update({d:'purge' for d in purged})
    split=dict(ratio=[.8,.1,.1],days=days,day_blocks=blocks,groups=groups,
               training_selection_days=days[:584],selection_includes_nominal_train_purge_days=True,
               fit_must_use_retained_training_only=True,
               original_date_order_unchanged=True,retrospective=True)
    dump(OUT/'split_manifest.json',split)
    newdays=days[438:584]
    dest=OUT/'additional_training_stock_days.csv.gz'
    if not dest.exists():
        partial=dest.with_name('additional_training_stock_days.partial.csv.gz')
        with gzip.open(partial,'wt',encoding='utf-8',newline='') as handle:
            w=csv.writer(handle);w.writerow(['symbol','trading_date','n_rows','core_ok','has_terminal','amount','bytes','sha256'])
            with ProcessPoolExecutor(max_workers=6) as pool:
                for i,rows in enumerate(pool.map(scan_day,[(d,folders[d]) for d in newdays],chunksize=2),1):
                    w.writerows(rows)
                    if i%10==0 or i==len(newdays):print(f'SCANNED added {i}/{len(newdays)} days {time.monotonic()-start:.1f}s',flush=True)
        partial.replace(dest)
    existing=pd.read_csv(old_scan,dtype={'symbol':str,'trading_date':str},float_precision='round_trip')
    added=pd.read_csv(dest,dtype={'symbol':str,'trading_date':str},float_precision='round_trip')
    assert set(existing.trading_date)==set(days[:438])
    assert set(added.trading_date)==set(newdays)
    assert len(added)==sum(len(folders[d]) for d in newdays)
    assert not added.duplicated(['symbol','trading_date']).any()
    scans=pd.concat([existing,added.drop(columns='sha256')],ignore_index=True)
    assert not scans.duplicated(['symbol','trading_date']).any()
    assert len(scans)==sum(len(folders[d]) for d in days[:584])
    scans.to_csv(OUT/'training_stock_days.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    inventory=read(OLD/'directory_inventory.json')
    stocks=inventory['stocks'];grouped=dict(tuple(scans.groupby('symbol')))
    rows=[]
    for sym in sorted(stocks):
        s=grouped.get(sym)
        n=0 if s is None else len(s)
        complete=0 if s is None else int((s.core_ok.eq(1)&s.has_terminal.eq(1)).sum())
        core=0 if s is None else int(s.core_ok.sum())
        terminal=0 if s is None else int(s.has_terminal.eq(0).sum())
        amount=0. if s is None else float(np.median(np.r_[s.amount.to_numpy(),np.zeros(584-n)]))
        presence=n/584;share=complete/n if n else 0.
        rows.append(dict(symbol=sym,first_date=stocks[sym]['first'],last_date=stocks[sym]['last'],
            observed_days_2017_2019=stocks[sym]['days'],training_days_present=n,presence_rate=presence,
            complete_days=complete,complete_share=share,core_only_days=core,core_only_share=core/n if n else 0.,
            missing_terminal_days=terminal,median_daily_amount_cny=amount,prefilter=presence>=.8,
            presence_pass=presence>=.95,eligible=presence>=.95 and share>=.9 and amount>0))
    u=pd.DataFrame(rows);assert len(u)==1573
    c=u[u.eligible].sort_values(['median_daily_amount_cny','symbol'],ascending=[False,True])
    ranks={s:i+1 for i,s in enumerate(c.symbol)}
    u['eligible_rank']=u.symbol.map(ranks).astype('Int64');u['selected']=u.eligible_rank.le(8).fillna(False)
    u['screening_result']=np.select([~u.prefilter,~u.presence_pass,u.complete_share.lt(.9),u.median_daily_amount_cny.le(0),u.selected],
        ['training coverage <80%','training coverage <95%','complete-day share <90%','nonpositive turnover','selected'],default='eligible, rank >8')
    c=u[u.eligible].sort_values('eligible_rank');selected=u[u.selected].sort_values('eligible_rank')
    for name,df in [('all_stocks.csv',u),('eligible.csv',c),('selected_8.csv',selected)]:
        df.to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.17g')
    # Reconcile additional days against older stored daily aggregates where available.
    archive=pd.read_csv(REPO/'data_catalog/coverage_by_symbol_day.csv.gz',dtype={'symbol':str,'trading_date':str},float_precision='round_trip')
    archive=archive[archive.trading_date.isin(newdays)&archive.n_bars.gt(0)]
    matched=archive.merge(added,on=['symbol','trading_date'],validate='one_to_one')
    assert len(matched)==len(archive)
    assert (matched.n_rows==matched.n_bars).all()
    assert np.allclose(matched.amount_x,matched.amount_y,rtol=1e-12,atol=1e-4)
    maxdiff=float(np.max(np.abs(matched.amount_x-matched.amount_y)))
    summary=dict(status='passed',source_root=str(RAW/'sh'),raw_files_unchanged=True,n_stocks=len(u),n_dates=len(days),
        stock_day_files=sum(map(len,folders.values())),n_training_days=584,training_dates=[days[0],days[583]],
        reused_training_days=438,added_training_days=146,reused_training_files=len(existing),additional_training_files=len(added),
        training_files_scanned=len(scans),training_rows_scanned=int(scans.n_rows.sum()),
        additional_training_rows_scanned=int(added.n_rows.sum()),
        additional_archived_rows_reconciled=len(matched),additional_turnover_max_abs_difference=maxdiff,
        funnel=dict(available_2017_2019=len(u),training_presence_80=int(u.prefilter.sum()),
            training_presence_95=int(u.presence_pass.sum()),complete_90_and_positive_amount=len(c),selected=len(selected)),
        missing_terminal_days=int(scans.has_terminal.eq(0).sum()),
        selected_symbols=selected.symbol.tolist(),old_selected_symbols=read(OLD/'summary.json')['selected'],
        input_sha256={str(old_scan):old_hash,str(dest):sha(dest),str(REPO/'data_catalog/coverage_by_symbol_day.csv.gz'):sha(REPO/'data_catalog/coverage_by_symbol_day.csv.gz')},
        elapsed_seconds=time.monotonic()-start)
    summary['old_selected_symbols']=[r['symbol'] for r in summary['old_selected_symbols']]
    dump(OUT/'selection_summary.json',summary)
    print('SELECTION_READY '+json.dumps({k:summary[k] for k in ['funnel','selected_symbols','additional_training_files','elapsed_seconds']},ensure_ascii=False),flush=True)
    return split,selected.symbol.tolist()

def read_returns(path):
    return pd.read_csv(path,dtype={'trading_date':str,'symbol':str},float_precision='round_trip')

def write_returns(df,dest):
    partial=dest.with_name(dest.name.replace('.csv.gz','.partial.csv.gz'))
    df.to_csv(partial,index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    partial.replace(dest)

def raw_frame(df,blocks,oos=False):
    drop=[c for c in df if c.startswith('log_score_') or c in ['normalized_return','deseasonalized_log_return','seasonality_scale']]
    df=df.drop(columns=drop).copy()
    df['trading_date']=df.trading_date.astype(str)
    df['log_return']=df.raw_log_return.to_numpy()
    df['block']='oos2026' if oos else df.trading_date.map(blocks)
    assert df.block.notna().all()
    return df

def validate_returns(df,sym,k):
    assert df.symbol.eq(sym).all()
    assert np.isfinite(df.raw_log_return).all()
    assert not df.cross_session.astype(bool).any()
    assert not df.duplicated(['trading_date','session_id','bar_start']).any()
    starts=pd.to_datetime(df.bar_start);ends=pd.to_datetime(df.bar_end)
    assert (ends-starts).dt.total_seconds().eq(60*k).all()
    mins=starts.dt.hour*60+starts.dt.minute
    sid=df.session_id.astype(int)
    assert ((sid.eq(1)&mins.ge(570+k)&mins.lt(690))|(sid.eq(2)&mins.ge(780+k)&mins.lt(900))).all()
    assert (df.groupby(['trading_date','session_id']).size()<=120//k-1).all()
    err=float(np.max(np.abs(np.log(df.close.to_numpy()/df.prev_close.to_numpy())-df.raw_log_return.to_numpy()))) if len(df) else 0.
    assert err<1e-12
    return err

def rebuild(sym,days,freqs):
    bars=[];sources=[];diagnostics=[]
    for day in days:
        path=RAW/'sh'/day/f'kline_{sym}.csv'
        if not path.exists():
            diagnostics.append(dict(symbol=sym,date=day,status='missing_file'));continue
        raw=path.read_bytes()
        identity=[line.split(b',') for line in raw.splitlines()[1:] if line.strip()]
        assert all(row[0].decode()==sym and row[2].decode()==day for row in identity),path
        frame=load_day_symbol(sym,day,root=str(RAW))
        diagnostics.append(dict(symbol=sym,date=day,**frame.attrs['diag']))
        sources.append(dict(symbol=sym,date=day,path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        frame.attrs.clear()
        if len(frame):bars.append(frame)
    if not bars:raise ValueError(f'No raw rows for {sym}')
    allbars=pd.concat(bars,ignore_index=True)
    frames={}
    for k in freqs:
        r,e=build_returns(allbars,SESSION_RULES_ID,k)
        gap=r.attrs.pop('gaps');r.attrs.clear()
        frames[k]=(r,e,gap)
    return frames,sources,diagnostics

def prepare_symbol(job):
    sym,split=job
    target=OUT/'historical';target.mkdir(exist_ok=True)
    oostarget=OUT/'oos2026';oostarget.mkdir(exist_ok=True)
    checks=[];inputs={};counts=[];source_rows=[];diag_rows=[]
    cached={}
    for k in [1,5,15]:
        preferred=OOS/'historical_inputs'/f'returns_{sym}_{k}min.csv.gz'
        alternate=REPO/'derived/minute'/f'returns_{sym}_{k}min.csv.gz'
        p=preferred if preferred.exists() else alternate
        if p.exists():cached[k]=p
    need=[k for k in [1,5,15] if k not in cached]
    built={}
    # Price reconstruction at historical interval endpoints verifies cached raw returns.
    probes=sorted(set([split['days'][i] for i in [0,437,438,578,579,583,584,651,652,656,657,725,726,730]]))
    for k in [5,1,15]:
        if k in cached:
            p=cached[k];df=read_returns(p);inputs[str(p)]=sha(p)
            for day in probes:
                expected=df[df.trading_date.eq(day)].sort_values(['session_id','bar_start'])
                if not len(expected):continue
                bars=load_day_symbol(sym,day,root=str(RAW))
                actual,_=build_returns(bars,SESSION_RULES_ID,k)
                actual=actual.sort_values(['session_id','bar_start'])
                assert len(actual)==len(expected),(sym,k,day,len(actual),len(expected))
                assert pd.to_datetime(actual.bar_start).tolist()==pd.to_datetime(expected.bar_start).tolist()
                er=float(np.max(np.abs(actual.raw_log_return.to_numpy()-expected.raw_log_return.to_numpy())))
                assert er<1e-12,(sym,k,day,er)
                checks.append(dict(symbol=sym,frequency=k,date=day,n=len(actual),max_reconstruction_error=er))
        else:
            if k not in built:
                requested=[k] if k==5 else [f for f in need if f not in built]
                fresh,sources,diag=rebuild(sym,split['days'],requested)
                built.update(fresh);source_rows.extend(sources);diag_rows.extend(diag)
            df,e,gap=built[k]
            e.to_csv(target/f'exclusions_{sym}_{k}min.csv',index=False)
            gap.to_csv(target/f'gaps_{sym}_{k}min.csv',index=False)
        df=raw_frame(df,split['day_blocks']);err=validate_returns(df,sym,k)
        assert set(df.trading_date).issubset(set(split['days']))
        dest=target/f'returns_{sym}_{k}min.csv.gz'
        write_returns(df,dest)
        rows={label:dict(n=int((df.block==label).sum()),days=int(df.loc[df.block.eq(label),'trading_date'].nunique()),
             first=df.loc[df.block.eq(label),'trading_date'].min(),last=df.loc[df.block.eq(label),'trading_date'].max())
             for label in ['train','valid','test','purge']}
        counts.append(dict(symbol=sym,frequency=k,period='historical',n=len(df),days=df.trading_date.nunique(),blocks=rows,
            train_sd=float(df.loc[df.block.eq('train'),'raw_log_return'].std(ddof=1)),price_error=err,
            source='cached_raw_only' if k in cached else 'rebuilt_from_raw',path=str(dest),sha256=sha(dest)))
        print(f'RETURNS_READY {sym} {k}min {rows}',flush=True)
    newdays=sorted(p.name for p in (RAW/'sh').iterdir() if p.is_dir() and p.name.isdigit() and '20260105'<=p.name<='20260831' and (p/'kline_601318.SH.csv').is_file())
    assert len(newdays)==160
    cached_oos=OOS/f'returns_and_scores_{sym}.csv.gz'
    if cached_oos.exists():
        df=read_returns(cached_oos);inputs[str(cached_oos)]=sha(cached_oos)
        avail=sorted(df.trading_date.unique())
        for day in [avail[0],avail[len(avail)//2],avail[-1]]:
            bars=load_day_symbol(sym,day,root=str(RAW));actual,_=build_returns(bars,SESSION_RULES_ID,5)
            expected=df[df.trading_date.eq(day)].sort_values(['session_id','bar_start'])
            actual=actual.sort_values(['session_id','bar_start'])
            assert len(actual)==len(expected)
            er=float(np.max(np.abs(actual.raw_log_return.to_numpy()-expected.raw_log_return.to_numpy())))
            assert er<1e-12
            checks.append(dict(symbol=sym,frequency=5,date=day,n=len(actual),max_reconstruction_error=er))
    else:
        frames,sources,diag=rebuild(sym,newdays,[5]);df,e,gap=frames[5]
        source_rows.extend(sources);diag_rows.extend(diag)
        e.to_csv(oostarget/f'exclusions_{sym}_5min.csv',index=False)
        gap.to_csv(oostarget/f'gaps_{sym}_5min.csv',index=False)
    df=raw_frame(df,{},True);err=validate_returns(df,sym,5)
    assert set(df.trading_date).issubset(newdays)
    dest=oostarget/f'returns_{sym}_5min.csv.gz'
    write_returns(df,dest)
    counts.append(dict(symbol=sym,frequency=5,period='oos2026',n=len(df),days=df.trading_date.nunique(),
        first=df.trading_date.min(),last=df.trading_date.max(),price_error=err,path=str(dest),sha256=sha(dest)))
    pd.DataFrame(source_rows).to_csv(OUT/f'raw_sources_{sym}.csv',index=False)
    pd.DataFrame(diag_rows).to_csv(OUT/f'raw_diagnostics_{sym}.csv',index=False)
    result=dict(symbol=sym,status='passed',raw_reconstruction_checks=checks,input_sha256=inputs,counts=counts)
    dump(OUT/f'return_checks_{sym}.json',result)
    print(f'SYMBOL_DONE {sym}',flush=True)
    return result

def verify_outputs():
    summary=read(OUT/'returns_summary.json');split=read(OUT/'split_manifest.json')
    old_checks=read(OLD/'display_audit.json')['input_sha256']
    old_checks={str((ROOT/rel).resolve()):digest for rel,digest in old_checks.items()}
    checked=0
    for path,digest in summary['input_sha256'].items():
        assert sha(Path(path))==digest,path
        if str(Path(path).resolve()) in old_checks:
            assert digest==old_checks[str(Path(path).resolve())],path
            checked+=1
    assert checked==16,checked
    outputs=[]
    for row in summary['counts']:
        path=Path(row['path']);assert sha(path)==row['sha256'],path
        frame=read_returns(path)
        assert len(frame)==row['n'] and frame.trading_date.nunique()==row['days'],path
        assert not any(c.startswith('log_score_') or c in ['normalized_return','deseasonalized_log_return','seasonality_scale'] for c in frame),path
        if row['period']=='historical':assert frame.block.equals(frame.trading_date.map(split['day_blocks']))
        else:assert frame.block.eq('oos2026').all()
        assert np.array_equal(frame.log_return.to_numpy(),frame.raw_log_return.to_numpy())
        validate_returns(frame,row['symbol'],row['frequency'])
        outputs.append({k:row[k] for k in ['symbol','frequency','period','n','days','path','sha256']})
    assert len(outputs)==32
    error=max(c['max_reconstruction_error'] for c in summary['raw_reconstruction_checks'])
    result=dict(status='passed',files=len(outputs),outputs=outputs,
        cached_5min_and_2026_hashes_matching_previous_audit=checked,
        raw_reconstruction_checks=len(summary['raw_reconstruction_checks']),
        raw_reconstruction_max_absolute_error=error,old_normalization_and_scores_removed=True,
        no_cross_session_returns=True,split_counts_nominal=[584,73,74],split_counts_retained=[579,68,69],
        selection_manifest_sha256=sha(OUT/'selection_summary.json'),split_manifest_sha256=sha(OUT/'split_manifest.json'),
        preparation_script_sha256=sha(Path(__file__)))
    dump(OUT/'final_manifest.json',result)
    print('VERIFIED '+json.dumps({k:result[k] for k in ['status','files','cached_5min_and_2026_hashes_matching_previous_audit','raw_reconstruction_checks','raw_reconstruction_max_absolute_error']}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selection-only',action='store_true');parser.add_argument('--returns-only',action='store_true');parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();start=time.monotonic()
    if args.verify_only:
        verify_outputs();return
    if args.returns_only:
        split=read(OUT/'split_manifest.json');symbols=read(OUT/'selection_summary.json')['selected_symbols']
    else:split,symbols=selection()
    if args.selection_only:return
    with ProcessPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(prepare_symbol,[(s,split) for s in symbols]))
    counts=[c for r in results for c in r['counts']]
    dump(OUT/'returns_summary.json',dict(status='passed',symbols=symbols,counts=counts,
        input_sha256={k:v for r in results for k,v in r['input_sha256'].items()},
        raw_reconstruction_checks=[c for r in results for c in r['raw_reconstruction_checks']],
        old_normalization_and_scores_removed=True,no_cross_session_returns=True,raw_files_modified=False,
        elapsed_seconds=time.monotonic()-start))
    print('DATA_COMPLETE '+json.dumps({'symbols':symbols,'files':len(counts),'seconds':time.monotonic()-start}),flush=True)
    verify_outputs()

if __name__=='__main__':main()

