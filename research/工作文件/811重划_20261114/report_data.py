"""Generate article data displays and the complete stock-catalog supplement."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'20261114_811重划版'
FIGS=REPORT/'figs'
DATA=ROOT/'依据/811重划实验_20261114/data'
STATS=ROOT/'依据/811重划实验_20261114/report_data'
NAMES={'601318.SH':'中国平安','600519.SH':'贵州茅台','600887.SH':'伊利股份','600030.SH':'中信证券',
       '603799.SH':'华友钴业','600516.SH':'方大炭素','601668.SH':'中国建筑','600036.SH':'招商银行'}


def nfmt(n): return f'{int(n):,}'.replace(',',r'\,')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for p in [REPORT,FIGS,STATS]:p.mkdir(parents=True,exist_ok=True)
    summary=json.loads((DATA/'selection_summary.json').read_text('utf-8'))
    frozen=json.loads((DATA/'final_manifest.json').read_text('utf-8'))
    split=json.loads((DATA/'split_manifest.json').read_text('utf-8'))
    assert summary['status']==frozen['status']=='passed'
    assert list(summary['funnel'].values())==[1573,1225,952,158,8]
    expected_hashes={str(Path(r['path'])):r['sha256'] for r in frozen['outputs']}
    assert len(split['days'])==731

    u=pd.read_csv(DATA/'all_stocks.csv',dtype={'symbol':str,'first_date':str,'last_date':str})
    selected=u[u.selected].sort_values('eligible_rank')
    u['presence_pct']=100*u.presence_rate;u['complete_pct']=100*u.complete_share
    u['amount_yi']=u.median_daily_amount_cny/1e8
    u[['presence_pct','complete_pct']].to_csv(FIGS/'universe_all.csv',index=False)
    u[u.eligible].sort_values('eligible_rank')[['presence_pct','complete_pct','eligible_rank','amount_yi']].to_csv(FIGS/'universe_eligible.csv',index=False)
    u[u.selected].sort_values('eligible_rank')[['presence_pct','complete_pct','eligible_rank','amount_yi']].to_csv(FIGS/'universe_selected.csv',index=False)
    (FIGS/'universe_selection.tex').write_text(r'''\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.5cm},width=.455\linewidth,height=6.2cm,
  grid=major,grid style={gray!15},tick label style={font=\small},label style={font=\small},title style={font=\small}]
\nextgroupplot[xmin=-2,xmax=102,ymin=-2,ymax=102,xlabel={训练期有记录日比例（\%）},ylabel={有记录日中的完整日比例（\%）},title={(a) 全部 1573 只股票}]
\addplot[only marks,mark=*,mark size=.65pt,gray!55] table[x=presence_pct,y=complete_pct,col sep=comma]{universe_all.csv};
\addplot[only marks,mark=*,mark size=1pt,blue!65] table[x=presence_pct,y=complete_pct,col sep=comma]{universe_eligible.csv};
\addplot[only marks,mark=star,mark size=2pt,orange!90!black] table[x=presence_pct,y=complete_pct,col sep=comma]{universe_selected.csv};
\draw[dashed,black!65] (axis cs:95,0)--(axis cs:95,100);
\draw[dashed,black!65] (axis cs:0,90)--(axis cs:100,90);
\nextgroupplot[xmin=1,xmax=158,ymode=log,xlabel={合格候选的成交额排名},ylabel={日成交额中位数（亿元）},title={(b) 158 只合格候选的排序}]
\addplot[thick,blue!65,no marks] table[x=eligible_rank,y=amount_yi,col sep=comma]{universe_eligible.csv};
\addplot[only marks,mark=*,mark size=1.7pt,orange!90!black] table[x=eligible_rank,y=amount_yi,col sep=comma]{universe_selected.csv};
\draw[dashed,black!65] (axis cs:8,.01)--(axis cs:8,100);
\end{groupplot}
\end{tikzpicture}
''',encoding='utf-8')

    counts=[];stats=[];curve_rows=[];input_hashes={};variance_errors=[]
    all_old_days=split['days']
    old_x={d:i for i,d in enumerate(all_old_days)}
    new_dates=sorted(pd.read_csv(DATA/'oos2026/returns_601318.SH_5min.csv.gz',usecols=['trading_date'],dtype=str).trading_date.unique())
    new_x={d:770+i for i,d in enumerate(new_dates)}
    for sym in selected.symbol:
        historical=DATA/f'historical/returns_{sym}_5min.csv.gz'
        later=DATA/f'oos2026/returns_{sym}_5min.csv.gz'
        assert sha(historical)==expected_hashes[str(historical)]
        assert sha(later)==expected_hashes[str(later)]
        a=pd.read_csv(historical,dtype={'trading_date':str},float_precision='round_trip')
        b=pd.read_csv(later,dtype={'trading_date':str},float_precision='round_trip')
        input_hashes[str(historical.relative_to(ROOT))]=sha(historical)
        input_hashes[str(later.relative_to(ROOT))]=sha(later)
        row={'symbol':sym,'name':NAMES[sym]}
        for block in ['train','valid','test']:
            v=a[a.block.eq(block)];row[block+'_n']=len(v);row[block+'_days']=v.trading_date.nunique()
        row['oos_n']=len(b);row['oos_days']=b.trading_date.nunique()
        for label,part in [('train',a[a.block.eq('train')]),('valid',a[a.block.eq('valid')]),('test',a[a.block.eq('test')]),('oos',b)]:
            row[label+'_complete_days']=int(part.groupby('trading_date').size().eq(46).sum())
            row[label+'_meets_original_gate']=row[label+'_complete_days']>=60 and len(part)>=5000
        train=a[a.block.eq('train')].raw_log_return
        row['training_sd_pct']=100*train.std(ddof=1);row['training_zero_pct']=100*train.eq(0).mean()
        counts.append(row)
        for period,d in [('2017--2019',a),('2026',b)]:
            daily=d.groupby('trading_date').raw_log_return.agg(['size','std'])
            for day, values in d.groupby('trading_date').raw_log_return:
                if len(values)>=2:
                    v=values.to_numpy()
                    direct=float(np.sqrt(np.sum((v-v.mean())**2)/(len(v)-1)))
                    variance_errors.append(abs(direct-float(daily.loc[day,'std'])))
            for day,r in daily.iterrows():
                if int(r['size'])>=2:
                    curve_rows.append(dict(symbol=sym,date=day,period=period,x=(old_x if period=='2017--2019' else new_x)[day],vol_pct=100*float(r['std']),n=int(r['size'])))
        for block,d in [('train',a[a.block.eq('train')]),('valid',a[a.block.eq('valid')]),('test',a[a.block.eq('test')]),('2026',b)]:
            v=d.raw_log_return.to_numpy()
            stats.append(dict(symbol=sym,name=NAMES[sym],block=block,n=len(v),days=d.trading_date.nunique(),
                mean_pct=100*float(v.mean()),sd_pct=100*float(v.std(ddof=1)),zero_pct=100*float((v==0).mean()),
                min_pct=100*float(v.min()),max_pct=100*float(v.max())))
    counts=pd.DataFrame(counts);counts.to_csv(STATS/'eight_stock_counts.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(stats).to_csv(STATS/'eight_stock_return_statistics.csv',index=False,encoding='utf-8-sig')
    curves=pd.DataFrame(curve_rows);curves.to_csv(STATS/'eight_stock_daily_volatility.csv',index=False)
    for sym in selected.symbol:
        for period,short in [('2017--2019','old'),('2026','new')]:
            part=curves[(curves.symbol==sym)&(curves.period==period)]
            # NaN rows preserve missing-date breaks; no interpolation or synthetic observations.
            full_x=range(731) if short=='old' else range(770,930)
            part=part.set_index('x').reindex(full_x).reset_index()
            part[['x','vol_pct']].to_csv(FIGS/f'panel_vol_{sym[:6]}_{short}.csv',index=False,na_rep='nan')
    panel=[r'''\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 4,horizontal sep=1.7cm,vertical sep=1.25cm},
width=.44\linewidth,height=4.3cm,xmin=0,xmax=929,ymin=0,unbounded coords=jump,
xtick={0,244,487,770},xticklabels={2017,2018,2019,2026},
tick label style={font=\scriptsize},title style={font=\small},label style={font=\small},
grid=major,grid style={gray!15},ylabel={日内标准差（\%）}]''']
    for sym in selected.symbol:
        panel.append(r'\nextgroupplot[title={'+NAMES[sym]+' '+sym+r'}]')
        panel.append(r'\addplot[blue!75,line width=.45pt,no marks] table[x=x,y=vol_pct,col sep=comma]{panel_vol_'+sym[:6]+'_old.csv};')
        panel.append(r'\addplot[orange!85!black,line width=.55pt,no marks] table[x=x,y=vol_pct,col sep=comma]{panel_vol_'+sym[:6]+'_new.csv};')
    panel.append(r'\end{groupplot}\end{tikzpicture}')
    (FIGS/'eight_stock_data.tex').write_text('\n'.join(panel),encoding='utf-8')

    selected_rows=[];count_rows=[]
    for _,r in selected.iterrows():
        selected_rows.append(f"{int(r.eligible_rank)} & {NAMES[r.symbol]} & {r.symbol} & {int(r.training_days_present)} & {100*r.presence_rate:.2f} & {100*r.complete_share:.2f} & {r.median_daily_amount_cny/1e8:.2f}"+r'\\')
    for _,r in counts.iterrows():
        cells=[f"{nfmt(r[b+'_n'])}/{int(r[b+'_days'])}" for b in ['train','valid','test','oos']]
        count_rows.append(r['name']+' & '+' & '.join(cells)+f" & {r.training_sd_pct:.4f} & {r.training_zero_pct:.2f}"+r'\\')
    selected_table=r'''\begin{table}[!htbp]\centering\small
\caption{入选的八只股票。排序只依据训练期日成交额中位数；完整日比例的分母为有记录日数，覆盖率的分母为584个名义训练日。}
\label{tab:selected_stocks}
\begin{tabular}{rlcrrrr}\toprule
排名 & 股票 & 证券代码 & 有记录日 & 覆盖率(\%) & 完整日(\%) & 成交额(亿元)\\\midrule
'''+ '\n'.join(selected_rows)+r'''
\bottomrule\end{tabular}\end{table}
'''
    count_table=r'''\begin{table}[!htbp]\centering\small
\caption{八股有效收益样本与训练期特征。各样本列为“收益条数/交易日数”；标准差以原始对数收益的百分数表示，零收益比例未作平滑。}
\label{tab:eight_counts}
\begin{tabular}{lrrrrrr}\toprule
股票 & 训练 & 验证 & 测试 & 2026年 & 标准差(\%) & 零收益(\%)\\\midrule
'''+ '\n'.join(count_rows)+r'''
\bottomrule\end{tabular}\end{table}
'''
    (FIGS/'selected_stocks_table.tex').write_text(selected_table,encoding='utf-8')
    (FIGS/'eight_counts_table.tex').write_text(count_table,encoding='utf-8')
    universe_text=r'''\subsection{数据范围与样本选择}\label{sec:data}
\subsubsection{完整股票范围与筛选规则}
初始分钟资料覆盖2017年1月3日至2019年12月31日的731个沪市交易日，共1573只股票、
996\,288个有记录的“股票—交易日”。2017、2018和2019年分别有1387、1447和1568只股票。
各股票的可用日期不同，1573只股票并非每天都有记录；逐股覆盖与筛选结果见补充材料\cite{supplement}。

筛选只使用前584个名义训练日，即2017年1月3日至2019年5月29日。
有记录日比例不低于80\%的股票有1225只，将门槛提高到95\%后剩952只；
再要求完整日比例不低于90\%、日成交额中位数为正，得到158只候选。
按训练期日成交额中位数降序取前8只，中国平安排名第一，作为主例。
缺记录日的成交额按零计入584日的中位数。80\%初筛不会排除满足最终95\%门槛的股票。

图\ref{fig:universe}(a)每个点代表一只股票。横轴为“有记录日数$/584\times100\%$”，
衡量缺了多少交易日；纵轴为“完整日数$/$有记录日数$\times100\%$”，衡量日内分钟记录是否齐全。
没有训练期记录的股票置于原点，仅用来标记缺失。完整日要求09:30--11:29、
13:00--14:56共237个核心分钟标签齐备，且15:00收盘记录存在。
这一口径考虑了收盘前部分分钟标签的系统性缺失，不能简单以每天的总行数代替。
上述统计共覆盖774\,275个训练期股票日、175\,980\,062条分钟记录。

\begin{figure}[!htbp]\centering\input{figs/universe_selection.tex}
\caption{完整股票范围与样本筛选。(a) 灰点为全部1573只股票，蓝点为158只合格候选，橙色为最终8只；重合点可能遮挡。
虚线为95\%覆盖率和90\%完整日门槛。(b) 合格候选按训练期成交额中位数排序，橙色为前8名，纵轴取对数。}
\label{fig:universe}\end{figure}
\input{figs/selected_stocks_table.tex}
\FloatBarrier

\subsubsection{样本划分与观测特征}
731个交易日按时间顺序以80\%、10\%、10\%划分训练、验证与测试段，
按累计比例向下取整得到584、73、74日。沿用每段末尾剔除5个交易日的规则，
实际保留579、68、69日，各股票本身缺失的日期不补值。保留训练段为2017年1月3日至2019年5月22日，
验证段为2019年5月30日至9月3日，测试段为2019年9月11日至12月24日。
选股使用584个名义训练日，归一化尺度、观测窗口及模型参数只用保留的训练收益估计。

价格按上午、下午分别聚合为5分钟区间，每个区间要求至少80\%的必需分钟标签，
15:00记录并入末个区间。只计算同一时段内相邻区间的收盘收益，完整日得到46个收益。
样本量门槛中的完整日按46条有效5分钟收益计数，与选股阶段的分钟标签完整性分别核对。
新验证、测试段通常只有3128和3174条收益，均不足既定的“60个完整日且5000条收益”门槛，
以下将其作为短留出段检查。

2026年1月5日至8月31日为后续年份检验。7股各有160日、7360条收益，满足上述门槛；
华友钴业仅有6月30日至8月31日的45日、2070条收益，单列为短样本。
2020年下半年的分钟资料仅覆盖所选八股中的两只，2021--2025年无可用分钟记录，
不将这些年份补齐或拼接成连续样本；日频资料也不替代缺失的分钟数据。
本次8:1:1划分是在查看已有结果后重新设定的回顾性分析，不是新的盲测。
后续行情不参与本次选股、尺度估计或参数重估。

\input{figs/eight_counts_table.tex}
中国平安训练期日成交额中位数最高，且584日均符合完整性条件，适合展示计算过程。
成交活跃度与完整性筛选也使样本偏向交易较活跃的股票；八股比较用于检查个股差异，
不将结果推广为全部A股的共同规律。图\ref{fig:panel_data}展示八股在全部可用日期上的波动变化。
\clearpage
\begin{figure}[p]\centering\input{figs/eight_stock_data.tex}
\caption{八只股票的真实收益波动。每点为当日全部有效5分钟收益的样本标准差。
蓝色为2017--2019年，橙色为2026年；两段之间留空，缺失年份不插值。
横轴按各段交易日顺序排列，压缩两段之间的日历间隔；纵轴独立取值以保留各股变化。
图中包括划分边界处仅用于展示的日期，这些日期不进入训练或留出评分。}
\label{fig:panel_data}\end{figure}\clearpage
'''
    (FIGS/'data_section.tex').write_text(universe_text,encoding='utf-8')

    # The supplemental catalog contains every stock, including exclusions; it is not extra narrative.
    suppl=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.8cm]{geometry}\usepackage[fontset=fandol]{ctex}
\usepackage{booktabs,longtable,array,amsmath,hyperref}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\renewcommand{\thetable}{S\arabic{table}}
\begin{document}
\begin{center}\Large\bfseries 数据补充材料：完整股票范围与样本筛选\end{center}
\noindent 本材料配合《A股收益的分层波动建模：层数可辨识性与样本外检验》使用。
统计对象为2017--2019年分钟数据中的全部1573只沪市股票；逐股表包含未通过筛选的股票，
并不只列最终入选者。本材料采用8:1:1的新划分，选股统计与主稿实验均据此重算。
本次划分为查看已有结果后的回顾性重划，不是新的盲测。

\section*{统计定义}
日历由731个有行情记录的交易日组成。筛选窗口为前584个名义训练日，即2017-01-03至2019-05-29。
覆盖率为有记录日数除以584；完整日比例为完整日数除以有记录日数。完整日要求237个核心
分钟标签及15:00记录齐备。日成交额由分钟成交额相加，缺记录日按零计入训练段中位数。
没有训练期记录的股票，其覆盖率与成交额中位数为零，完整日比例在逐股表记为“--”。
主稿样本量门槛中的完整日按46条有效5分钟收益计数，与本表采用的分钟标签完整性分别核对。

股票全集1573只，经80\%覆盖率初筛剩1225只，95\%覆盖率筛选剩952只；结合90\%完整日
门槛与正成交额条件，得到158只合格候选，再按成交额中位数取前8只。
代码唯一标识证券；合格排名仅在158只候选内计算。筛选状态依次为：
覆盖不足（未达95\%）、完整性不足、成交额不合格、合格未入选、入选。

\input{figs/selected_stocks_table.tex}
\input{figs/eight_counts_table.tex}
\clearpage\section*{全部股票的覆盖与筛选结果}
\scriptsize
\begin{longtable}{lrrr r r r l}
\caption{全部1573只股票，按证券代码排列。全期日数统计2017--2019年有记录日；其余指标只用584日名义训练筛选窗口。
成交额单位为亿元，完整比例单位为\%。}\label{tab:full_universe}\\
\toprule 证券代码 & 全期日数 & 训练日数 & 完整日数 & 完整比例 & 成交额 & 合格排名 & 筛选状态\\\midrule\endfirsthead
\multicolumn{8}{l}{表S3（续）}\\\toprule 证券代码 & 全期日数 & 训练日数 & 完整日数 & 完整比例 & 成交额 & 合格排名 & 筛选状态\\\midrule\endhead
\midrule\multicolumn{8}{r}{续下页}\endfoot\bottomrule\endlastfoot
'''
    cat=[]
    for _,r in u.sort_values('symbol').iterrows():
        complete=f'{r.complete_pct:.2f}' if r.training_days_present else '--'
        rank=str(int(r.eligible_rank)) if pd.notna(r.eligible_rank) else '--'
        state='入选' if r.selected else '合格未入选' if r.eligible else '覆盖不足' if not r.presence_pass else '完整性不足' if r.complete_share<.9 else '成交额不合格'
        cat.append(f"{r.symbol} & {r.observed_days_2017_2019} & {r.training_days_present} & {r.complete_days} & {complete} & {r.amount_yi:.4f} & {rank} & {state}"+r'\\')
    suppl+='\n'.join(cat)+r'\end{longtable}\normalsize\end{document}'
    (REPORT/'data_supplement.tex').write_text(suppl,encoding='utf-8')
    assert len(u)==u.symbol.nunique()==1573
    assert set(u.symbol)==set(pd.read_csv(DATA/'all_stocks.csv').symbol)
    assert not counts.valid_meets_original_gate.any()
    assert not counts.test_meets_original_gate.any()
    assert int(counts.oos_meets_original_gate.sum())==7
    for name in ['all_stocks.csv','selected_8.csv','selection_summary.json','split_manifest.json','final_manifest.json']:
        input_hashes[str((DATA/name).relative_to(ROOT))]=sha(DATA/name)
    # Independent variance recomputation for every plotted stock-day.
    assert len(counts)==8 and int(counts.oos_n.sum())==53590
    assert max(variance_errors)<1e-14
    audit={'status':'passed','all_stocks':len(u),'eligible_stocks':int(u.eligible.sum()),'selected_stocks':len(selected),
           'plotted_stock_days':len(curves),'oos_returns':int(counts.oos_n.sum()),'input_sha256':input_hashes,
           'daily_volatility_definition':'sample standard deviation of all available within-session 5-minute raw returns in each day, ddof=1',
           'daily_sd_independent_recalculation_max_error':max(variance_errors),
           'nominal_days':[584,73,74],'retained_days':[579,68,69],
           'selection_training_day_denominator':584,
           'retrospective_resplit_not_blind_test':True,
           'original_gate':{'minimum_complete_days':60,'minimum_returns':5000},
           'gate_qualified_stocks':{b:int(counts[b+'_meets_original_gate'].sum()) for b in ['train','valid','test','oos']},
           'full_catalog_unique_symbols':int(u.symbol.nunique()),
           'output_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [REPORT/'data_supplement.tex',*sorted(FIGS.glob('universe_*')),*sorted(FIGS.glob('panel_vol_*')),FIGS/'selected_stocks_table.tex',FIGS/'eight_counts_table.tex',FIGS/'eight_stock_data.tex',FIGS/'data_section.tex']},
           'missing_dates_interpolated':False,'raw_data_modified':False}
    (STATS/'display_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['status','all_stocks','eligible_stocks','selected_stocks','plotted_stock_days','oos_returns']},ensure_ascii=False))


if __name__=='__main__':main()
