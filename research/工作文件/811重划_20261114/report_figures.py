"""Rebuild report figures and compact metrics solely from the 8:1:1 run."""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import math
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / '依据/811重划实验_20261114'
FITS = RUN / 'fits'
DATA = RUN / 'data'
OUT = RUN / 'report_metrics'
FIG = ROOT / '20261114_811重划版/figs'
OLD = ROOT / '20261114要求核对版/figs'
PRIMARY = '601318.SH_5min_raw'
NAMES = {'601318.SH':'中国平安','600519.SH':'贵州茅台','600030.SH':'中信证券',
         '600887.SH':'伊利股份','600036.SH':'招商银行','603799.SH':'华友钴业',
         '600516.SH':'方大炭素','601668.SH':'中国建筑'}
sys.path.insert(0, str(FITS / 'runtime'))
engine = importlib.import_module('fit_engine')
from pj5_h.inference.fit import rolling_variance


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def tex(name, text):
    (FIG / name).write_text(text, encoding='utf-8')


def csv(name, frame):
    frame.to_csv(FIG / name, index=False, na_rep='nan', float_format='%.17g')
    check = pd.read_csv(FIG / name, float_precision='round_trip')
    for col in frame.select_dtypes(include='number'):
        np.testing.assert_allclose(check[col], frame[col], rtol=1e-14, atol=1e-15, equal_nan=True)


def checked(key, suffix):
    value = load(FITS / f'{key}.{suffix}.json')
    assert value['status'] == 'passed', (key, suffix)
    return value


def raw_primary(base):
    path = DATA / 'historical/returns_601318.SH_5min.csv.gz'
    assert sha(path) == base['input_sha256']['historical']
    frame = pd.read_csv(path, dtype={'trading_date': str}, float_precision='round_trip')
    assert float(frame.loc[frame.block.eq('train'), 'raw_log_return'].std(ddof=1)) == base['scale']
    return frame.sort_values(['trading_date', 'session_id', 'bar_end'])


def overview(base, frame):
    train = frame.loc[frame.block.eq('train')]
    counts = train.groupby(['trading_date', 'session_id']).size().unstack(fill_value=0)
    days = counts.index[(counts == 23).all(axis=1)][:5].tolist()
    subset = train.loc[train.trading_date.isin(days)]
    rows, errors = [], []
    m = base['M_star']
    assert m == 8 and len(subset) == 230
    price_error = float(np.max(np.abs(np.log(subset.close / subset.prev_close) - subset.raw_log_return)))
    assert price_error < 1e-12
    for sid, ((day, session), group) in enumerate(subset.groupby(['trading_date', 'session_id'], sort=True)):
        group = group.sort_values('bar_end')
        x = group.raw_log_return.to_numpy() / base['scale']
        variance, ends = rolling_variance(x, m, np.zeros(len(x), dtype=int))
        direct = np.array([np.var(x[j-m+1:j+1], ddof=1) for j in range(m-1, len(x))])
        errors.extend(np.abs(variance - direct).tolist())
        np.testing.assert_allclose(variance, direct, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(group.prev_close.to_numpy()[1:], group.close.to_numpy()[:-1], rtol=0, atol=1e-12)
        start = sid * 24
        rows.append(dict(index=start, date=day, session=int(session), clock=str(group.iloc[0].bar_start),
                         price=float(group.iloc[0].prev_close), x=np.nan, vhat=np.nan))
        v = dict(zip(map(int, ends), map(float, variance)))
        for j, (_, row) in enumerate(group.iterrows()):
            rows.append(dict(index=start+j+1, date=day, session=int(session), clock=str(row.bar_end),
                             price=float(row.close), x=float(x[j]), vhat=v.get(j, np.nan)))
        if sid < 9:
            rows.append(dict(index=start+23.5, date=day, session=int(session), clock='session_break',
                             price=np.nan, x=np.nan, vhat=np.nan))
    plot = pd.DataFrame(rows)
    assert [int(plot[c].notna().sum()) for c in ['price', 'x', 'vhat']] == [240, 230, 160]
    csv('model_data_overview.csv', plot)
    pmin = math.floor(plot.price.min()*10)/10-.05
    pmax = math.ceil(plot.price.max()*10)/10+.05
    xmax = math.ceil(plot.x.abs().max()*2)/2
    vmax = math.ceil(plot.vhat.max()*5)/5+.2
    def dividers(lo, hi):
        return '\n'.join(r'\draw[gray!35,densely dotted] (axis cs:' + f'{23.5+24*k},{lo}' +
                         r') -- (axis cs:' + f'{23.5+24*k},{hi}' + ');' for k in range(9))
    figure = r'''\begin{tikzpicture}
\begin{groupplot}[group style={group size=1 by 3,vertical sep=1.0cm},
width=0.94\linewidth,height=5.3cm,xmin=0,xmax=239,
xtick={23.5,71.5,119.5,167.5,215.5},xticklabels={01-03,01-04,01-05,01-06,01-09},
grid=major,grid style={gray!15},unbounded coords=jump,
tick label style={font=\small},title style={font=\small},
scaled y ticks=false,yticklabel style={/pgf/number format/fixed}]
'''
    panels = [('price', '(a) 5 分钟收盘价', '价格（元）', pmin, pmax, 'blue!70!black'),
              ('x', r'(b) 归一化收益 $x_t=r_t/s_{\rm tr}$', r'$x_t$（无量纲）', -xmax, xmax, 'blue!70!black'),
              ('vhat', r'(c) 8 个收益估计的局部方差 $\widehat v_{t,8}$', '方差（无量纲）', 0, vmax, 'orange!85!black')]
    for col, title, ylabel, lo, hi, color in panels:
        extra = r'xticklabels=\empty' if col != 'vhat' else r'xlabel={交易日期（2017 年，休市间隔压缩）}'
        figure += r'\nextgroupplot[' + f'title={{{title}}},ylabel={{{ylabel}}},ymin={lo},ymax={hi},' + extra + ']\n'
        if col == 'x':
            figure += r'\addplot[gray!65,dashed] coordinates {(0,0) (239,0)};' + '\n'
        figure += r'\addplot[' + color + r',thick] table[x=index,y=' + col + r',col sep=comma]{model_data_overview.csv};' + '\n'
        figure += dividers(lo, hi) + '\n'
    tex('model_data_overview.tex', figure + '\\end{groupplot}\n\\end{tikzpicture}\n')
    framework = (OLD / 'modeling_framework_detailed.tex').read_text(encoding='utf-8')
    assert '2018--2019 年留出段' in framework
    framework = framework.replace('2018--2019 年留出段', '2019 年留出段')
    framework = framework.replace('按日期划分训练与评价样本', '按日期以 $8:1:1$ 划分训练、验证与测试样本')
    tex('modeling_framework_detailed.tex', framework)
    return {'days': days, 'prices': 240, 'returns': 230, 'variances': 160, 'scale': base['scale'], 'M': m,
            'price_return_max_abs_error': price_error, 'rolling_variance_max_abs_error': max(errors)}


def densities(base, frame):
    edges = np.array(base['bins']['x_edges'])
    assert len(edges) == 81
    center = (edges[1:] + edges[:-1]) / 2
    width = np.diff(edges)
    audit = {}
    for block in ['train', 'test']:
        x = frame.loc[frame.block.eq(block), 'raw_log_return'].to_numpy() / base['scale']
        count = np.histogram(x, edges)[0]
        density = count / (len(x) * width)
        csv(f'density_{block}.csv', pd.DataFrame({'x': center, 'bin_left': edges[:-1], 'bin_right': edges[1:],
                                                'count': count, 'density': density,
                                                'plot_density': np.where(count > 0, density, np.nan)}))
        audit[block] = {'returns': len(x), 'inside_bins': int(count.sum()), 'outside_bins': int(len(x)-count.sum()),
                        'plotted_probability_mass': float(np.sum(density*width))}
    grid = np.linspace(edges[0], edges[-1], 801)
    models = {'x': grid, 'Gaussian': stats.norm.pdf(grid, base['gaussian']['mean'], base['gaussian']['sd'])}
    for n in [1, 8]:
        fit = next(row for row in base['selection_table'] if row['N'] == n)
        models[f'N{n}'] = np.exp(engine.logpdf(grid, engine.grid(n, fit['beta'])))
        if n == 1:
            exact = stats.t.pdf(grid, df=2*fit['beta']+2, scale=math.sqrt(fit['beta']/(fit['beta']+1)))
            audit['N1_closed_form_max_abs_error'] = float(np.max(np.abs(models['N1'] - exact)))
            assert audit['N1_closed_form_max_abs_error'] < 1e-7
    csv('density_models.csv', pd.DataFrame(models))
    figure = r'''\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.25cm},
width=.45\linewidth,height=6.5cm,ymode=log,ymin=0.00001,ymax=2,
xlabel={归一化收益 $x$},ylabel={概率密度},
grid=major,grid style={gray!15},unbounded coords=jump,
tick label style={font=\small},title style={font=\small},
legend style={font=\scriptsize,draw=none,at={(.98,.98)},anchor=north east},
'''+f'xmin={edges[0]},xmax={edges[-1]}'+']\n'
    for block, title in [('train', '(a) 训练段'), ('test', '(b) 测试段')]:
        figure += r'\nextgroupplot[title={' + title + '}' + (',ylabel={}' if block=='test' else '') + ']\n'
        series = [(r'only marks,mark=*,mark size=1.1pt,black!65', 'plot_density', f'density_{block}.csv', '经验密度'),
                  ('gray!75!black,dashed,thick', 'Gaussian', 'density_models.csv', '高斯'),
                  ('blue!70!black,thick', 'N1', 'density_models.csv', '单层'),
                  ('orange!85!black,thick', 'N8', 'density_models.csv', '八层')]
        for style, col, path, label in series:
            figure += r'\addplot['+style+r'] table[x=x,y='+col+r',col sep=comma]{'+path+'};\n'
            if block == 'train':
                figure += r'\addlegendentry{'+label+'}\n'
    tex('density_plot.tex', figure+'\\end{groupplot}\n\\end{tikzpicture}\n')
    audit.update(n_bins=80, edges=edges.tolist(), empirical_normalization='count / (all returns in the period * bin width)',
                 models_fit_on='train', train_and_test_use_identical_model_curves=True)
    return audit


def depths():
    primary = checked(PRIMARY, 'depth')
    internal = checked('600030.SH_5min_raw', 'depth')
    validate = checked('600030.SH_5min_raw', 'validation')
    assert internal['finite_minimum_in_tested_grid']
    best = internal['best_N_in_tested_grid']
    assert min(validate['interior_refined_neighbors'], key=lambda r:r['error'])['N'] == best
    for name, value in [('depth_primary.csv', primary), ('depth_internal.csv', internal)]:
        frame = pd.DataFrame([{k:r[k] for k in ['N', 'beta', 'error', 'term_returns', 'term_background']} for r in value['fits']])
        frame['limit'] = value['limit']['error']
        csv(name, frame)
    lo, hi = max(1, best-5), min(32, best+11)
    near = [r['error'] for r in internal['fits'] if lo <= r['N'] <= hi] + [internal['limit']['error']]
    padding = (max(near)-min(near))*.15
    ymin, ymax = min(near)-padding, max(near)+padding
    figure = r'''\begin{tikzpicture}
\begin{axis}[width=.48\linewidth,height=6cm,xmode=log,log basis x=2,
xlabel={层数 $N$},ylabel={联合误差},title={中国平安：接近极限},
xtick={1,8,64,512,4096},xticklabels={1,8,64,512,4096},
grid=major,grid style={gray!20},legend style={font=\scriptsize,at={(.98,.98)},anchor=north east}]
\addplot[blue!70!black,thick,mark=*,mark size=1pt] table[x=N,y=error,col sep=comma]{depth_primary.csv};
\addlegendentry{已测层数}
\addplot[orange!80!black,dashed,thick] table[x=N,y=limit,col sep=comma]{depth_primary.csv};
\addlegendentry{大层数极限}
\end{axis}
\end{tikzpicture}\hfill
\begin{tikzpicture}
\begin{axis}[width=.48\linewidth,height=6cm,
'''+f'xmin={lo},xmax={hi},ymin={ymin},ymax={ymax},\n'+r'xlabel={层数 $N$},ylabel={联合误差},title={中信证券：'+f'{best} 层最低'+r'},'+ '\n'+r'xtick={'+f'{lo},{best},{best+5},{hi}'+r'},scaled y ticks=false,yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=3},'+ '\n'+f'restrict x to domain={lo}:{hi},'+r'''unbounded coords=jump,grid=major,grid style={gray!20}]
\addplot[blue!70!black,thick,mark=*,mark size=1.4pt] table[x=N,y=error,col sep=comma]{depth_internal.csv};
\addplot[orange!80!black,dashed,thick] table[x=N,y=limit,col sep=comma]{depth_internal.csv};
\addplot[only marks,red!80!black,mark=*,mark size=2.8pt] coordinates {'''+f'({best},{internal["best"]["error"]})'+r'''};
\end{axis}
\end{tikzpicture}
'''
    tex('depth_extension.tex', figure)
    return {'primary_last_N': primary['last_N'], 'internal_symbol': internal['symbol'], 'internal_best_N': best,
            'internal_plot_domain': [lo, hi], 'internal_plot_ylim': [ymin, ymax]}


def oos(evaluation):
    lookup = {(r['a'], r['b'], r['block']): r for r in evaluation['paired_intervals']}
    output, allbounds = {}, []
    for index, (a, b) in enumerate([('base_N1', 'base_N8'), ('base_N8', 'Gaussian')], 1):
        rows = []
        for y, period in [(3, '2026'), (2, '2026_Jan_Apr'), (1, '2026_May_Aug')]:
            result = lookup[a,b,period]
            lo, hi = result['ci95_by_block_days']['5']
            assert lo <= result['mean'] <= hi
            rows.append(dict(period=period, value=result['mean'], y=y+(.075 if index==1 else -.075),
                             minus=result['mean']-lo, plus=hi-result['mean'], ci_lower=lo, ci_upper=hi))
            allbounds.extend([lo,hi])
        csv(f'oos_2026_{index}.csv', pd.DataFrame(rows))
        output[f'{a}_minus_{b}'] = rows
    span = max(allbounds)-min(allbounds)
    xmin, xmax = min(0.,min(allbounds))-.08*span, max(allbounds)+.08*span
    figure = (OLD / 'oos_2026.tex').read_text(encoding='utf-8')
    figure = figure.replace('xmin=-0.035,xmax=0.145', f'xmin={xmin},xmax={xmax}')
    assert f'xmin={xmin}' in figure
    tex('oos_2026.tex', figure)
    return output


def thinfit(row):
    return {k:row[k] for k in ['N','beta','tau','error','term_returns','term_background','parameter_boundary','optimizer_success'] if k in row}


def metrics():
    protocol = load(FITS / 'protocol.json')
    entries, pending, sources = [], [], {}
    for case in protocol['cases']:
        key = f"{case['symbol']}_{case['frequency_minutes']}min_{case['variant']}"
        paths = {suffix:FITS/f'{key}.{suffix}.json' for suffix in ['base','depth','evaluation','validation']}
        if not all(path.exists() for path in paths.values()):
            pending.append(key)
            continue
        base, depth, evaluation, validation = [checked(key, suffix) for suffix in paths]
        assert evaluation['input_sha256']['base'] == sha(paths['base'])
        assert evaluation['input_sha256']['depth'] == sha(paths['depth'])
        assert validation['evaluation_sha256'] == sha(paths['evaluation'])
        sources.update({path.relative_to(ROOT).as_posix():sha(path) for path in paths.values()})
        entries.append({**case, 'key':key, 'name':NAMES[case['symbol']], 'scale':base['scale'], 'M':base['M_star'],
                        'base_selected':base['selected'], 'base_selection_table':base['selection_table'],
                        'counts':evaluation['counts'], 'gaussian':base['gaussian'], 'seasonality':base['seasonality'],
                        'depth':{'last_N':depth['last_N'], 'best_N':depth['best_N_in_tested_grid'],
                                 'finite_minimum':depth['finite_minimum_in_tested_grid'], 'practical_plateau':depth['practical_plateau'],
                                 'best':thinfit(depth['best']), 'limit':thinfit(depth['limit']),
                                 'fits':[thinfit(r) for r in depth['fits']]},
                        'models':evaluation['models'], 'scores':evaluation['scores'], 'paired_intervals':evaluation['paired_intervals'],
                        'numerical_validation':{'status':validation['status'], 'final_refined_error_delta':validation['final_refined_error_delta'],
                                                'final_characteristic_error_delta':validation['final_characteristic_error_delta'],
                                                'interior_refined_neighbors':[thinfit(r) for r in validation['interior_refined_neighbors']],
                                                'max_log_density_error':max(r['max_log_density_error'] for r in validation['observed_range_density_checks'])}})
    primary = next(row for row in entries if row['key'] == PRIMARY)
    path = FITS / f'{PRIMARY}.point_scores.csv.gz'
    ev = checked(PRIMARY, 'evaluation')
    assert sha(path) == ev['point_scores_sha256']
    points = pd.read_csv(path, dtype={'trading_date':str}, float_precision='round_trip')
    full = points.loc[points.block.astype(str).eq('2026')]
    periods = {}
    for label, subset in [('2026',full), ('2026_Jan_Apr',full.loc[full.trading_date.le('20260430')]),
                          ('2026_May_Aug',full.loc[full.trading_date.ge('20260501')])]:
        assert len(subset)
        periods[label] = {'n':len(subset), 'days':int(subset.trading_date.nunique()),
                          'first_date':subset.trading_date.min(), 'last_date':subset.trading_date.max(),
                          'mean_log_scores':{col.removeprefix('log_score_'):float(subset[col].mean()) for col in subset if col.startswith('log_score_')},
                          'paired_intervals':[r for r in primary['paired_intervals'] if r['block']==label]}
        for interval in periods[label]['paired_intervals']:
            difference = periods[label]['mean_log_scores'][interval['a']] - periods[label]['mean_log_scores'][interval['b']]
            assert abs(difference-interval['mean']) < 1e-12
    for row in ev['scores']:
        if row['block']=='2026':
            assert abs(row['mean_log_score']-periods['2026']['mean_log_scores'][row['model']]) < 1e-12
    primary['periods_2026'] = periods
    sources[path.relative_to(ROOT).as_posix()] = sha(path)
    summary = {'status':'passed' if not pending else 'incomplete', 'pending_cases':pending,
               'completed_cases':len(entries), 'expected_cases':12, 'primary':primary,
               'eight_stock_5min_raw':[row for row in entries if row['variant']=='raw' and row['frequency_minutes']==5],
               'twelve_depth_cases':entries, 'source_sha256':sources, 'script_sha256':sha(__file__)}
    dump(OUT / 'summary.json', summary)
    print('SUMMARY', summary['status'], len(entries), 'of 12; pending:', pending, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all','figures','summary'], default='all')
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ['all','figures']:
        base = checked(PRIMARY, 'base')
        evaluation = checked(PRIMARY, 'evaluation')
        frame = raw_primary(base)
        result = {'status':'passed', 'model_data':overview(base, frame), 'density':densities(base, frame),
                  'depth':depths(), 'oos':oos(evaluation), 'script_sha256':sha(__file__)}
        names = ['model_data_overview.csv','model_data_overview.tex','modeling_framework_detailed.tex',
                 'density_train.csv','density_test.csv','density_models.csv','density_plot.tex',
                 'depth_primary.csv','depth_internal.csv','depth_extension.tex','oos_2026_1.csv','oos_2026_2.csv','oos_2026.tex']
        result['output_sha256'] = {name:sha(FIG/name) for name in names}
        result['input_sha256'] = {str(path.relative_to(ROOT)):sha(path) for path in [DATA/'historical/returns_601318.SH_5min.csv.gz',
                                  FITS/f'{PRIMARY}.base.json', FITS/f'{PRIMARY}.depth.json', FITS/f'{PRIMARY}.evaluation.json',
                                  FITS/'600030.SH_5min_raw.depth.json', FITS/'600030.SH_5min_raw.validation.json']}
        dump(OUT / 'figure_audit.json', result)
        print('FIGURES PASSED', result['model_data'], flush=True)
    if args.stage in ['all','summary']:
        metrics()


if __name__ == '__main__':
    main()
