"""Persistence of variance across two adjacent, nonoverlapping intraday windows."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


DATA = Path(__file__).resolve().parents[1] / '依据/811重划实验_20261114/data'
WINDOW = 8
DRAWS = 2000
BLOCK = 5
SEED = 20261118


def session_energies(frame, scale):
    """One row per complete day, columns (AM first, AM second, PM first, PM second)."""
    frame = frame.copy()
    clock = pd.to_datetime(frame['bar_start'])
    frame['minute'] = clock.dt.hour * 60 + clock.dt.minute
    frame['energy'] = (frame.raw_log_return / scale) ** 2
    prices = frame.pivot(index='trading_date', columns='minute', values='energy').sort_index()
    spans = []
    for start in (9 * 60 + 35, 13 * 60 + 5):
        for offset in (0, 5 * WINDOW):
            minutes = start + offset + 5 * np.arange(WINDOW)
            spans.append(prices.reindex(columns=minutes).mean(axis=1, skipna=False))
    return pd.concat(spans, axis=1).dropna().to_numpy()


def paired_rank_stat(data):
    """Average morning and afternoon rank correlations for each bootstrap sample."""
    ranks = rankdata(data, axis=1, method='average')
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    a, b = centered[:, :, ::2], centered[:, :, 1::2]
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    if np.any(denom == 0):
        raise ValueError('Constant window variances in a rank comparison')
    return ((a * b).sum(axis=1) / denom).mean(axis=1)


def summarize(data, rng):
    n = len(data)
    observed = [float(spearmanr(data[:, i], data[:, i + 1]).statistic) for i in (0, 2)]
    estimate = np.mean(observed)
    if not np.isclose(estimate, paired_rank_stat(data[None, :, :])[0]):
        raise ValueError('Rank estimates disagree')
    statistics = []
    for _ in range(DRAWS // 50):
        starts = rng.integers(n, size=(50, (n + BLOCK - 1) // BLOCK))
        days = ((starts[:, :, None] + np.arange(BLOCK)) % n).reshape(50, -1)[:, :n]
        statistics.extend(paired_rank_stat(data[days]))
    lo, hi = np.quantile(statistics, [0.025, 0.975])
    return dict(days=n, rho_am=observed[0], rho_pm=observed[1], rho=estimate,
                lower=lo, upper=hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stocks = pd.read_csv(DATA / 'selected_8.csv').sort_values('eligible_rank').symbol
    records, hashes = [], {}
    columns = ['trading_date', 'bar_start', 'raw_log_return', 'block']
    for position, symbol in enumerate(stocks):
        samples = []
        for directory in ('historical', 'oos2026'):
            path = DATA / directory / f'returns_{symbol}_5min.csv.gz'
            hashes[str(path.relative_to(DATA).as_posix())] = hashlib.sha256(path.read_bytes()).hexdigest()
            samples.append(pd.read_csv(path, usecols=columns, dtype={'trading_date': str},
                                       float_precision='round_trip'))
        historical, later = samples
        train = historical.loc[historical.block.eq('train')]
        scale = train.raw_log_return.std(ddof=1)
        periods = [('train', train), ('test', historical.loc[historical.block.eq('test')]),
                   ('2026', later)]
        for sequence, (period, sample) in enumerate(periods):
            energies = session_energies(sample, scale)
            rng = np.random.default_rng(np.random.SeedSequence([SEED, position, sequence]))
            records.append(dict(symbol=symbol, period=period, **summarize(energies, rng)))
    out = pd.DataFrame(records)
    out.to_csv(args.output / 'session_persistence.csv', index=False, float_format='%.12g')
    graph = pd.DataFrame({'symbol': list(stocks), 'row': np.arange(len(stocks), 0, -1)})
    for period, prefix in [('test', 'test'), ('2026', 'later')]:
        selected = out.loc[out.period.eq(period)].set_index('symbol').reindex(graph.symbol)
        graph[prefix] = selected.rho.to_numpy()
        graph[prefix + '_minus'] = (selected.rho - selected.lower).to_numpy()
        graph[prefix + '_plus'] = (selected.upper - selected.rho).to_numpy()
    graph.to_csv(args.output / 'session_persistence_plot.csv', index=False, float_format='%.12g')
    (args.output / 'protocol.json').write_text(json.dumps(dict(window_returns=WINDOW,
        minutes_per_return=5, block_days=BLOCK, draws=DRAWS, seed=SEED,
        statistic='mean of two within-session Spearman correlations across complete days',
        source_sha256=hashes), indent=2) + '\n', encoding='utf-8')
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
