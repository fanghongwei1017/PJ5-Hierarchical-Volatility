"""Independent artifact checks; no model refits and no old-file writes."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / '依据/811重划实验_20261114'
OUT = RUN / 'aux'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def main():
    results = [read(p) for p in ('seasonality_check.json', 'clock_check.json', 'training_reselection.json')]
    season, clock, reselection = results
    for result in results:
        assert result['status'] == 'passed'
        evidence = result['inputs']
        for key, path in (
            ('source_sha256', ROOT / evidence['source_file']),
            ('split_sha256', RUN / 'data/split_manifest.json'),
            ('selected_sha256', RUN / 'data/selected_8.csv'),
            ('runtime_sha256', RUN / 'fits/runtime/manifest.json'),
            ('script_sha256', Path(__file__).with_name('aux_runner.py')),
        ):
            assert evidence[key] == sha(path), key
    raw = pd.read_csv(ROOT / clock['inputs']['source_file'], dtype={'trading_date': str}, float_precision='round_trip')
    tr = raw.loc[raw.block.eq('train')]
    base = json.loads((RUN / 'fits/601318.SH_5min_raw.base.json').read_text(encoding='utf-8'))
    assert reselection['train_scale_fixed'] == base['scale'] == float(tr.raw_log_return.std(ddof=1))
    assert tr.trading_date.nunique() == 579 and len(tr) == 26634
    assert raw.loc[raw.block.eq('test'), 'trading_date'].nunique() == 69
    days = np.sort(tr.trading_date.unique())
    rng = np.random.default_rng(20261107)
    assert len(reselection['replicates']) == 20
    for row in reselection['replicates']:
        draw = rng.choice(days, size=len(days), replace=True)
        assert row['draw_sha256'] == hashlib.sha256('|'.join(draw).encode()).hexdigest()
        assert row['session_occurrences'] == 1158 and row['sample_rows'] == 26634
        assert row['n_windows'] == 1158 * (24 - row['M'])
        assert len(row['selection_table']) == 8
        assert all(item['status'] == 'ok' for item in row['selection_table'])
    pairs = pd.read_csv(OUT / 'paired_dates.csv', dtype={'date': str, 'previous_trading_date': str}, float_precision='round_trip')
    endpoints = ('lunch_before_am_last', 'lunch_after_pm_first', 'night_before_prev_pm_last', 'night_after_am_first')
    summary = pd.read_csv(OUT / 'clock_summary.csv', float_precision='round_trip')
    errors = []
    for scope, frame in (
        ('all_sample_exploratory', pairs),
        ('test_with_both_days_in_test', pairs.loc[pairs.block.eq('test') & pairs.previous_block.eq('test')]),
    ):
        n = len(frame)
        if scope.startswith('test'):
            assert n == 68 and frame.previous_trading_date.min() == '20190911' and frame.date.max() == '20191224'
        values = frame[[name + '_raw_abs' for name in endpoints]].to_numpy()
        gaps = frame.calendar_day_gap.to_numpy()
        starts = np.random.default_rng(20261110).integers(0, n, (2000, math.ceil(n / 5)))
        indices = ((starts[:, :, None] + np.arange(5)) % n).reshape(2000, -1)[:, :n]
        for subset, mask in (('all_calendar_gaps', np.ones(n, dtype=bool)), ('calendar_day_gap_eq_1', gaps == 1)):
            target = summary.loc[summary.scope.eq(scope) & summary.subset.eq(subset) & summary.block_trading_days.eq(5)].iloc[0]
            val = values[mask]
            corr = np.corrcoef(val.T)
            errors.extend([abs(corr[0, 1] - target.lunch_pearson), abs(corr[2, 3] - target.night_pearson)])
            differences = []
            for indices_one in indices:
                sampled = values[indices_one[mask[indices_one]]]
                corr = np.corrcoef(sampled.T)
                differences.append(corr[2, 3] - corr[0, 1])
            lo, hi = np.quantile(differences, [.025, .975])
            errors.extend([abs(lo - target.ci95_lower), abs(hi - target.ci95_upper)])
    assert max(errors) < 1e-12
    intraday = pd.read_csv(OUT / 'intraday_mean_absolute_returns.csv', float_precision='round_trip')
    for label, column in (('before', 'mean_abs_raw'), ('after', 'mean_abs_deseasonalized')):
        assert abs(float(intraday[column].max() / intraday[column].min()) - season['mean_absolute_amplitude_' + label]) < 1e-12
    report = {'status': 'passed', 'source_hashes_unchanged': True, 'training_scale_matches_main_fit': True,
              'bootstrap_draws_and_session_counts_checked': 20, 'clock_independent_point_and_5day_ci_cases': 4,
              'clock_ci_replicates_per_case': 2000, 'clock_max_abs_difference_numpy_corrcoef': max(errors),
              'intraday_group_count': len(intraday), 'validator_sha256': sha(__file__)}
    (OUT / 'independent_validation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
