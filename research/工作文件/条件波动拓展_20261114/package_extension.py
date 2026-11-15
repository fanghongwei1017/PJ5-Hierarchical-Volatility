"""Build and verify the portable, read-only conditional-volatility experiment.

Frozen model and scientific-validator sources are never rewritten. The sole
relocation adapter selects the sixteen used five-minute records from the old
32-record provenance manifest and resolves them beneath the portable root.
"""
from pathlib import Path, PurePosixPath
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CODE = Path('工作文件/条件波动拓展_20261114')
BASE = Path('依据/条件波动拓展_20261114')
EXP = BASE / 'experiment'
OLD = Path('依据/811重划实验_20261114')
MANIFEST = BASE / 'delivery_manifest.json'
ARCHIVE = '条件波动拓展复算_20261115.zip'
# The caller chooses the interpreter; protocol_checks enforces frozen versions.
PYTHON = Path(sys.executable)


def load(path):
    return json.loads(Path(path).read_text('utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError(f'Unsafe package member: {name}')
    return p


def validate_zip_members(z, expected):
    seen = set()
    for entry in z.infolist():
        safe_relative(entry.filename)
        key = entry.filename.casefold()
        if key in seen or entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16):
            raise ValueError(f'Duplicate, directory, or symbolic link: {entry.filename}')
        seen.add(key)
    if set(z.namelist()) != expected:
        raise ValueError('Archive member set differs from manifest')


def verify_payload(root, records):
    root = Path(root).resolve()
    seen = set()
    for record in records:
        name = record['path']; safe_relative(name)
        if name.casefold() in seen:
            raise ValueError(f'Duplicate inventory path: {name}')
        seen.add(name.casefold())
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing or escaped input: {name}')
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError(f'Package payload changed: {name}')


def relocate_inputs(manifest, root, symbols):
    """Copy metadata only; keep original checksums and never touch source bytes."""
    selected = []
    for source in manifest['outputs']:
        if source['frequency'] != 5 or source['symbol'] not in symbols:
            continue
        period = source['period']
        if period not in ('historical', 'oos2026'):
            raise ValueError(f'Unexpected period: {period}')
        record = copy.deepcopy(source)
        folder = 'historical' if period == 'historical' else 'oos2026'
        record['path'] = str(Path(root) / OLD / 'data' / folder / f"returns_{record['symbol']}_5min.csv.gz")
        selected.append(record)
    keys = [(r['symbol'], r['period']) for r in selected]
    expected = {(s, p) for s in symbols for p in ('historical', 'oos2026')}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Used five-minute manifest must contain each symbol/period exactly once')
    result = copy.deepcopy(manifest)
    result['outputs'] = selected; result['files'] = len(selected)
    return result


def dependencies():
    symbols = load(ROOT / EXP / 'protocol.json')['symbols']
    paths = {CODE / n for n in ('extension_fit.py', 'extension_validate.py', 'package_extension.py')}
    paths.update(BASE / n for n in ('README_复算.md',))
    paths.update(OLD / 'data' / n for n in ('selected_8.csv', 'split_manifest.json', 'final_manifest.json'))
    paths.add(OLD / 'fits/runtime/fit_engine.py')
    paths.add(EXP / 'runtime/fit_engine.py')
    for s in symbols:
        paths.add(OLD / 'fits' / f'{s}_5min_raw.base.json')
        for folder in ('historical', 'oos2026'):
            paths.add(OLD / 'data' / folder / f'returns_{s}_5min.csv.gz')
        for suffix in ('evaluation.json', 'point_scores.csv.gz', 'states.json', 'validation.json'):
            paths.add(EXP / f'{s}.{suffix}')
    for name in ('protocol.json', 'selection.json', 'preflight_tests.json', 'summary.json',
                 'paired_score_summary.csv', 'state_floor_audit.json', 'validate_csv_schema.py', 'summarize_results.py'):
        paths.add(EXP / name)
    training = sorted((ROOT / EXP / 'training').glob('*.json'))
    if len(training) != 52:
        raise ValueError('Expected exactly 52 frozen training records')
    paths.update(p.relative_to(ROOT) for p in training)
    audit_names = ['preflight.json', 'independent_validation.json', 'brief_summary.json',
                   'coverage_scope.json', 'coverage_preflight.json', 'test_package_contract.py']
    audit_names += [f'{s}.json' for s in symbols]
    paths.update(BASE / 'validation' / n for n in audit_names)
    return sorted(paths, key=lambda p: p.as_posix())


def inventory(paths):
    records = []
    for rel in paths:
        path = ROOT / rel
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'Missing dependency or link outside workspace: {rel}')
        records.append({'path': rel.as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    return records


def payload_digest(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def import_validator(output):
    path = ROOT / CODE / 'extension_validate.py'
    spec = importlib.util.spec_from_file_location('portable_independent_validator', path)
    v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
    original_load = v.load
    original_path = (ROOT / OLD / 'data/final_manifest.json').resolve()
    symbols = load(ROOT / EXP / 'protocol.json')['symbols']
    relocated = relocate_inputs(load(original_path), ROOT, symbols)
    def portable_load(path):
        return copy.deepcopy(relocated) if Path(path).resolve() == original_path else original_load(path)
    v.load = portable_load
    v.OUT = output
    v.init()
    return v


def quick_case(v, symbol, h):
    """Recompute state and provenance, without refitting or expensive CI/CDF audit."""
    np, pd = v.np, v.pd
    ep = ROOT / EXP / f'{symbol}.evaluation.json'
    ev = load(ep)
    point = ROOT / EXP / f'{symbol}.point_scores.csv.gz'
    d = pd.read_csv(point, dtype={'trading_date': str, 'block': str, 'session_id': str},
                    low_memory=False, float_precision='round_trip')
    hist, hp = v.source(symbol, 'historical'); later, lp = v.source(symbol, '2026'); later['block'] = '2026'
    assert ev['input_sha256'] == {'historical': sha(hp), '2026': sha(lp), 'selection': sha(ROOT / EXP / 'selection.json'), 'protocol': sha(ROOT / EXP / 'protocol.json')}
    assert ev['selected_h'] == h and ev['point_scores_sha256'] == sha(point)
    scale = float(hist.loc[hist.block.eq('train'), 'raw_log_return'].std(ddof=1))
    assert abs(scale - ev['scale']) < 1e-14
    a, ah = v.independent_state(hist, h, scale); b, bh = v.independent_state(later, h, scale)
    ref = pd.concat([a, b], ignore_index=True)
    for col in ('trading_date', 'session_id', 'bar_start', 'block', 'observed_day_ordinal'):
        assert d[col].astype(str).equals(ref[col].astype(str)), (symbol, col)
    np.testing.assert_array_equal(d.raw_log_return, ref.raw_log_return)
    np.testing.assert_allclose(d.x, ref.x, rtol=0, atol=2e-12)
    np.testing.assert_allclose(d.q, ref.q, rtol=0, atol=2e-12)
    assert float(d.loc[d.block.eq('2026'), 'q'].iloc[0]) == 1.
    stored = load(ROOT / EXP / f'{symbol}.states.json')
    for period, expected in [('historical', ah), ('2026', bh)]:
        assert len(stored[period]) == len(expected)
        for got, wanted in zip(stored[period], expected):
            assert got['trading_date'] == wanted['trading_date'] and got['n'] == wanted['n']
            for col in ('q_before', 'q_next', 'mean_x2_after_observation'):
                assert abs(got[col] - wanted[col]) < 2e-12
    masks = v.masks(d)
    for block, mask in masks.items():
        part = d.loc[mask]; counts = part.groupby('trading_date').size(); c = ev['counts'][block]
        assert len(part) == c['returns'] and len(counts) == c['days']
        assert int(counts.eq(46).sum()) == c['complete_5min_days']
        assert str(part.trading_date.min()) == c['first_date'] and str(part.trading_date.max()) == c['last_date']
    old = load(ROOT / OLD / 'fits' / f'{symbol}_5min_raw.base.json')
    for model, param in ev['models'].items():
        if model.startswith(('static', 'dynamic')):
            n = 0 if param['family'] == 'Gaussian' else param['N']
            name = f"{symbol}_{'dynamic' if model.startswith('dynamic') else 'static'}_N{n}" + (f'_h{h}' if model.startswith('dynamic') else '')
            trained = load(ROOT / EXP / 'training' / f'{name}.json')
            assert trained['parameter'] == param
            assert trained['input_sha256'] == {'protocol': sha(ROOT / EXP / 'protocol.json'), 'historical': sha(hp)}
            assert trained['training_returns'] == int(masks['train_fit'].sum())
            assert trained['training_days'] == d.loc[masks['train_fit']].trading_date.nunique()
        elif model == 'old_Gaussian':
            assert param == {'family': 'old_Gaussian', 'mean': old['gaussian']['mean'], 'sd': old['gaussian']['sd'], 'c': old['gaussian']['sd'] ** 2}
        else:
            n = int(model.removeprefix('oldjoint_N'))
            assert param == {'family': 'hierarchy', 'N': n, 'beta': next(r['beta'] for r in old['selection_table'] if r['N'] == n), 'c': 1.}
    audit = load(ROOT / BASE / 'validation' / f'{symbol}.json')
    assert audit['status'] == ev['status'] == 'passed'
    assert audit['script_sha256'] == sha(ROOT / CODE / 'extension_validate.py')
    assert audit['evaluation_sha256'] == sha(ep) and audit['point_scores_sha256'] == sha(point)
    assert audit['worker_numerical_validation_sha256'] == sha(ROOT / EXP / f'{symbol}.validation.json')
    for source in audit['parameter_sources'].values():
        assert sha(ROOT / Path(source['path'])) == source['sha256']
    return {'symbol': symbol, 'returns_checked': len(d), 'counts': ev['counts'],
            'state_max_error': float(np.max(np.abs(d.q - ref.q))), 'parameter_models_checked': len(ev['models'])}


def install_guard(output):
    """Python open audit, not an operating-system sandbox."""
    output = output.resolve()
    allowed = [ROOT.resolve(), Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    opened = set(); rejected = []
    def audit(event, args):
        if event != 'open' or isinstance(args[0], int):
            return
        try:
            p = Path(os.fsdecode(args[0])).resolve()
        except TypeError:
            raise PermissionError('Unsupported open path in package check')
        writable = bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if not any(p.is_relative_to(base) for base in allowed) or (writable and not p.is_relative_to(output)):
            rejected.append(str(p)); raise PermissionError(f'Package guard denied: {p}')
        if p.is_relative_to(ROOT):
            opened.add(p.relative_to(ROOT).as_posix())
    sys.addaudithook(audit)
    return opened, rejected


def verify_internal(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Verification output must be under the package root')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    opened, rejected = install_guard(output)
    # Negative control is caught before any attempt to open an outside file.
    try:
        (ROOT.parent / '__extension_guard_outside_probe__').read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError('Outside-package read was not rejected')
    manifest = load(ROOT / MANIFEST); verify_payload(ROOT, manifest['payload'])
    v = import_validator(output); v.preflight()
    protocol, selection, inputs = v.protocol_checks()
    cases = [quick_case(v, s, selection['selected_h']) for s in protocol['symbols']]
    archived = load(ROOT / BASE / 'validation/independent_validation.json')
    assert archived['status'] == 'passed' and archived['script_sha256'] == sha(ROOT / CODE / 'extension_validate.py')
    assert archived['protocol_sha256'] == sha(ROOT / EXP / 'protocol.json')
    assert archived['symbols'] == protocol['symbols'] and archived['selected_h'] == selection['selected_h']
    for case in archived['cases']:
        assert case == load(ROOT / BASE / 'validation' / f"{case['symbol']}.json")
    verify_payload(ROOT, manifest['payload'])
    assert len(rejected) == 1
    result = {'status': 'passed', 'versions': protocol['versions'], 'selected_h': selection['selected_h'],
              'elapsed_seconds': time.perf_counter() - started, 'used_input_files_verified': len(inputs),
              'payload_files_verified': len(manifest['payload']), 'payload_sha256': payload_digest(manifest['payload']),
              'scope': 'Unzipped copy: independent preflight, h validation selection, all eight state/normalization/row masks and parameter provenance, archived audit identity; no retraining and no rerun of full score/CDF/bootstrap audit.',
              'guard': {'kind': 'Python open audit hook; not OS isolation', 'read_roots': ['extracted package', 'Python installation'],
                        'write_root': 'new verification output directory only', 'outside_read_negative_control_rejected': True,
                        'unexpected_rejected_opens': 0, 'package_open_paths': sorted(opened)}, 'cases': cases}
    save(output / 'package_selfcheck.json', result)
    print(json.dumps({'status': 'passed', 'stocks': len(cases), 'inputs': len(inputs), 'seconds': result['elapsed_seconds']}, ensure_ascii=False), flush=True)


def write_archive(path, records, manifest):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for record in records:
            z.write(ROOT / record['path'], record['path'])
        z.writestr(MANIFEST.as_posix(), json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(path) as z:
        validate_zip_members(z, {r['path'] for r in records} | {MANIFEST.as_posix()})
        if z.testzip() is not None:
            raise ValueError('Archive CRC verification failed')


def build():
    records = inventory(dependencies())
    manifest = {'version': 1, 'status': 'building', 'archive': ARCHIVE, 'payload_files': len(records),
                'archive_files': len(records) + 1, 'payload_bytes': sum(r['bytes'] for r in records),
                'payload_sha256': payload_digest(records), 'payload': records,
                'dependencies': {'old_used_returns': '16 immutable raw 5-minute CSVs, historical and 2026 for 8 stocks',
                    'old_baselines': '8 raw 5-minute baseline fits plus frozen fit engine',
                    'old_provenance': 'selected_8, split_manifest, original 32-record final_manifest (only 16 five-minute inputs required)',
                    'new_results': '52 training records; selection/protocol; 8 evaluations, state paths, point-score files, numerical validations',
                    'audit': 'final independent audit, preflight, coverage scope and behavior tests; logs and report figures excluded'}}
    candidate = ROOT / (ARCHIVE + '.building')
    write_archive(candidate, records, manifest)
    # Resolve/check the actual temporary directory before its automatic cleanup.
    with tempfile.TemporaryDirectory(prefix='conditional_extension_unpack_') as temp:
        temp_path = Path(temp).resolve(); temp_base = Path(tempfile.gettempdir()).resolve()
        if not temp_path.is_relative_to(temp_base) or temp_path == temp_base or temp_path.is_relative_to(ROOT) or ROOT.is_relative_to(temp_path):
            raise ValueError('Unsafe temporary extraction directory')
        extracted = temp_path / 'package'; extracted.mkdir()
        with zipfile.ZipFile(candidate) as z:
            validate_zip_members(z, {r['path'] for r in records} | {MANIFEST.as_posix()}); z.extractall(extracted)
        script = extracted / CODE / 'package_extension.py'
        checkdir = extracted / '.package_verification'
        subprocess.run([str(PYTHON), '-I', '-B', '-X', 'utf8', str(script), '--verify-internal', str(checkdir)],
                       cwd=extracted, check=True)
        result = load(checkdir / 'package_selfcheck.json')
        verify_payload(extracted, records)
        fresh = temp_path / 'fresh'
        subprocess.run([str(PYTHON), '-I', '-B', '-X', 'utf8', str(script), '--fresh-copy', str(fresh)],
                       cwd=extracted, check=True)
        setup = load(fresh / 'fresh_setup.json'); verify_payload(fresh, setup['files'])
        for forbidden in ('training', 'selection.json'):
            if (fresh / EXP / forbidden).exists():
                raise AssertionError('Fresh copy contains a new fitted-model cache')
        if list((fresh / EXP).glob('*.evaluation.json')) or (fresh / BASE / 'validation').exists():
            raise AssertionError('Fresh copy contains an evaluation/audit cache')
        result['fresh_copy_check'] = {'status': 'passed', 'files_copied': len(setup['files']),
                                      'all_copied_hashes_match': True, 'new_model_result_caches_absent': True,
                                      'retrained': False}
        manifest['selfcheck'] = result
    verify_payload(ROOT, records)
    manifest['status'] = 'passed'; manifest['built_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save(ROOT / MANIFEST, manifest)
    write_archive(candidate, records, manifest)
    os.replace(candidate, ROOT / ARCHIVE)
    print(json.dumps({'archive': str(ROOT / ARCHIVE), 'files': len(records) + 1,
                      'archive_bytes': (ROOT / ARCHIVE).stat().st_size, 'archive_sha256': sha(ROOT / ARCHIVE),
                      'selfcheck': 'passed', 'retrained': False}, ensure_ascii=False), flush=True)


def fresh_copy(destination):
    dest = Path(destination).resolve()
    if dest.exists():
        raise FileExistsError(f'Fresh destination must not exist: {dest}')
    if dest.is_relative_to(ROOT) or ROOT.is_relative_to(dest):
        raise ValueError('Fresh destination must be separate from this package/workspace')
    manifest = load(ROOT / MANIFEST)
    records = [r for r in manifest['payload'] if r['path'].startswith(OLD.as_posix() + '/')
               or r['path'].startswith(CODE.as_posix() + '/')
               or r['path'] in {(EXP / p).as_posix() for p in ('runtime/fit_engine.py', 'protocol.json', 'validate_csv_schema.py', 'summarize_results.py')}
               or r['path'] == (BASE / 'README_复算.md').as_posix()]
    verify_payload(ROOT, records)
    dest.mkdir(parents=True, exist_ok=False)
    for record in records:
        path = dest / record['path']; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / record['path'], path)
    verify_payload(dest, records)
    save(dest / 'fresh_setup.json', {'status': 'passed', 'source_payload_sha256': manifest['payload_sha256'],
                                   'files': records, 'new_training_or_selection_or_evaluation_caches_copied': False})
    print(f'FRESH INPUT COPY: {dest}; {len(records)} files; no new fit/evaluation cache', flush=True)


def full_validation():
    """Portable repeat of the scientific audit; write to a new directory."""
    if (ROOT / 'fresh_setup.json').exists():
        verify_payload(ROOT, load(ROOT / 'fresh_setup.json')['files'])
    else:
        verify_payload(ROOT, load(ROOT / MANIFEST)['payload'])
    output = ROOT / BASE / ('portable_validation_' + time.strftime('%Y%m%dT%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    v = import_validator(output); v.preflight(); v.final()
    save(output / 'portable_scope.json', {'original_input_manifest_unmodified': True,
        'inputs_checked': 16, 'original_archive_audit_inputs': 32,
        'relocation': 'five-minute records only; symbol and period mapped below current package root',
        'package_helper_sha256': sha(Path(__file__))})
    print(f'PORTABLE FULL AUDIT: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--build', action='store_true')
    modes.add_argument('--verify', action='store_true')
    modes.add_argument('--verify-internal', metavar='OUTPUT', help=argparse.SUPPRESS)
    modes.add_argument('--fresh-copy', metavar='NONEXISTENT_DESTINATION')
    modes.add_argument('--validate-full', action='store_true')
    args = parser.parse_args()
    if args.build:
        build()
    elif args.fresh_copy:
        fresh_copy(args.fresh_copy)
    elif args.validate_full:
        full_validation()
    elif args.verify_internal:
        verify_internal(args.verify_internal)
    else:
        output = ROOT / ('.package_verification_' + time.strftime('%Y%m%dT%H%M%S'))
        subprocess.run([str(PYTHON), '-I', '-B', '-X', 'utf8', str(Path(__file__)), '--verify-internal', str(output)], check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
