"""Package the frozen 8:1:1 experiment; never modify the experiment's results."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
REL_RUN = Path('依据/811重划实验_20261114')
REL_CODE = Path('工作文件/811重划_20261114')
RUN = ROOT / REL_RUN
ARCHIVE = ROOT / '实验复算_811_20261114.zip'
PYTHON = Path(r'C:\Users\lenovo\anaconda3\python.exe')
SCRIPTS = ('data_prepare.py', 'fit_runner.py', 'fit_validate.py', 'aux_runner.py', 'aux_validate.py')
LARGE_SELECTION_TABLES = {'training_stock_days.csv.gz', 'additional_training_stock_days.csv.gz'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def payload_files():
    files = [ROOT / REL_CODE / name for name in (*SCRIPTS, 'package_experiments.py')]
    files.append(RUN / 'README_复算.md')
    for path in (RUN / 'data').rglob('*'):
        if path.is_file() and path.name not in LARGE_SELECTION_TABLES and path.suffix != '.log':
            files.append(path)
    for path in (RUN / 'fits').rglob('*'):
        rel = path.relative_to(RUN / 'fits')
        if not path.is_file() or any('pilot' in part or part == '__pycache__' for part in rel.parts):
            continue
        if '.partial.' in path.name or '.failure.' in path.name:
            continue
        if rel.parts[0] == 'runtime':
            if path.suffix in ('.py', '.json'):
                files.append(path)
        elif path.suffix == '.json' or path.name.endswith('.point_scores.csv.gz') or path.name == 'summary.csv':
            files.append(path)
    files.extend(path for path in (RUN / 'aux').iterdir() if path.is_file() and path.suffix in ('.json', '.csv'))
    files = sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())
    assert all(path.is_file() and path.resolve().is_relative_to(ROOT) for path in files)
    return files


def inventory(files):
    return {path.relative_to(ROOT).as_posix(): {'bytes': path.stat().st_size, 'sha256': sha(path)} for path in files}


def payload_digest(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_archive(path, files, manifest):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for source in files:
            archive.write(source, source.relative_to(ROOT).as_posix())
        archive.writestr((REL_RUN / 'delivery_manifest.json').as_posix(),
                         json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))


GUARDED_VALIDATE = r'''
import json, os, pathlib, runpy, sys
root = pathlib.Path(sys.argv[1]).resolve()
script = root / sys.argv[2]
allowed = [root, pathlib.Path(sys.prefix).resolve(), pathlib.Path(sys.base_prefix).resolve()]
opened = set()
def audit(event, args):
    if event != 'open' or not args or isinstance(args[0], int):
        return
    path = pathlib.Path(os.fsdecode(args[0])).resolve()
    if not any(path.is_relative_to(base) for base in allowed):
        raise PermissionError('Unbundled dependency: ' + str(path))
    opened.add(str(path))
os.chdir(root)
sys.path.insert(0, str(script.parent))
sys.addaudithook(audit)
runpy.run_path(str(script), run_name='__main__')
print('PACKAGE_PATH_CHECK ' + json.dumps({'passed': True, 'opened_paths': len(opened),
    'allowed_roots': [str(path) for path in allowed]}, ensure_ascii=False))
'''


def check_archive_members(path, manifest):
    records = manifest['files']
    manifest_name = (REL_RUN / 'delivery_manifest.json').as_posix()
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == set(records) | {manifest_name}
        assert len(archive.namelist()) == len(records) + 1
        assert json.loads(archive.read(manifest_name)) == manifest
        for name, record in records.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            contents = archive.read(name)
            assert len(contents) == record['bytes']
            assert hashlib.sha256(contents).hexdigest() == record['sha256'], name


def extracted_self_check(path, records):
    assert PYTHON.is_file(), PYTHON
    temporary_base = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix='811_package_check_', dir=temporary_base) as temporary:
        destination = Path(temporary).resolve()
        assert destination.parent == temporary_base and destination.name.startswith('811_package_check_')
        assert not destination.is_relative_to(ROOT) and not ROOT.is_relative_to(destination)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(destination)
        for name, record in records.items():
            assert sha(destination / name) == record['sha256'], name
        checks = []
        for name in ('fit_validate.py', 'aux_validate.py'):
            command = [str(PYTHON), '-I', '-X', 'utf8', '-c', GUARDED_VALIDATE, str(destination), str(REL_CODE / name)]
            completed = subprocess.run(command, cwd=destination, text=True, encoding='utf-8',
                                       capture_output=True, timeout=600, check=False)
            checks.append({'validator': name, 'interpreter': str(PYTHON), 'isolated_python': True,
                           'working_directory': 'temporary archive extraction', 'returncode': completed.returncode,
                           'stdout': completed.stdout, 'stderr': completed.stderr})
            print(name, 'exit', completed.returncode, flush=True)
            if completed.returncode:
                raise RuntimeError(completed.stdout + completed.stderr)
        for name, record in records.items():
            assert sha(destination / name) == record['sha256'], ('validator changed archived file', name)
        return {'status': 'passed', 'payload_sha256': payload_digest(records),
                'mode': 'extract archive; run both validators under Python audit file-open restrictions',
                'application_file_reads_restricted_to_extracted_tree': True,
                'fixed_python_environment_remains_required': True,
                'all_payload_hashes_unchanged_after_validation': True, 'checks': checks}


def fresh_copy(destination):
    """Create a new output-free copy. Never delete or move an existing artifact."""
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise ValueError('Target must not exist: ' + str(destination))
    if destination.is_relative_to(ROOT) or ROOT.is_relative_to(destination):
        raise ValueError('Target must be separate from the source package: ' + str(destination))
    files = []
    for source in payload_files():
        rel = source.relative_to(ROOT)
        if rel.is_relative_to(REL_RUN / 'aux'):
            continue
        if rel.is_relative_to(REL_RUN / 'fits'):
            fit_rel = rel.relative_to(REL_RUN / 'fits')
            if fit_rel != Path('protocol.json') and fit_rel.parts[0] != 'runtime':
                continue
        files.append(source)
    destination.mkdir(parents=True, exist_ok=False)
    for source in files:
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(target) == sha(source)
    (destination / REL_RUN / 'aux').mkdir(exist_ok=True)
    write_json(destination / REL_RUN / 'fresh_setup.json', {
        'status': 'prepared_not_run', 'source_package': str(ROOT),
        'preserved_protocol_sha256': sha(destination / REL_RUN / 'fits/protocol.json'),
        'files': inventory(files), 'existing_source_results_unchanged': True,
        'next': 'run fit_runner.py --phase all, aux_runner.py --stage all, then both validators'})
    print('Fresh copy prepared, no experiment launched:', destination)


def build():
    files = payload_files()
    records = inventory(files)
    assert sum(name.endswith('min.csv.gz') and '/returns_' in name for name in records) == 32
    assert sum(name.endswith('.point_scores.csv.gz') for name in records) == 12
    protocol = json.loads((RUN / 'fits/protocol.json').read_text('utf-8'))
    manifest = {'schema_version': 1, 'status': 'self_check_pending',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'archive_filename': ARCHIVE.name,
        'purpose': 'Recompute models and checks from archived valid raw returns, not from original minute quotation files.',
        'interpreter_used': str(PYTHON), 'versions': protocol['versions'],
        'included_file_count_excluding_this_manifest': len(records),
        'uncompressed_payload_bytes': sum(record['bytes'] for record in records.values()),
        'payload_sha256': payload_digest(records), 'files': records,
        'omitted': ['original minute quotations', 'old experiment data/parameters',
                    'pilot directories', 'partial checkpoints', '__pycache__', 'execution logs',
                    *sorted(LARGE_SELECTION_TABLES)],
        'limits': ['data_prepare.py needs the original external data and historical cache; it is an audit source here.',
                   'The default fit_runner all phase resumes matching saved results.',
                   'A fresh run requires a separate result-free copy and is not part of package self-check.',
                   'File-open audit checks Python-level dependencies; this is not an operating-system sandbox.']}
    candidate = ARCHIVE.with_suffix('.zip.building')
    make_archive(candidate, files, manifest)
    check_archive_members(candidate, manifest)
    checked = extracted_self_check(candidate, records)
    assert inventory(files) == records, 'Source payload changed while packaging'
    manifest['status'] = 'passed'
    manifest['self_check'] = checked
    write_json(RUN / 'delivery_manifest.json', manifest)
    make_archive(candidate, files, manifest)
    check_archive_members(candidate, manifest)
    candidate.replace(ARCHIVE)
    print(json.dumps({'status': 'passed', 'archive': str(ARCHIVE), 'bytes': ARCHIVE.stat().st_size,
                      'sha256': sha(ARCHIVE), 'included_files': len(records) + 1}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh-copy', type=Path, help='Create a new separate copy without prior fit/aux results; no computation.')
    args = parser.parse_args()
    if args.fresh_copy is not None:
        fresh_copy(args.fresh_copy)
    else:
        build()


if __name__ == '__main__':
    main()
