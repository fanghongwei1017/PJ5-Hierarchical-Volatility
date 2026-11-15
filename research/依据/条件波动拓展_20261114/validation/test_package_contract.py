"""Behavioral tests for archive integrity and safe portable relocation."""
from pathlib import Path
import importlib.util
import io
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / '工作文件/条件波动拓展_20261114/package_extension.py'
spec = importlib.util.spec_from_file_location('package_under_test', SOURCE)
pkg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pkg)


class PackageContract(unittest.TestCase):
    def test_subprocesses_reuse_the_selected_interpreter(self):
        self.assertEqual(pkg.PYTHON, Path(sys.executable))

    def archive(self, names):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as z:
            for name in names:
                z.writestr(name, b'abc')
        stream.seek(0)
        return zipfile.ZipFile(stream)

    def test_safe_unicode_relative_payload_accepted(self):
        names = ['依据/数据.csv', '工作文件/run.py']
        with self.archive(names) as z:
            pkg.validate_zip_members(z, set(names))

    def test_traversal_absolute_duplicate_and_unexpected_rejected(self):
        for names in [['../outside'], ['/outside'], ['C:/outside'], ['a\\b'],
                      ['a', 'a'], ['a', 'A'], ['extra']]:
            with self.subTest(names=names), self.archive(names) as z:
                with self.assertRaises(ValueError):
                    pkg.validate_zip_members(z, {'a'})

    def test_payload_tamper_and_missing_file_detected(self):
        with tempfile.TemporaryDirectory(prefix='extension_contract_') as temp:
            root = Path(temp)
            file = root / 'input.csv'
            file.write_text('unchanged', encoding='utf-8')
            records = [{'path': 'input.csv', 'sha256': pkg.sha(file), 'bytes': file.stat().st_size}]
            pkg.verify_payload(root, records)
            file.write_text('tampered', encoding='utf-8')
            with self.assertRaises(ValueError):
                pkg.verify_payload(root, records)
            file.unlink()
            with self.assertRaises(ValueError):
                pkg.verify_payload(root, records)

    def test_fresh_copy_rejects_existing_target_without_changing_it(self):
        with tempfile.TemporaryDirectory(prefix='extension_contract_') as temp:
            target = Path(temp)
            marker = target / 'keep.txt'
            marker.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                pkg.fresh_copy(target)
            self.assertEqual(marker.read_text('utf-8'), 'keep')

    def test_only_used_five_minute_inputs_are_relocated(self):
        syms = ['601318.SH']
        records = [dict(symbol=syms[0], frequency=f, period=p, path='C:/old/absolute.csv', sha256='abc')
                   for f in (5, 10) for p in ('historical', 'oos2026')]
        root = Path('C:/portable')
        got = pkg.relocate_inputs({'outputs': records}, root, syms)
        self.assertEqual(len(got['outputs']), 2)
        self.assertTrue(all(r['frequency'] == 5 and r['sha256'] == 'abc' for r in got['outputs']))
        self.assertTrue(all(Path(r['path']).is_relative_to(root) for r in got['outputs']))
        self.assertEqual(records[0]['path'], 'C:/old/absolute.csv')
        with self.assertRaises(ValueError):
            pkg.relocate_inputs({'outputs': records[:1]}, root, syms)


if __name__ == '__main__':
    unittest.main()
