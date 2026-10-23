"""CPU-only contract tests. Run: python3 -m unittest discover -s tools/workloads/tests"""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compare_results', HERE / 'compare_results.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GateTests(unittest.TestCase):
    def test_order_ties_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            # Q5 projects (n_name, revenue) and orders on revenue, field 1.
            a.write_text('A,10\nB,10\nC,9\n')
            b.write_text('B,10\nA,10\nC,9\n')
            self.assertTrue(module.compare('tpch', 'Q5', a, b))
            b.write_text('C,9\nA,10\nB,10\n')
            self.assertFalse(module.compare('tpch', 'Q5', a, b))
            b.write_text('A,10\nA,10\nC,9\n')
            self.assertFalse(module.compare('tpch', 'Q5', a, b))

    def test_output_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / 'result'
            result.write_text('1,2,3\n')
            self.assertEqual(module.read_result('tpch', 'Q12', result),
                             [['1', '2', '3']])
            with self.assertRaises(module.SchemaError):
                module.read_result('tpch', 'Q5', result)
            with self.assertRaises(module.SchemaError):
                module.read_result('ssb', 'Q1_2', result)

    def test_matrix_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            # A fake launcher replaces numactl and the GPU binary only here.
            launcher = bin_dir / 'numactl'
            launcher.write_text('''#!/usr/bin/env python3
import os, pathlib, re, sys
args = ' '.join(sys.argv)
compressed = '--compression_type=NONE' not in args
fault = os.environ.get('FAULT', '') if compressed else ''
out = pathlib.Path(os.environ['PROTEUS_RESULT_DUMP_DIR'])
query = re.search(r'INGPU_(Q[0-9_]+)/', args).group(1)
rows = {
    'Q1_1': '1\\n', 'Q2_1': '1,2,3\\n', 'Q3_1': '1,2,3,4\\n',
    'Q4_1': '1,2,3\\n', 'Q5': '1,2\\n', 'Q12': '1,2,3\\n',
    'Q11': '1\\n', 'Q14': '1\\n'
}
if fault != 'missing':
    value = rows[query]
    if fault == 'mismatch': value = '9' + value[1:]
    (out / 'result').write_text(value)
if fault == 'extra': (out / 'extra').write_text('1\\n')
sys.exit(134 if fault == 'abort' else 0)
''')
            launcher.chmod(0o755)
            build = root / 'build'
            for workload in ('ssb', 'tpch', 'stack'):
                binary = (build / 'apps' / 'benchmarks' / 'olap' /
                          f'benchmark-{workload}' / f'proteus-benchmark-{workload}')
                binary.parent.mkdir(parents=True)
                binary.write_text('#!/bin/sh\nexit 0\n')
                binary.chmod(0o755)
            data = root / 'data'
            for workload in ('ssb', 'tpch', 'stack'):
                for layout in ('uncompressed', 'rfor-t1024', 'rfor-t512', 'flbp-t1024'):
                    ds = data / workload / layout
                    ds.mkdir(parents=True)
                    (ds / 'catalog.json').write_text('{}')
            env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}',
                   'PROTEUS_DATA_ROOT': str(data), 'PELAGO': str(root / 'pelago'),
                   'BUILD': str(build),
                   'WORKLOAD_OUTPUT_ROOT': str(root / 'outputs'),
                   'WORKLOAD_FIXTURE_LOCK': str(root / 'fixture.lock'),
                   'REPO_ROOT': str(HERE.parents[1])}
            # Isolate controls inherited from the developer's environment.
            for key in list(env):
                if key.endswith('_DS') or key in ('LOG_DIR', 'WORKLOADS', 'WINDOWS',
                                                  'LAYOUTS', 'MODES',
                                                  'PROTEUS_WORKLOAD_FIXTURE_LOCK_HELD'):
                    env.pop(key)
            good = subprocess.run(['bash', str(HERE / 'gate_results.sh')], env=env, capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr + good.stdout)
            report = next((root / 'outputs').glob('*/results.tsv')).read_text().splitlines()
            self.assertEqual(len(report) - 1, 248)
            self.assertFalse(any('\tFAIL\t' in row for row in report[1:]))
            for fault in ('missing', 'mismatch', 'extra', 'abort'):
                result = subprocess.run(['bash', str(HERE / 'gate_results.sh')],
                                        env={**env, 'WORKLOADS': 'ssb',
                                             'SSB_QUERIES': 'Q1_1', 'WINDOWS': '18',
                                             'FAULT': fault}, capture_output=True)
                self.assertNotEqual(result.returncode, 0, fault)
                fault_report = max((root / 'outputs').glob('*/results.tsv'),
                                   key=lambda path: path.stat().st_mtime_ns).read_text()
                expected_class = 'query_mismatch' if fault == 'mismatch' else 'infrastructure'
                self.assertIn(f'\t{expected_class}\t', fault_report)
                if fault == 'abort':
                    self.assertIn('\t134\tmatch\tprocess_rc_134\t', fault_report)
            (data / 'ssb' / 'rfor-t512' / 'catalog.json').unlink()
            result = subprocess.run(['bash', str(HERE / 'gate_results.sh')],
                                    env={**env, 'WORKLOADS': 'ssb',
                                         'SSB_QUERIES': 'Q1_1', 'WINDOWS': '18'},
                                    capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / 'pelago' / 'inputs' / 'ssbm100').is_symlink())


if __name__ == '__main__':
    unittest.main()
