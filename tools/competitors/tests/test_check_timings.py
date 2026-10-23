import subprocess
import tempfile
import unittest
from pathlib import Path


CHECKER = Path(__file__).parents[1] / "check_timings.py"


class TimingValidationTest(unittest.TestCase):
    def run_checker(self, contents, *arguments):
        with tempfile.TemporaryDirectory() as directory:
            timing_file = Path(directory) / "timings"
            timing_file.write_text(contents, encoding="utf-8")
            return subprocess.run(
                ["python3", str(CHECKER), *arguments, "--file", str(timing_file)],
                capture_output=True,
                text=True,
                check=False,
            )

    def run_heavydb_checker(self, timings, profiles, *arguments):
        with tempfile.TemporaryDirectory() as directory:
            timing_file = Path(directory) / "timings.tsv"
            profile_file = Path(directory) / "profile_timings.tsv"
            timing_file.write_text(timings, encoding="utf-8")
            profile_file.write_text(profiles, encoding="utf-8")
            return subprocess.run(
                [
                    "python3",
                    str(CHECKER),
                    "heavydb",
                    *arguments,
                    "--file",
                    str(timing_file),
                    "--profile-file",
                    str(profile_file),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_heavydb_accepts_complete_numeric_timings(self):
        result = self.run_heavydb_checker(
            "query\trun\texecution_ms\toutput_file\n"
            "q1_1\t1\t12.5\tone.log\n"
            "q1_1\t2\t11\ttwo.log\n",
            "query\trun\tmarker\tmarker_count\telapsed_ms_sum\n"
            "q1_1\t1\tHEAVYDB_BENCH_EXECUTE_PLAN_ENVELOPE_MS\t1\t10.5\n"
            "q1_1\t2\tHEAVYDB_BENCH_EXECUTE_PLAN_ENVELOPE_MS\t1\t9.5\n",
            "--queries",
            "q1_1",
            "--iterations",
            "2",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_heavydb_rejects_missing_and_na_timings(self):
        result = self.run_heavydb_checker(
            "query\trun\tms\nq11\t1\tNA\n",
            "query\trun\tmarker\tmarker_count\telapsed_ms_sum\n"
            "q11\t1\tHEAVYDB_BENCH_EXECUTE_PLAN_ENVELOPE_MS\t1\t5.0\n"
            "q11\t2\tHEAVYDB_BENCH_EXECUTE_PLAN_ENVELOPE_MS\t1\t4.0\n",
            "--queries",
            "q11",
            "--iterations",
            "2",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid timing", result.stderr)
        self.assertIn("expected 2 rows", result.stderr)

    def test_heavydb_rejects_missing_preferred_marker(self):
        result = self.run_heavydb_checker(
            "query\trun\tms\nq11\t1\t5.0\n",
            "query\trun\tmarker\tmarker_count\telapsed_ms_sum\n"
            "q11\t1\tHEAVYDB_BENCH_EXECUTE_PLAN_ENVELOPE_MS\t0\t0.0\n",
            "--queries",
            "q11",
            "--iterations",
            "1",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("preferred marker count is 0", result.stderr)

    def test_sirius_checks_only_new_complete_rows(self):
        result = self.run_checker(
            "query,iteration,mode,status,runtime_s,execute_ms\n"
            "old,1,time,ok,1.0,2.0\n"
            "q5,1,time,ok,0.1,3.5\n"
            "q5,2,time,ok,0.2,4.5\n",
            "sirius",
            "--query",
            "q5",
            "--mode",
            "time",
            "--iterations",
            "2",
            "--from-line",
            "2",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sirius_rejects_fallback_and_missing_timing(self):
        result = self.run_checker(
            "query,iteration,mode,status,runtime_s,execute_ms\n"
            "q12,1,time,fallback,0.1,NA\n",
            "sirius",
            "--query",
            "q12",
            "--mode",
            "time",
            "--iterations",
            "1",
            "--from-line",
            "1",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("status is 'fallback'", result.stderr)
        self.assertIn("invalid execute_ms", result.stderr)


if __name__ == "__main__":
    unittest.main()
