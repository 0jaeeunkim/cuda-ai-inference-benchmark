"""Bounded setup regression checks; no network, installation, or HOME changes."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/project2_setup.sh"


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name, path in (("dirname", "/usr/bin/dirname"), ("mv", "/usr/bin/mv"),
                           ("python3", sys.executable)):
            (self.bin / name).symlink_to(path)
        self.env = {**os.environ, "PATH": str(self.bin),
                    "PROJECT2_CONDA": str(self.root / "missing-conda")}

    def tearDown(self):
        self.temp.cleanup()

    def run_script(self, stage):
        return subprocess.run(["/bin/bash", str(SCRIPT), stage], env=self.env,
                              text=True, capture_output=True, timeout=10)

    def helper_download(self, url, destination):
        return subprocess.run(
            ["/bin/bash", "-c", 'source "$1" --help >/dev/null; download "$2" "$3"',
             "project2-test", str(SCRIPT), url, str(destination)],
            env=self.env, text=True, capture_output=True, timeout=10)

    def executable(self, name, contents):
        path = self.bin / name
        path.write_text("#!/bin/sh\n" + contents)
        path.chmod(0o755)
        return path

    def test_python_download_without_curl_or_wget(self):
        fixture = self.root / "payload.bin"
        fixture.write_bytes(b"offline regression fixture\x00\xff")
        destination = self.root / "download with spaces.bin"
        result = self.helper_download(fixture.as_uri(), destination)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(destination.read_bytes(), fixture.read_bytes())
        self.assertFalse(Path(str(destination) + ".partial").exists())

    def test_wget_fallback_without_curl(self):
        fixture = self.root / "payload.bin"
        fixture.write_bytes(b"wget regression fixture")
        self.env["PROJECT2_FIXTURE"] = str(fixture)
        self.executable("wget", '/bin/cp "$PROJECT2_FIXTURE" "$4"\n')
        destination = self.root / "wget-download.bin"
        result = self.helper_download("https://example.invalid/not-contacted", destination)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(destination.read_bytes(), fixture.read_bytes())

    def test_failed_download_does_not_replace_destination(self):
        self.executable("curl", "exit 7\n")
        destination = self.root / "existing.bin"
        destination.write_bytes(b"preserve")
        result = self.helper_download("https://example.invalid/not-contacted", destination)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(destination.read_bytes(), b"preserve")

    def test_missing_conda_stops_all_dependent_stages(self):
        for stage in ("create-env", "install-packages", "download-cache"):
            with self.subTest(stage=stage):
                result = self.run_script(stage)
                self.assertEqual(result.returncode, 1)
                self.assertIn("Conda is missing", result.stderr)
                self.assertIn("install-conda", result.stderr)
                self.assertNotIn("No such file or directory", result.stderr)

    def test_missing_environment_stops_package_and_cache_work(self):
        self.env["PROJECT2_CONDA"] = str(self.executable("conda", "exit 1\n"))
        for stage in ("install-packages", "download-cache"):
            with self.subTest(stage=stage):
                result = self.run_script(stage)
                self.assertEqual(result.returncode, 1)
                self.assertIn("Environment cuda-project2 is unavailable", result.stderr)
                self.assertIn("create-env", result.stderr)

    def test_all_stops_after_first_failed_stage(self):
        stage_log = self.root / "stages.txt"
        self.env["PROJECT2_STAGE_LOG"] = str(stage_log)
        self.executable("bash", 'printf "%s\\n" "$2" >> "$PROJECT2_STAGE_LOG"\nexit 17\n')
        result = self.run_script("all")
        self.assertEqual(result.returncode, 17)
        self.assertEqual(stage_log.read_text().splitlines(), ["install-conda"])

    def existing_conda(self, version_status):
        self.env["PROJECT2_CONDA"] = str(self.executable("conda", f'''
case "$1" in
  env) printf '%s\\n' '{{"envs":["/test/envs/cuda-project2"]}}' ;;
  run)
    case "$*" in
      *'python -c'*) exit {version_status} ;;
      *) printf '%s\\n' 'Python 3.10.20' ;;
    esac ;;
  *) printf '%s\\n' 'Unexpected environment mutation' >&2; exit 99 ;;
esac
'''))

    def test_existing_environment_is_reused(self):
        self.existing_conda(version_status=0)
        result = self.run_script("create-env")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Reusing existing cuda-project2", result.stdout)

    def test_existing_environment_with_other_python_is_preserved(self):
        self.existing_conda(version_status=1)
        result = self.run_script("create-env")
        self.assertEqual(result.returncode, 1)
        self.assertIn("different Python version", result.stderr)
        self.assertNotIn("Unexpected environment mutation", result.stderr)


if __name__ == "__main__":
    unittest.main()
