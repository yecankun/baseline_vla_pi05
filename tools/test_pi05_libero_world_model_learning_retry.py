"""Synthetic wrapper controls only; mocked engine never trains or writes weights."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_pi05_libero_world_model_learning_retry as retry


class RetryTests(unittest.TestCase):
    def test_authorization_and_bound_files_verify_read_only(self):
        auth, evidence = retry.validate(retry.ROOT, retry.AUTH_SHA, True)
        self.assertEqual(auth["optimizer_updates"], 200)
        self.assertEqual(auth["output_directory_name"], retry.OUT_NAME)
        self.assertGreaterEqual(len(evidence), 19)

    def test_wrong_flag_hash_or_bound_evidence_fail_closed(self):
        for flag in (False, 1, "yes", None):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                retry.validate(retry.ROOT, retry.AUTH_SHA, flag)
        with self.assertRaises(ValueError):
            retry.validate(retry.ROOT, "0" * 64, True)
        with patch.object(retry, "digest", return_value="0" * 64), self.assertRaises(ValueError):
            retry.validate(retry.ROOT, retry.AUTH_SHA, True)

    def test_fixed_argv_has_no_training_parameter_overrides(self):
        auth = retry.read(retry.ROOT / retry.AUTH_PATH)
        argv = retry.engine_argv(retry.ROOT, auth)
        self.assertEqual(argv[-1], "--execute-authorized-diagnostic")
        self.assertIn(retry.AUTH_SHA, argv)
        self.assertIn(str(retry.ROOT / "simulation_output" / retry.OUT_NAME), argv)
        self.assertFalse(set(argv) & {"--steps", "--seed", "--lr", "--device", "--max-records"})

    def fixture(self, root, error=None):
        (root / "simulation_output").mkdir()
        auth = retry.read(retry.ROOT / retry.AUTH_PATH)
        engine = SimpleNamespace(AUTHORIZATION_SHA=retry.OLD_AUTH_SHA, calls=0)
        def main(argv):
            engine.calls += 1
            self.assertEqual(engine.AUTHORIZATION_SHA, retry.AUTH_SHA)
            if error is not None:
                raise error
            out = root / "simulation_output" / retry.OUT_NAME
            out.mkdir()
            # Deliberately mocked completion metadata, not actual training evidence.
            (out / "report.json").write_text(json.dumps({"status": "passed", "optimizer_steps": 200,
                "backward_calls": 200, "authorization_sha256": retry.AUTH_SHA,
                "checkpoint_metadata": {"authorization_sha256": retry.AUTH_SHA}}), encoding="utf-8")
            (out / "status.json").write_text('{"status": "completed"}', encoding="utf-8")
        engine.main = main
        return auth, engine

    def test_single_call_success_and_exception_restore_original_pin(self):
        for error in (None, ValueError("synthetic failure"), SystemExit(7), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                auth, engine = self.fixture(root, error)
                with patch.object(retry, "validate", return_value=(auth, {})), \
                     patch.object(retry, "load_engine", return_value=engine), redirect_stdout(StringIO()):
                    if error is None:
                        retry.execute_once(root, retry.AUTH_SHA, True)
                    else:
                        with self.assertRaises(type(error)):
                            retry.execute_once(root, retry.AUTH_SHA, True)
                self.assertEqual(engine.calls, 1)
                self.assertEqual(engine.AUTHORIZATION_SHA, retry.OLD_AUTH_SHA)
                record = retry.read(root / "simulation_output" / auth["invocation_file_name"])
                self.assertEqual(record["status"], "completed" if error is None else "failed")
                self.assertTrue(record["original_authorization_pin_restored"])
                self.assertFalse(record["automatic_retry_allowed"])

    def test_existing_output_or_invocation_never_dispatches(self):
        for kind in ("out", "invocation"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                auth, engine = self.fixture(root)
                if kind == "out":
                    (root / "simulation_output" / retry.OUT_NAME).mkdir()
                else:
                    (root / "simulation_output" / auth["invocation_file_name"]).touch()
                with patch.object(retry, "validate", return_value=(auth, {})), \
                     patch.object(retry, "load_engine", return_value=engine), self.assertRaises(FileExistsError):
                    retry.execute_once(root, retry.AUTH_SHA, True)
                self.assertEqual(engine.calls, 0)
                self.assertEqual(engine.AUTHORIZATION_SHA, retry.OLD_AUTH_SHA)

    def test_engine_import_path_and_old_pin_are_checked(self):
        fake = SimpleNamespace(__file__=str(retry.ROOT / "tools" / f"{retry.ENGINE_NAME}.py"),
                               AUTHORIZATION_SHA="0" * 64)
        with patch.object(retry.importlib, "import_module", return_value=fake), self.assertRaises(ValueError):
            retry.load_engine(retry.ROOT)
        fake.AUTHORIZATION_SHA = retry.OLD_AUTH_SHA
        fake.__file__ = str(retry.ROOT / "other.py")
        with patch.object(retry.importlib, "import_module", return_value=fake), self.assertRaises(ValueError):
            retry.load_engine(retry.ROOT)


if __name__ == "__main__":
    unittest.main()
