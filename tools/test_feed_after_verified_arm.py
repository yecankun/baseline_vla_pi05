"""Verify that staged feeding cannot repeat an ambiguous packet or precede review."""
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import feed_after_verified_arm_http as runner


class FeedChecks(unittest.TestCase):
    def exercise(self, *, confirmed=True, drift=False, ambiguous=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm = {"status": "completed_arm_only_awaiting_physical_review", "motion_commands_accepted": 1,
                   "feeder_packets": 0, "elite_ip": "test", "tcp_after_mm_rad": [0.]*6}
            (root/"report.json").write_text(json.dumps(arm))
            (root/"physical_review.json").write_text(json.dumps({"arm_physical_execution_confirmed": confirmed}))
            ec = SimpleNamespace(sock_cmd=Mock(), current_pose=[1. if drift else 0.]+[0.]*5,
                                 disconnect_ETController=Mock())
            ec.sock_cmd.getpeername.return_value = ("test", 8055)
            factory = Mock(return_value=ec)
            feeder = SimpleNamespace(feed_once=Mock(side_effect=TimeoutError("ambiguous send") if ambiguous else None,
                                                   return_value=b'{"status":"ok"}'), close=Mock())
            with patch("sys.argv", ["feed", "--arm-dir", directory, "--execute"]), \
                 patch.dict("sys.modules", {"elite": SimpleNamespace(EC=factory)}), \
                 patch.object(runner, "robot_ready", return_value={}), \
                 patch.object(runner, "load_feeder", return_value=feeder), \
                 patch.object(runner, "capture", side_effect=lambda *args: ({}, {"side": runner.time.time(), "top": runner.time.time()})), \
                 patch.object(runner.time, "sleep"), patch("builtins.print"):
                if not confirmed or drift or ambiguous:
                    with self.assertRaises((RuntimeError, TimeoutError)):
                        runner.main()
                else:
                    runner.main()
                if confirmed:
                    # Even failed/ambiguous executions keep the one-attempt marker.
                    with self.assertRaises(FileExistsError):
                        runner.main()
            self.assertEqual(feeder.feed_once.call_count, 1 if confirmed and not drift else 0)
            if not confirmed:
                factory.assert_not_called()

    def test_one_packet_and_no_repeat(self):
        self.exercise()

    def test_ambiguous_send_cannot_repeat(self):
        self.exercise(ambiguous=True)

    def test_missing_review_or_drift_prevents_packet(self):
        self.exercise(confirmed=False)
        self.exercise(drift=True)

    def test_device_recovery_needs_fresh_connectivity_and_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"feed_once").mkdir()
            prior = {"feeder_send_attempted": True, "feeder_ack_received": False,
                     "destination": "192.168.5.13:8888"}
            prior_path = root/"feed_once/report.json"
            prior_path.write_text(json.dumps(prior))
            original = prior_path.read_bytes()
            network = {"recorded_at": datetime.now(timezone.utc).isoformat(),
                       "ping_exit_code": 1, "neighbor": "192.168.5.13 dev enp5s0 INCOMPLETE"}
            (root/"network.json").write_text(json.dumps(network))
            with self.assertRaises(RuntimeError): runner.output_for_attempt(root, root)
            network.update(ping_exit_code=0, neighbor="192.168.5.13 dev enp5s0 lladdr 00:11:22:33:44:55 REACHABLE")
            (root/"network.json").write_text(json.dumps(network))
            out, recovery = runner.output_for_attempt(root, root)
            out.mkdir(exist_ok=False)
            with self.assertRaises(FileExistsError): out.mkdir(exist_ok=False)
            self.assertEqual(prior_path.read_bytes(), original)
            # The user may identify a changed address. It needs its own matching
            # fresh connectivity evidence, while the old send stays untouched.
            with self.assertRaises(RuntimeError): runner.output_for_attempt(root, root, "192.168.5.5")
            network["neighbor"] = "192.168.5.5 dev enp5s0 lladdr 00:11:22:33:44:55 REACHABLE"
            (root/"network.json").write_text(json.dumps(network))
            _, changed = runner.output_for_attempt(root, root, "192.168.5.5")
            self.assertEqual(changed["prior_destination"], "192.168.5.13:8888")
            self.assertEqual(changed["current_destination"], "192.168.5.5:8888")
            self.assertEqual(prior_path.read_bytes(), original)
            prior["feeder_ack_received"] = True
            prior_path.write_text(json.dumps(prior))
            with self.assertRaises(RuntimeError): runner.output_for_attempt(root, root)


if __name__ == "__main__":
    unittest.main()
