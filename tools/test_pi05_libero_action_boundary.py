"""CPU observer transparency and fail-closed trace arithmetic tests."""
from copy import deepcopy
import unittest

import numpy as np

from probe_pi05_libero_action_boundary import Observer, array, summarize


class Target:
    def __init__(self):
        self.calls = 0
        self.output = np.array([1.])

    def action(self, value):
        self.calls += 1
        value[0] = 4.
        return self.output


def fixture():
    native = [.1, .2, .3, 0., 0., 0., -1.03]
    row = dict(step=0, policy_postprocessed_7=native[:], environment_input_7=native[:],
               raw_camera_shapes={"image": [1, 256, 256, 3], "image2": [1, 256, 256, 3]},
               policy_camera_shapes={"observation.images.image": [1, 3, 256, 256],
                                     "observation.images.image2": [1, 3, 256, 256]},
               env_processed_state_shape=[1, 8])
    ctrl = dict(input_min=[-1.] * 6, input_max=[1.] * 6,
                output_min=[-.05] * 3 + [-.5] * 3, output_max=[.05] * 3 + [.5] * 3,
                gripper_speed=.01)
    events = [dict(kind="step", step=0, input=native[:]),
              dict(kind="scale_action", step=0, input=native[:6], output=[.005, .01, .015, 0, 0, 0]),
              dict(kind="format_action", step=0, input=native[6:], before=[.5, -.5], output=[.51, -.51]),
              dict(kind="format_action", step=0, input=native[6:], before=[.51, -.51], output=[.52, -.52]),
              dict(kind="_preprocess_images", step=0, shapes=[[1, 3, 224, 224]] * 3,
                   masks=[[True], [True], [False]])]
    return [row], events, ctrl


class TestObserver(unittest.TestCase):
    def test_original_called_once_return_identity_copied_input_restored(self):
        events = []
        target = Target()
        obs = Observer(events.append)
        obs.wrap(target, "action", lambda x: array(x).tolist(),
                 lambda snap, result: dict(input=snap, output=array(result).tolist()))
        obs.step = 0
        value = np.array([2.])
        result = target.action(value)
        self.assertIs(result, target.output)
        self.assertEqual(target.calls, 1)
        self.assertEqual(value.tolist(), [4.])
        target.output[0] = 100
        self.assertEqual(events[0]["input"], [2.])
        self.assertEqual(events[0]["output"], [1.])
        obs.restore()
        self.assertNotIn("action", vars(target))
        self.assertEqual(target.action.__func__, Target.action)

    def test_warmup_is_called_but_not_logged(self):
        target, events = Target(), []
        obs = Observer(events.append)
        obs.wrap(target, "action", lambda x: self.fail("warmup snapshot"), lambda a, b: {})
        target.action(np.array([1.]))
        self.assertEqual(target.calls, 1)
        self.assertEqual(events, [])
        obs.restore()

    def test_existing_instance_method_restored(self):
        target = Target()
        original = lambda x: x
        target.action = original
        obs = Observer(lambda e: None)
        obs.wrap(target, "action", lambda x: None, lambda a, b: {})
        obs.restore()
        self.assertIs(target.action, original)

    def test_native_exception_propagates_once(self):
        target, calls = Target(), []
        def fail(x):
            calls.append(x)
            raise RuntimeError("native failure")
        target.action = fail
        obs = Observer(lambda e: self.fail("failed native call emitted success"))
        obs.wrap(target, "action", lambda x: x, lambda a, b: {})
        obs.step = 0
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            target.action(1)
        obs.restore()
        self.assertEqual(calls, [1])
        self.assertIs(target.action, fail)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            array([float("nan")])


class TestSummary(unittest.TestCase):
    def test_gripper_overshoot_multiple_substeps_and_no_mutation(self):
        fixture_data = fixture()
        before = deepcopy(fixture_data)
        report = summarize(*fixture_data)
        self.assertEqual(fixture_data, before)
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["out_of_bounds_steps"], 1)
        self.assertEqual(report["arm_saturated_steps"], 0)
        self.assertEqual(report["native_call_counts"][0]["gripper_calls"], 2)
        self.assertEqual([r["out_of_bounds_steps"] for r in report["per_dimension"]], [0] * 6 + [1])

    def test_arm_native_saturation_observed_not_applied(self):
        rows, events, ctrl = fixture()
        rows[0]["environment_input_7"][0] = 1.1
        rows[0]["policy_postprocessed_7"][0] = 1.1
        events[0]["input"][0] = events[1]["input"][0] = 1.1
        events[1]["output"][0] = .05
        result = summarize(rows, events, ctrl)
        self.assertEqual(result["arm_saturated_steps"], 1)
        self.assertTrue(all(result["checks"].values()))

    def test_inner_action_change_detected(self):
        rows, events, ctrl = fixture()
        events[0]["input"][6] = -1.
        self.assertFalse(summarize(rows, events, ctrl)["checks"]["inner_environment_input_preserved"])

    def test_wrong_gripper_formula_detected(self):
        rows, events, ctrl = fixture()
        events[2]["output"] = [0., 0.]
        self.assertFalse(summarize(rows, events, ctrl)["checks"]["gripper_original_output_matches_native_formula"])

    def test_missing_or_duplicate_osc_rejected(self):
        rows, events, ctrl = fixture()
        with self.assertRaises(ValueError):
            summarize(rows, [e for e in events if e["kind"] != "scale_action"], ctrl)
        with self.assertRaises(ValueError):
            summarize(rows, events + [events[1]], ctrl)

    def test_config_size_cannot_replace_observed_shape(self):
        rows, events, ctrl = fixture()
        rows[0]["raw_camera_shapes"]["image"][1:3] = [360, 360]
        self.assertFalse(summarize(rows, events, ctrl)["checks"]["two_runtime_cameras_256"])

    def test_missing_valid_camera_detected(self):
        rows, events, ctrl = fixture()
        events[-1]["masks"] = [[True], [False], [False]]
        self.assertFalse(summarize(rows, events, ctrl)["checks"]["internal_two_valid_224_plus_empty"])


if __name__ == "__main__":
    unittest.main()
