"""Small analytical NumPy fixtures only; no model, public data or training."""
import json
import unittest

import numpy as np

from pi05_libero_projection_geometry import projection_geometry


def f32(value):
    return np.asarray(value, dtype=np.float32)


class ProjectionGeometryTests(unittest.TestCase):
    def geometry(self, x, weight=((1, 1),), bias=(0,), final_weight=None, final_bias=None):
        return projection_geometry(f32(x), f32(weight), f32(bias),
            f32(weight if final_weight is None else final_weight), f32(bias if final_bias is None else final_bias))

    def test_isotropic_features_match_diagonal_covariance_reference(self):
        result = self.geometry([[-1, -1], [-1, 1], [1, -1], [1, 1]])["initial"]
        self.assertEqual(result["observed_centered_energy"], 2.)
        self.assertEqual(result["diagonal_covariance_reference_energy"], 2.)
        self.assertEqual(result["offdiagonal_covariance_energy"], 0.)
        self.assertEqual(result["observed_to_diagonal_energy_ratio"], 1.)

    def test_correlated_features_retain_signed_offdiagonal_energy(self):
        positive = self.geometry([[-1, -1], [1, 1]])["initial"]
        negative = self.geometry([[-1, 1], [1, -1]])["initial"]
        self.assertEqual(positive["observed_centered_energy"], 4.)
        self.assertEqual(positive["offdiagonal_covariance_energy"], 2.)
        self.assertEqual(positive["observed_to_diagonal_energy_ratio"], 2.)
        self.assertEqual(negative["observed_centered_energy"], 0.)
        self.assertEqual(negative["offdiagonal_covariance_energy"], -2.)
        self.assertEqual(negative["observed_to_diagonal_energy_ratio"], 0.)

    def test_offset_centered_and_bias_decomposition_is_analytical(self):
        result = self.geometry([[2, 4], [4, 6]], weight=((1, 2),), bias=(1,))
        initial = result["initial"]
        self.assertEqual(initial["mean_term_energy"], 196.)
        self.assertEqual(initial["observed_centered_energy"], 9.)
        self.assertEqual(initial["pre_energy"], 205.)
        self.assertEqual(initial["mean_centered_cross_term"], 0.)
        self.assertEqual(result["input"]["coordinate_mean_energy"], 17.)
        self.assertEqual(result["input"]["coordinate_std_energy"], 1.)
        self.assertEqual(result["input"]["raw_energy"], 18.)

    def test_identical_endpoints_have_exact_zero_difference(self):
        result = self.geometry([[1, 2], [3, 4]], weight=((1, 2), (3, 4)), bias=(2, 3))
        self.assertEqual(result["initial"], result["final"])
        for key in ("weight_frobenius", "bias_l2", "pre_delta_energy", "weight_term_energy", "bias_term_energy", "weight_bias_cross_term"):
            self.assertEqual(result["difference"][key], 0.)
        self.assertAlmostEqual(result["difference"]["weight_cosine_initial_final"], 1.)

    def test_linear_delta_includes_weight_bias_cancellation_not_percentages(self):
        result = self.geometry([[1, 2], [3, 4]], weight=((1, -1),), final_weight=((2, -1),), final_bias=(-2,))["difference"]
        self.assertEqual(result["weight_term_energy"], 5.)
        self.assertEqual(result["bias_term_energy"], 4.)
        self.assertEqual(result["weight_bias_cross_term"], -8.)
        self.assertEqual(result["pre_delta_energy"], 1.)
        self.assertEqual(result["linear_identity"]["max_abs_error"], 0.)
        self.assertEqual(result["energy_identity"]["max_abs_error"], 0.)

    def test_zero_denominator_returns_none_and_constant_input_has_no_variance(self):
        result = self.geometry([[3, 7]], weight=((0, 0),), bias=(2,))
        self.assertEqual(result["initial"]["pre_rms"], 2.)
        self.assertEqual(result["initial"]["centered_rms"], 0.)
        self.assertIsNone(result["initial"]["observed_to_diagonal_energy_ratio"])
        self.assertIsNone(result["difference"]["weight_cosine_initial_final"])
        self.assertIsNone(result["difference"]["weight_frobenius_relative_to_initial"])

    def test_weight_norm_and_negative_direction_cosine(self):
        result = self.geometry([[1, 2]], weight=((3, 4), (0, 0)), bias=(0, 0), final_weight=((-3, -4), (0, 0)))
        initial, difference = result["initial"], result["difference"]
        self.assertEqual(initial["weight_frobenius"], 5.)
        self.assertEqual(initial["weight_row_norm_mean"], 2.5)
        self.assertEqual(initial["weight_row_norm_std"], 2.5)
        self.assertEqual(initial["weight_row_norm_min"], 0.)
        self.assertEqual(initial["weight_row_norm_max"], 5.)
        self.assertEqual(difference["weight_frobenius"], 10.)
        self.assertEqual(difference["weight_cosine_initial_final"], -1.)

    def test_readonly_noncontiguous_inputs_unchanged_and_output_only_json_scalars(self):
        x = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
        weight = np.arange(6, dtype=np.float32).reshape(2, 3)
        bias = f32([1, -1])
        for value in (x, weight, bias): value.flags.writeable = False
        before = [value.tobytes() for value in (x, weight, bias)]
        result = projection_geometry(x, weight, bias, -weight, -bias)
        self.assertEqual([value.tobytes() for value in (x, weight, bias)], before)
        encoded = json.dumps(result, allow_nan=False)
        self.assertNotIn("tanh", encoded)
        self.assertFalse(result["actual_float32_activations"])
        self.assertEqual(result["model_forwards"], 0)
        self.assertEqual(result["counts"], dict(rows=4, features=3, outputs=2))

    def test_rejects_nonfinite_wrong_dtype_shape_and_nonarray(self):
        good = [f32([[1, 2], [3, 4]]), f32([[1, 1]]), f32([0]), f32([[1, 1]]), f32([0])]
        for index in range(5):
            for mode in ("nan", "inf", "dtype", "nonarray"):
                values = [v.copy() for v in good]
                if mode == "dtype": values[index] = values[index].astype(np.float64)
                elif mode == "nonarray": values[index] = values[index].tolist()
                else: values[index].flat[0] = float(mode)
                with self.subTest(index=index, mode=mode), self.assertRaises(ValueError):
                    projection_geometry(*values)
        for index, value in ((0, f32([])), (0, np.empty((0, 2), np.float32)), (1, f32([[1, 2, 3]])),
                             (2, f32([[0]])), (3, f32([[1, 2], [3, 4]])), (4, f32([0, 0]))):
            values = list(good)
            values[index] = value
            with self.subTest(index=index, shape=value.shape), self.assertRaises(ValueError): projection_geometry(*values)

    def test_multidimensional_random_identity_errors_stay_within_recorded_tolerance(self):
        rng = np.random.default_rng(19)
        x = rng.normal(2, 3, (17, 19)).astype(np.float32)
        weight = rng.normal(0, .5, (7, 19)).astype(np.float32)
        bias = rng.normal(0, 1, 7).astype(np.float32)
        result = projection_geometry(x, weight, bias, weight * 2, bias - 2)
        for section in ("initial", "final", "difference"):
            for key in ("linear_identity", "energy_identity"):
                self.assertLessEqual(result[section][key]["max_abs_error"], result[section][key]["tolerance"])


if __name__ == "__main__":
    unittest.main()
