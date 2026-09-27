"""Pure NumPy projection geometry: descriptive float64 algebra, not model forwards.

The diagonal-covariance energy is a reference, not a feature permutation or
causal attribution. Signed cross terms permit cancellation; norms are not
contribution percentages. No tanh, file access, torch, model load or training.
"""
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _energy(value):
    return float(np.mean(np.square(value)))


def _rms(value):
    return float(np.sqrt(_energy(value)))


def _maximum(value):
    return float(np.max(np.abs(value)))


def _identity_error(actual, reconstructed, scale, name):
    error = _maximum(actual - reconstructed)
    tolerance = 1e-10 * (1.0 + float(scale))
    _require(np.isfinite(error) and error <= tolerance, name + " float64 identity failed")
    return dict(max_abs_error=error, tolerance=tolerance)


def projection_geometry(x, initial_weight, initial_bias, final_weight, final_bias):
    """Return scalar-only endpoint geometry for float32 X[N,D], W[h,D], b[h].

    Each row of X is counted once as supplied; the caller owns split, sample
    weighting and checkpoint provenance. Float64 preactivations here are NOT
    actual float32 activations. No inputs are modified or returned by reference.
    """
    source = (x, initial_weight, initial_bias, final_weight, final_bias)
    _require(all(isinstance(a, np.ndarray) and not isinstance(a, np.matrix)
                 and a.dtype == np.float32 and np.isfinite(a).all() for a in source),
             "all inputs must be finite float32 NumPy arrays")
    _require(x.ndim == 2 and min(x.shape) > 0 and initial_weight.ndim == 2
             and initial_weight.shape[0] > 0 and initial_weight.shape[1] == x.shape[1]
             and final_weight.shape == initial_weight.shape
             and initial_bias.shape == final_bias.shape == (initial_weight.shape[0],),
             "expected nonempty X[N,D], matching W[h,D] and b[h]")
    values, w0, b0, wf, bf = (np.asarray(a, dtype=np.float64) for a in source)
    mean = values.mean(axis=0)
    centered = values - mean
    variance = np.mean(centered * centered, axis=0)

    def endpoint(weight, bias):
        pre = values @ weight.T + bias
        mean_term = mean @ weight.T + bias
        centered_term = centered @ weight.T
        reconstruction = _identity_error(pre, mean_term + centered_term,
            _maximum(pre) + _maximum(mean_term) + _maximum(centered_term), "mean/centered")
        row_norms = np.linalg.norm(weight, axis=1)
        diagonal = float(np.sum(variance * np.sum(weight * weight, axis=0)) / weight.shape[0])
        observed = _energy(centered_term)
        cross = float(2 * np.mean(centered_term * mean_term))
        pre_energy, mean_energy = _energy(pre), _energy(mean_term)
        energy_error = _identity_error(np.asarray(pre_energy), np.asarray(mean_energy + observed + cross),
            pre_energy + mean_energy + observed + abs(cross), "endpoint energy")
        report = dict(weight_frobenius=float(np.linalg.norm(weight)),
            weight_row_norm_mean=float(row_norms.mean()), weight_row_norm_std=float(row_norms.std()),
            weight_row_norm_min=float(row_norms.min()), weight_row_norm_max=float(row_norms.max()),
            weight_row_norm_rms=_rms(row_norms), bias_rms=_rms(bias),
            pre_energy=pre_energy, pre_rms=float(np.sqrt(pre_energy)),
            mean_term_energy=mean_energy, mean_term_rms=float(np.sqrt(mean_energy)),
            observed_centered_energy=observed, centered_rms=float(np.sqrt(observed)),
            diagonal_covariance_reference_energy=diagonal,
            offdiagonal_covariance_energy=observed - diagonal,
            observed_to_diagonal_energy_ratio=observed / diagonal if diagonal > 0 else None,
            mean_centered_cross_term=cross, linear_identity=reconstruction, energy_identity=energy_error)
        return report, pre

    initial, z0 = endpoint(w0, b0)
    final, zf = endpoint(wf, bf)
    dw, db = wf - w0, bf - b0
    weight_term = values @ dw.T
    actual_delta = zf - z0
    linear = _identity_error(actual_delta, weight_term + db,
        _maximum(z0) + _maximum(zf) + _maximum(weight_term) + _maximum(db), "endpoint difference")
    weight_energy, bias_energy = _energy(weight_term), _energy(db)
    cross = float(2 * np.mean(weight_term * db))
    delta_energy = _energy(actual_delta)
    energy = _identity_error(np.asarray(delta_energy), np.asarray(weight_energy + bias_energy + cross),
        delta_energy + weight_energy + bias_energy + abs(cross), "difference energy")
    norm0, normf = initial["weight_frobenius"], final["weight_frobenius"]
    cosine = float(np.clip(np.sum(w0 * wf) / (norm0 * normf), -1., 1.)) if norm0 > 0 and normf > 0 else None
    result = dict(schema="pi05_libero_projection_geometry_v1",
        interpretation="float64 descriptive affine algebra; diagonal covariance reference, not permutation or causal shares",
        actual_float32_activations=False, model_forwards=0,
        counts=dict(rows=int(x.shape[0]), features=int(x.shape[1]), outputs=int(w0.shape[0])),
        input=dict(raw_energy=_energy(values), raw_rms=_rms(values),
            coordinate_mean_energy=_energy(mean), coordinate_mean_rms=_rms(mean),
            coordinate_std_energy=float(variance.mean()), coordinate_std_rms=float(np.sqrt(variance.mean()))),
        initial=initial, final=final,
        difference=dict(weight_frobenius=float(np.linalg.norm(dw)), bias_l2=float(np.linalg.norm(db)),
            bias_rms=_rms(db), weight_cosine_initial_final=cosine,
            weight_frobenius_relative_to_initial=float(np.linalg.norm(dw)) / norm0 if norm0 > 0 else None,
            pre_delta_energy=delta_energy, pre_delta_rms=float(np.sqrt(delta_energy)),
            weight_term_energy=weight_energy, weight_term_rms=float(np.sqrt(weight_energy)),
            bias_term_energy=bias_energy, bias_term_rms=float(np.sqrt(bias_energy)),
            weight_bias_cross_term=cross, linear_identity=linear, energy_identity=energy))

    def finite_scalars(value):
        if isinstance(value, dict):
            return all(finite_scalars(v) for v in value.values())
        return value is None or isinstance(value, (str, bool, int)) or (type(value) is float and np.isfinite(value))

    _require(finite_scalars(result), "geometry must return finite JSON scalars only")
    return result
