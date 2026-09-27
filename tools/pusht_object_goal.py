"""RGB-only object/goal representation for the native colored Push-T task.

This is a task-specific perception/scoring interface, NOT a trained predictor,
not a generic segmentation method and not compatible with old ResNet latents.
No simulator pose, segmentation, coverage, reward or future lookup is read here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

SPACE_ID = "pusht_rgb_gray_T_occupancy24_v1"
PLAN = {
    "schema": SPACE_ID, "rgb_shape": [96, 96, 3], "grid_shape": [24, 24],
    "pixel_rule": {"red_min": 40, "channel_max": 225, "blue_minus_red_min": -10,
                   "blue_minus_red_max": 70, "abs_2green_minus_red_minus_blue_max": 40},
    "component_rule": "largest_8_connected_non_border_component",
    "minimum_visible_pixels": 16, "pooling": "nonoverlapping4x4_binary_mask_mean",
    "coordinates": "normalized_pixel_centers_X_right_Y_down",
    "orientation": "implicit_in_spatial_T_silhouette_no_ambiguous_PCA_angle",
    "goal_source": "gray_object_in_same_fixed_train_episode1_frame117_index278_RGB",
    "score": "sum((P-G)^2)/(sum(P^2)+sum(G^2))_terminal_only",
    "score_name": "quadratic_soft_Dice_loss", "horizon": 8, "candidates": 5,
    "ties": "retain_reference_lowest_index", "shape_scale_translation_alignment": "none",
    "simulator_truth_used": False, "thresholds_fitted_to_coverage": False,
    "detection_validity_is_calibrated_confidence": False,
    "compatible_trained_dynamics_available": False,
}


@dataclass(frozen=True)
class ObjectGridFeatures:
    values: np.ndarray  # current/goal [B,24,24]; future [B,5,8,24,24]
    space_id: str = SPACE_ID


@dataclass(frozen=True)
class ObjectObservation:
    features: ObjectGridFeatures
    mask96: np.ndarray
    centroid_xy: np.ndarray  # [2], NaN when invalid; reporting/optional future interface
    visible_pixels: int
    valid: bool

    def relative_to(self, goal: "ObjectObservation") -> dict:
        if not self.valid or not goal.valid:
            return {"valid": False, "centroid_delta_xy": None, "visible_area_ratio": None}
        return {"valid": True, "centroid_delta_xy": (self.centroid_xy - goal.centroid_xy).tolist(),
                "visible_area_ratio": self.visible_pixels / goal.visible_pixels}


def observe_rgb(rgb: np.ndarray) -> ObjectObservation:
    """Extract one visible gray T without querying environment metadata.

    Largest component is a single-object Push-T assumption. Agent occlusion,
    anti-aliasing, compression and other render styles can alter the result;
    no hidden/occluded object pixels are reconstructed from simulator knowledge.
    """
    if not isinstance(rgb, np.ndarray) or rgb.shape != (96, 96, 3) or rgb.dtype != np.uint8:
        raise ValueError("requires uint8 native96 RGB, not BGR/features/privileged state")
    image = rgb.astype(np.int16)
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    p = PLAN["pixel_rule"]
    candidate = ((r >= p["red_min"]) & (image.max(-1) <= p["channel_max"])
                 & (b - r >= p["blue_minus_red_min"]) & (b - r <= p["blue_minus_red_max"])
                 & (np.abs(2 * g - r - b) <= p["abs_2green_minus_red_minus_blue_max"]))
    labels, count = ndimage.label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1], [0])))
    sizes[border] = 0
    selected = int(sizes.argmax())
    pixels = int(sizes[selected])
    valid = pixels >= PLAN["minimum_visible_pixels"]
    mask = (labels == selected) if valid else np.zeros((96, 96), dtype=bool)
    grid = mask.astype(np.float32).reshape(24, 4, 24, 4).mean(axis=(1, 3))
    yy, xx = np.nonzero(mask)
    centroid = np.array([(xx.mean() + .5) / 96, (yy.mean() + .5) / 96], dtype=np.float32) if valid else np.full(2, np.nan, dtype=np.float32)
    return ObjectObservation(ObjectGridFeatures(grid[None]), mask, centroid, pixels, valid)


@dataclass(frozen=True)
class ObjectGoalScores:
    costs: np.ndarray  # [B,5], invalid action candidates masked +inf
    selected_index: np.ndarray


def score_candidate_grids(predicted: ObjectGridFeatures, goal: ObjectGridFeatures,
                          candidate_valid: np.ndarray) -> ObjectGoalScores:
    """Score a future occupancy-grid predictor's native8-step outputs.

    The present world model does NOT output these grids. Recorded future RGB
    grids may exercise this API offline only; they are not planning predictions.
    Shape/space checks reject using old [9,512] features by padding or reshaping.
    """
    if predicted.space_id != SPACE_ID or goal.space_id != SPACE_ID:
        raise ValueError("requires the RGB-object occupancy space, not ACT/ResNet features")
    future, target = predicted.values, goal.values
    if target.ndim != 3 or target.shape[1:] != (24, 24):
        raise ValueError("goal must be [B,24,24]")
    batch = len(target)
    if future.shape != (batch, 5, 8, 24, 24):
        raise ValueError("predicted occupancy must be [B,5,8,24,24]")
    valid = np.asarray(candidate_valid)
    if valid.shape != (batch, 5) or valid.dtype != bool or not valid[:, 0].all():
        raise ValueError("valid candidates must retain each ACT reference")
    for values in (target, future[valid]):
        if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("occupancy must be finite floating values in0..1")
    if (target.sum((-2, -1)) <= 0).any():
        raise ValueError("goal object was not observed; no empty-goal fallback")
    terminal = future[:, :, -1].astype(np.float64)
    target64 = target[:, None].astype(np.float64)
    costs = ((terminal - target64) ** 2).sum((-2, -1)) / ((terminal ** 2).sum((-2, -1)) + (target64 ** 2).sum((-2, -1)))
    costs[~valid] = np.inf
    return ObjectGoalScores(costs, costs.argmin(axis=1))
