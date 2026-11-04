"""Geographic prediction metrics (coordinates are longitude, latitude)."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088
DISTANCE_THRESHOLDS_KM = (1, 10, 25, 100, 500, 2500)
TOP_K = (1, 5, 10)


def haversine_km(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    lon1, lat1 = np.radians(actual[..., 0]), np.radians(actual[..., 1])
    lon2, lat2 = np.radians(predicted[..., 0]), np.radians(predicted[..., 1])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def calculate_metrics(actual_coordinates, actual_paths, top_classes, hierarchy, centers):
    actual_paths = np.asarray(actual_paths, dtype=np.int64)
    top_classes = np.asarray(top_classes, dtype=np.int64)
    if len(top_classes) == 0:
        raise ValueError("No valid examples to evaluate")
    distances = haversine_km(actual_coordinates[:, None, :], centers[top_classes])
    errors = distances[:, 0]
    valid_distance = np.isfinite(errors)
    finite_errors = errors[valid_distance]

    def distance_stat(stat):
        return float(stat(finite_errors)) if len(finite_errors) else None
    true_fine = actual_paths[:, 0]
    summary = {
        "count": int(len(errors)),
        "mean_error_km": distance_stat(np.mean),
        "median_error_km": distance_stat(np.median),
        "p75_error_km": distance_stat(lambda values: np.percentile(values, 75)),
        "p90_error_km": distance_stat(lambda values: np.percentile(values, 90)),
        "valid_distance_count": int(valid_distance.sum()),
        "invalid_distance_count": int((~valid_distance).sum()),
        "accuracy_within_km": {
            str(threshold): float(np.mean(errors <= threshold))
            for threshold in DISTANCE_THRESHOLDS_KM
        },
        "hierarchy_top1_accuracy": [
            float(value) for value in np.mean(hierarchy[top_classes[:, 0]] == actual_paths, axis=0)
        ],
        "fine_cell_topk_accuracy": {},
        "best_topk_median_error_km": {},
    }
    for k in TOP_K:
        end = min(k, top_classes.shape[1])
        summary["fine_cell_topk_accuracy"][str(k)] = float(
            np.mean(np.any(top_classes[:, :end] == true_fine[:, None], axis=1))
        )
        # An invalid grid representative point must not poison every distance statistic.
        best = np.min(np.where(np.isfinite(distances[:, :end]), distances[:, :end], np.inf), axis=1)
        best = best[np.isfinite(best)]
        summary["best_topk_median_error_km"][str(k)] = float(np.median(best)) if len(best) else None
    return summary, distances
