"""Calculate geographic distance and retrieval accuracy."""

import math


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two coordinates in kilometers."""
    radius = 6371.0088

    lat1, lon1, lat2, lon2 = map(
        math.radians, (lat1, lon1, lat2, lon2)
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    return 2 * radius * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def evaluate_matches(matches, latitude, longitude):
    """Calculate geographic error for each retrieved image."""
    for match in matches:
        match["distance_km"] = haversine_distance(
            latitude, longitude, match["latitude"], match["longitude"]
        )

    return matches


def accuracy_at_distance(distances, threshold_km):
    """Return the fraction of predictions within a distance threshold."""
    if not distances:
        return 0.0

    return sum(d <= threshold_km for d in distances) / len(distances)