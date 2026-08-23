import numpy as np


def normalize_angle(angle) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def euclidean_distance(p1, p2) -> float:
    return np.linalg.norm(np.array(p1) - np.array(p2))


def horizontal_distance(p1, p2) -> float:
    return np.sqrt((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2)


def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def round_coords(coords, decimals = 2):
    if isinstance(coords, np.ndarray):
        return np.round(coords, decimals)
    elif isinstance(coords, (list, tuple)):
        return [round(c, decimals) for c in coords]
    return round(coords, decimals)
