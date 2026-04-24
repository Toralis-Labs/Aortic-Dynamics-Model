from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import numpy as np

EPS = 1.0e-12


def as_point(value: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(value), dtype=float).reshape(3)
    return arr


def unit(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n <= EPS:
        return np.zeros_like(v, dtype=float)
    return v / n


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def polyline_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def cumulative_arclength(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    if pts.shape[0] == 1:
        return np.zeros((1,), dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def point_at_arclength(points: np.ndarray, target_s: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return np.zeros(3, dtype=float)
    if pts.shape[0] == 1:
        return pts[0].copy()
    s = cumulative_arclength(pts)
    target = float(np.clip(target_s, 0.0, float(s[-1])))
    idx = int(np.searchsorted(s, target, side="right") - 1)
    idx = max(0, min(idx, pts.shape[0] - 2))
    denom = float(s[idx + 1] - s[idx])
    if denom <= EPS:
        return pts[idx].copy()
    t = float((target - s[idx]) / denom)
    return (1.0 - t) * pts[idx] + t * pts[idx + 1]


def tangent_at_arclength(points: np.ndarray, target_s: float, window: float = 1.0) -> np.ndarray:
    total = polyline_length(points)
    if total <= EPS:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    a = point_at_arclength(points, max(0.0, float(target_s) - float(window)))
    b = point_at_arclength(points, min(total, float(target_s) + float(window)))
    t = unit(b - a)
    if float(np.linalg.norm(t)) <= EPS and len(points) >= 2:
        t = unit(np.asarray(points[-1], dtype=float) - np.asarray(points[0], dtype=float))
    return t


def concatenate_polylines(parts: list[np.ndarray]) -> np.ndarray:
    merged: list[np.ndarray] = []
    for part in parts:
        pts = np.asarray(part, dtype=float)
        if pts.shape[0] == 0:
            continue
        if not merged:
            merged.extend([p.copy() for p in pts])
            continue
        start = pts[0]
        if distance(merged[-1], start) <= 1.0e-8:
            pts = pts[1:]
        merged.extend([p.copy() for p in pts])
    if not merged:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(merged, dtype=float)


def orthonormal_frame(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = unit(normal)
    if float(np.linalg.norm(n)) <= EPS:
        n = np.array([0.0, 0.0, 1.0], dtype=float)
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(ref, n))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = unit(np.cross(n, ref))
    v = unit(np.cross(n, u))
    return u, v


def polygon_area_normal(points: np.ndarray) -> Tuple[float, np.ndarray, float]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return 0.0, np.zeros(3, dtype=float), float("inf")
    accum = np.zeros(3, dtype=float)
    for i in range(pts.shape[0]):
        p = pts[i]
        q = pts[(i + 1) % pts.shape[0]]
        accum += np.cross(p, q)
    area = 0.5 * float(np.linalg.norm(accum))
    normal = unit(accum)
    centroid = np.mean(pts, axis=0)
    rms = float(np.sqrt(np.mean(np.square((pts - centroid) @ normal)))) if area > EPS else float("inf")
    return area, normal, rms


def projected_major_minor_diameters(points: np.ndarray, normal_hint: Optional[np.ndarray] = None) -> Tuple[Optional[float], Optional[float]]:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return None, None
    center = np.mean(pts, axis=0)
    if normal_hint is not None and float(np.linalg.norm(normal_hint)) > EPS:
        u, v = orthonormal_frame(np.asarray(normal_hint, dtype=float))
        xy = np.column_stack([(pts - center) @ u, (pts - center) @ v])
    else:
        _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
        xy = np.column_stack([(pts - center) @ vh[0], (pts - center) @ vh[1]])
    cov = np.cov(xy.T)
    if not np.all(np.isfinite(cov)):
        return None, None
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    xy2 = xy @ eigvecs[:, order]
    major = float(np.max(xy2[:, 0]) - np.min(xy2[:, 0]))
    minor = float(np.max(xy2[:, 1]) - np.min(xy2[:, 1]))
    if not math.isfinite(major) or not math.isfinite(minor):
        return None, None
    return major, minor


def equivalent_diameter_from_area(area: float) -> Optional[float]:
    area_f = float(area)
    if not math.isfinite(area_f) or area_f <= 0.0:
        return None
    return float(math.sqrt(4.0 * area_f / math.pi))

