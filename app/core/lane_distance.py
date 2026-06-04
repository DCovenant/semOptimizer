"""Normalised distance of a point from a lane's stop line.

For each incoming-lane polygon we find its principal (travel) axis and project
points onto it, returning a value in [0, 1]: 0 at the stop line, 1 at the far
end of the lane. The stop-line end is the polygon end nearest the camera.

Geometry note: `_cameras_from_lights` in carla_intersection.py mounts each
camera at the signal pole near the junction, looking *outward toward the
approaching queue*. So the junction / stop line sits at the bottom of the frame
(largest image-y) and the queue recedes upward — hence the stop-line end is the
axis extreme with the greatest y.

This is the cheap, calibration-free estimate (any polygon >= 3 points). A metric
(metres) upgrade would swap this for a homography from a 4-corner lane quad to a
known 3.5 m-wide rectangle.
"""
import numpy as np


def _axis_for(polygon):
    """Return (stop_point, unit_vector_toward_far, length) for a polygon.

    The axis is the polygon's dominant direction (largest-variance, via the
    covariance eigenvector). Endpoints are the extreme vertex projections; the
    stop end is whichever endpoint has the larger y (closer to the camera).
    """
    pts = np.asarray(polygon, dtype=float)
    if len(pts) < 2:
        return None
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    # principal direction = eigenvector of largest eigenvalue of the cov matrix
    cov = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    proj = centred @ axis                         # scalar projections
    p_lo = centroid + axis * proj.min()
    p_hi = centroid + axis * proj.max()
    # stop line = the end nearest the camera = larger image-y
    stop, far = (p_lo, p_hi) if p_lo[1] >= p_hi[1] else (p_hi, p_lo)
    length = float(np.linalg.norm(far - stop))
    if length < 1e-6:
        return None
    unit = (far - stop) / length
    return stop, unit, length


def stop_line_distance(polygon, foot) -> float:
    """Normalised [0,1] distance of `foot` from the lane's stop line.

    0 = at the stop line (nearest the junction), 1 = far end of the lane.
    `polygon` is a list of (x, y); `foot` is an (x, y) image point.
    """
    axis = _axis_for(polygon)
    if axis is None:
        return 0.0
    stop, unit, length = axis
    t = float(np.dot(np.asarray(foot, dtype=float) - stop, unit)) / length
    return max(0.0, min(1.0, t))
