#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:08:54 2024

@author: schoelleh96
"""

import numpy as np
from sklearn.neighbors import BallTree
import scipy.sparse as sps
from alphashape import alphashape
from typing import Optional, List, Tuple, Dict
from warnings import warn
from scipy.spatial import ConvexHull
from trimesh.base import Trimesh


def calc_k(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    """
    Calculate the velocity-based scaling parameter.

    Parameters
    ----------
    u : ndarray
        Horizontal velocity in the x direction.
    v : ndarray
        Horizontal velocity in the y direction.
    w : ndarray
        Vertical velocity.

    Returns
    -------
    float
        Scaling parameter.
    """
    # Calculate the magnitude of the horizontal velocity
    u_h = np.sqrt(u**2 + v**2)
    # Return the scaling parameter
    return u_h.mean() / np.abs(w).mean()

def calc_dist(lon: np.ndarray, lat: np.ndarray, z: np.ndarray,
              r: float, k: float):
    """
    Calculate pointwise distances given positions on earth.

    Parameters
    ----------
    lon : ndarray
        longitudes.
    lat : ndarray
        latitudes.
    z : ndarray
        vertical coordinate.
    r : float
        cut-off radius in km.
    k : float
        scaling parameter bringing vertical coordinate to horizontal coordinate
        value range.

    Returns
    -------
    scipy.sparse.csc_matrix
        lower triangle of point-wise distance matrix.

    """
    # Calculate horizontal distances, if horizontal distance > r, 3d
    # distance will be > r, too
    dd = np.array([np.deg2rad(lon), np.deg2rad(lat)]).T
    BT = BallTree(dd, metric='haversine')
    idx, hdist = BT.query_radius(dd, r=r / 6371, return_distance=True)
    hdist = hdist * 6371
    # each element in idx/hdist corresponds to a point whose NN has
    # been queried
    x = list()
    y = list()
    v = list()

    for i in range(lon.shape[0]):
        # Save only one triangle of symmetric matrix
        hdist[i] = hdist[i][idx[i] > i]
        idx[i] = idx[i][idx[i] > i]

        vdist = z[idx[i]] - z[i]

        dist = np.sqrt(np.power(hdist[i], 2) + k * np.power(vdist, 2))

        # Now with the custom distance
        valid = np.where(dist < r)[0]
        x.extend([i] * len(valid))
        y.extend(idx[i][valid])
        v.extend(dist[valid])

    return sps.csc_matrix((np.asarray(v), (np.asarray(x), np.asarray(y))),
                          shape=(lon.shape[0],lon.shape[0]))

def calc_bounds(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                timesteps: np.ndarray, convex: bool,
                alpha: Optional[float] = None) -> Tuple[
                    Dict[np.datetime64, np.ndarray],
                    Dict[np.datetime64, Trimesh]]:
    """
    If Convex=True, find Convex Hull, else calculate alpha shape for given
    alpha or estimate optimal alpha else.

    Parameters
    ----------
    x : np.ndarray
        coordinate.
    y : np.ndarray
        coordinate.
    z : np.ndarray
        coordinate.
    timesteps : np.ndarray
        timesteps belonging to axis 1 of the coords.
    convex : bool
        whether to find convex of concave bounding hull.
    alpha : Optional[float], optional
        alpha parameter. The default is None.

    Returns
    -------
    Tuple[dict]
        1. bounds: Mapping from timesteps to boundary flags
            (True if inside or on boundary).
        2. hulls: Mapping from timesteps to the Trimesh object representing
            the hull.
    """
    def opt_alpha(alpha_0: float, points: List[Tuple[float, float, float]],
                  max_iter: int, max_no_change: int):
        '''
        Find the highes alpha shape parameter that leads to a hull
        that contains all points

        Parameters
        ----------
        alpha_0 : float
            initial alpha.
        points : list of tupel
            the points.
        max_iter : int
            maximum number of iterations.
        max_no_change : int
            maximum number of iterations showing no change in the best alpha.

        Returns
        -------
        best alpha : float
            the best alpha found.

        '''

        best_alpha = alpha_0
        i_best_alpha = 0  # Index at which the best alpha was found
        alpha = alpha_0  # Initialize alpha with the starting value
        no_improvement_streak = 0

        for i in range(max_iter):
            ashp = alphashape(points.tolist(), alpha)
            if not ashp.faces:  # Check if alphashape is degenerate
                out_no_bound = float('inf')
            else:
                bound_arr = np.array([point for point in ashp.exterior.coords]) if hasattr(ashp, 'exterior') else ashp.points
                boundary = np.asarray([tuple(p) in bound_arr for p in points])
                in_out = ashp.contains(points)  # Check if points are inside the shape
                out_no_bound = (~in_out & ~boundary).sum()

            if out_no_bound > 0:
                alpha *= np.sqrt(0.1)  # Decrease alpha if there are points outside
            else:
                if alpha > best_alpha:
                    best_alpha = alpha
                    i_best_alpha = i
                    no_improvement_streak = 0  # Reset no improvement streak
                else:
                    no_improvement_streak += 1  # Increment no improvement streak

            if no_improvement_streak > max_no_change:
                break  # Exit if no improvement in alpha for a while

            alpha = best_alpha + best_alpha * np.sqrt(0.1)  # Try a larger alpha

        return best_alpha
    ###
    def get_ind(x, z):
        y = np.empty_like(x.flatten())
        for i, xx in enumerate(x.flatten()):
            y[i] = np.where(z == xx)[0]
        return y.reshape(x.shape)
    ###

    bounds, hulls = {}, {}

    for t, timestep in enumerate(timesteps):
        current_alpha = alpha[t] if isinstance(alpha, np.ndarray) else alpha
        points = np.column_stack((x[:, t], y[:, t], z[:, t]))

        if not convex:
            if current_alpha is None:
                current_alpha = opt_alpha(0.01, points, 100, 10)
            print(f"alpha={current_alpha:.3E}")
            alpha_shape = alphashape(points.tolist(), current_alpha)

            if hasattr(alpha_shape, "vertices"):
                bound = np.any(np.all(points[:, None] ==
                                      alpha_shape.vertices.__array__(),
                                      axis=-1), axis=1)
                hull = alpha_shape
            elif hasattr(alpha_shape, "boundary"):
                warn("Alpha shape is a 2D polygon; points likely lie on a 2D surface. Returning Convex Hull.")
                hull = ConvexHull(points)
                hull = Trimesh(vertices=hull.points, faces=hull.simplices)
                bound = np.isin(range(len(points)), hull.vertices)
        else:
            hull = ConvexHull(points)
            bound = np.isin(np.arange(len(points)), hull.vertices)
            hull = Trimesh(vertices=hull.points, faces=hull.simplices)

        bounds[timestep], hulls[timestep] = bound, hull

    return bounds, hulls