#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:08:54 2024

@author: schoelleh96
"""

from typing import Optional, List, Tuple, Dict, Callable
from warnings import warn, catch_warnings, simplefilter
from datetime import datetime
import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
from sklearn.neighbors import BallTree
from sklearn.cluster import KMeans
from alphashape import alphashape
from scipy.spatial import ConvexHull
from trimesh.base import Trimesh
import shapely as shp


def calc_k(u: np.typing.NDArray[float], v: np.typing.NDArray[float],
           w: np.typing.NDArray[float]) -> float:
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


def calc_dist(lon: np.typing.NDArray[float],
              lat: np.typing.NDArray[float], z: np.typing.NDArray[float],
              r: float, k: float) -> sps.csr_matrix:
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
    scipy.sparse.csr_matrix
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

        dist = np.sqrt(np.power(hdist[i], 2) + np.power(k * vdist, 2))

        # Now with the custom distance
        valid = np.where(dist < r)[0]
        x.extend([i] * len(valid))
        y.extend(idx[i][valid])
        v.extend(dist[valid])

    return sps.csr_matrix((np.asarray(v), (np.asarray(x), np.asarray(y))),
                          shape=(lon.shape[0], lon.shape[0]))


def calc_bounds(x: np.typing.NDArray[float], y: np.typing.NDArray[float],
                z: np.typing.NDArray[float],
                timesteps: np.typing.NDArray[datetime], convex: bool,
                alpha: Optional[float] = None) -> Tuple[
                    Dict[datetime, np.typing.NDArray[bool]],
                    Dict[datetime, Trimesh]]:
    """
    Calculate Boundary.

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
        """
        Find optimal alpha.

        Highest alpha that leads to a hull that contains all points.

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

        """
        best_alpha = alpha_0
        alpha = alpha_0  # Initialize alpha with the starting value
        no_improvement_streak = 0

        for i in range(max_iter):
            ashp = alphashape(points, alpha)
            if (ashp.faces.shape[0] == 0 or isinstance(
                    ashp, shp.geometry.polygon.Polygon)):
                # Check if alphashape is degenerate
                out_no_bound = float('inf')
            else:
                # Expand dimensions of points and vertices for broadcasting
                points_expanded = np.expand_dims(points, axis=1)
                vertices_expanded = np.expand_dims(ashp.vertices, axis=0)
                # Perform an element-wise comparison and then reduce
                matches = np.all(points_expanded == vertices_expanded, axis=2)
                is_boundary = np.any(matches, axis=1)
                inside = ashp.contains(points)
                out_no_bound = (~inside & ~is_boundary).sum()
                # out_no_bound are the number of points that are neither
                # inside nor on the boundary
            if out_no_bound > 0:
                alpha *= np.sqrt(0.1)
                no_improvement_streak += 1
            else:
                if alpha > best_alpha:
                    best_alpha = alpha
                    no_improvement_streak = 0
                else:
                    no_improvement_streak += 1
                alpha = best_alpha + best_alpha * np.sqrt(0.1)

            if no_improvement_streak > max_no_change:
                break  # Exit if no improvement in alpha for a while

        return best_alpha
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
                warn("Alpha shape is a 2D polygon; "
                     "points likely lie on a 2D surface."
                     " Returning Convex Hull.")
                hull = ConvexHull(points)
                hull = Trimesh(vertices=hull.points, faces=hull.simplices)
                bound = np.isin(range(len(points)), hull.vertices)
        else:
            hull = ConvexHull(points)
            bound = np.isin(np.arange(len(points)), hull.vertices)
            hull = Trimesh(vertices=hull.points, faces=hull.simplices)

        bounds[timestep], hulls[timestep] = bound, hull

    return bounds, hulls


def calc_diff_map(
    eps: float, is_bound: Dict[datetime, np.typing.NDArray[bool]],
    N_v: int, n_traj: int, dates: np.typing.NDArray[datetime],
    dist_mats: Optional[Dict[datetime, sps.csr_matrix]] = None,
    calc_dist: Optional[Callable[[datetime], sps.csr_matrix]] = None
) -> Tuple[np.typing.NDArray[float], np.typing.NDArray[float]]:
    """
    Calculate diffusion maps.

    Diffusion maps: eigenvectors of the averaged diffusion transition matrix)
                        along with its eigenvalues.

    Parameters
    ----------
    eps : float
        diffusion bandwidth.
    is_bound : Dict[datetime, np.typing.NDArray[bool]]
        indicates boundary points at each timestep.
    N_v : int
        how many eigenvalues and -vectors to compute.
    n_traj : int
        number of trajectories.
    dates : np.typing.NDArray[datetime]
        The timesteps of the trajectories.
    dist_mats : Optional[Dict[datetime, sps.csr_matrix]], optional
        Dictonary mapping dates to distance matrices. The default is None.
    calc_dist : Optional[Callable[[datetime], sps.csr_matrix]], optional
        Function handle to a function returning distance matrices given a date.
        The default is None.

    Returns
    -------
    vals : np.typing.NDArray[float]
        Eigenvalues.
    vecs : np.typing.NDArray[float]
        Eigenvectors (the diffusion maps).

    """
    Q = sps.csr_matrix((n_traj, n_traj))
    for d in dates:
        K = dist_mats[d] if dist_mats is not None else calc_dist(d)
        K.data = np.exp(-K.data**2/eps)
        K = K + sps.eye(n_traj, format='csr') + K.T
        K = K.multiply(1/K.sum(axis=1))
        K = K.multiply(sps.diags(1 / K.sum(axis=1).A1))
        # apply BCs
        is_bound_indices = is_bound[d].nonzero()[0]
        with catch_warnings():
            simplefilter("ignore")
            K[is_bound_indices, :] = 0
            K[:, is_bound_indices] = 0
        K.eliminate_zeros()

        Q = Q + K

    try:
        vals, vecs = sps.linalg.eigs(Q, k=N_v)
    except Exception as e:
        print(f"Eigs (sparse) failed with error {e}, using eig")
        vals, vecs = spl.eig(Q.toarray())
    vals = np.flip(np.sort(np.real(vals)))[:N_v]
    return (vals, np.real(vecs))


def kmeans(E_vecs: np.typing.NDArray[float], N_k: int) -> KMeans:
    """
    Cluster using kmeans algorithm from scikit-learn.

    Parameters
    ----------
    E_vecs : np.typing.NDArray[float]
        Coordinates which to cluster(eigenvectors).
    N_k : int
        Number of clusters.

    Returns
    -------
    KMeans
        kmeans object.

    """
    N_v = N_k-1
    kcluster = KMeans(n_clusters=N_k, n_init='auto').fit(E_vecs[:, 0:(N_v)])
    return kcluster
