#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:08:54 2024

@author: schoelleh96
"""

import numpy as np
from sklearn.neighbors import BallTree
import scipy.sparse as sps

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

def calc_dist(lon, lat, z, r, k):
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