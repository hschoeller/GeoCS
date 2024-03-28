#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:08:54 2024

@author: schoelleh96
"""

import numpy as np

def calcK(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
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

