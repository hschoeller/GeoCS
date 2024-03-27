#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:10:26 2024

@author: schoelleh96
"""

from .BaseData import Data
from ..operations import plotLib
from typing import Optional, List
from datetime import datetime
import numpy as np
import cartopy

class TrajData(Data):
    """Class for handling trajectory data"""
    nTraj: Optional[int] = None # Number of trajectories
    nSteps: Optional[int] = None # Number of timesteps
    dt: Optional[datetime] = None # Timestep size
    trajs: Optional[np.ndarray] = None # Trajectories
    # In Lon/Lat, used for plotting, default is whole Earth
    _extent: Optional[List] = [-180, 180, -90, 90]
    _projection: Optional[cartopy.crs] = cartopy.crs.Mercator()

    def __str__(self) -> str:
        parent_str : str = super().__str__()
        return (f"{parent_str}, Number of Trajectories: {self.nTraj}, "
                f"Number of Steps: {self.nSteps}, Stepsize: {self.dt}")

    def load(self) -> None:
        """
        Load trajectory data (npy) from file specified in loadDataPath

        Returns
        -------
        None.

        """
        self.trajs = np.load(self.loadDataPath)
        self.nTraj, self.nSteps = self.trajs.shape
        self.dt = self.trajs['time'][0,1] - self.trajs['time'][0,0]

    def save(self) -> None:
        """
        Saves trajectory data (npy) to file specified in saveDataPath

        Returns
        -------
        None.

        """
        np.save(self.saveDataPath, self.trajs)

    @property
    def extent(self):
        return self._extent

    @extent.setter
    def setExtent(self, extent: List) -> None:
        self._extent = extent

    @property
    def projection(self):
        return self._projection

    @projection.setter
    def setProjection(self, projection: cartopy.crs.Projection) -> None:

        self._projection = projection

    def plot(self, **kwargs):
        """
        Default Trajectory plot. Invokes plot2D.

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        fig : matplotlib figure
        ax : matplotlib ax

        """
        fig, ax = self.plot2D(**kwargs)
        return fig, ax

    def plot2D(self, **kwargs):
        """
        Simple 2D trajectory plot.

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        fig : matplotlib figure
        ax : matplotlib ax

        """
        fig, ax = plotLib.plotTraj2D(self.trajs, self._projection,
                                     self._extent,
                                     **kwargs)
        return fig, ax
