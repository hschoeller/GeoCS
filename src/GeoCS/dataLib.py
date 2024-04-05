#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:10:26 2024

@author: schoelleh96
"""

from . import plotLib as pp
from . import calcLib as cc
from typing import Optional, List
from datetime import datetime
import numpy as np
import cartopy
from abc import ABC, abstractmethod
import os
import scipy.sparse as sps

# %%

class Data(ABC):
    """
    Abstract base class for all kinds of data in this package.

    Attributes:
        _dataPath (str): Path where the data is stored or to be stored.
        _startDate (datetime): The start date of the trajectories.
        _nTraj (Optional[int]): Number of trajectories, initialized to None.
        _nSteps (Optional[int]): Number of time steps, initialized to None.
        _dt (Optional[datetime]): Time step size, initialized to None.
    """
    def __init__(self, dataPath: str, startDate: datetime):
        self._dataPath = dataPath
        self._startDate = startDate
        self._nTraj: Optional[int] = None
        self._nSteps: Optional[int] = None
        self._dt: Optional[datetime] = None

    def __str__(self) -> str:
        dateStr: str = self._startDate.strftime("%Y-%m-%d %H:%M:%s")
        return (f"{dateStr}, Number of Trajectories: {self._nTraj}, "
                f"Number of Steps: {self._nSteps}, Stepsize: {self._dt}")

    @abstractmethod
    def load(self) -> None:
        """Load data. Implementation required."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Save data. Implementation required."""
        pass

    @abstractmethod
    def plot(self) -> None:
        """Plot data. Implementation required."""
        pass

    @property
    def dataPath(self) -> str:
        return self._dataPath

    @dataPath.setter
    def dataPath(self, value: str) -> None:
        self._dataPath = value

    @property
    def startDate(self) -> datetime:
        return self._startDate

    @startDate.setter
    def startDate(self, value: datetime) -> None:
        if not isinstance(value, datetime):
            raise TypeError("startDate must be a datetime object")
        self._startDate = value

    @property
    def dt(self):
        return self._dt

    @property
    def nTraj(self):
        return self._nTraj

    @property
    def nSteps(self):
        return self._nSteps

# %%

class TrajData(Data):
    """
    A class for handling trajectory data.

    Attributes:
        _dataPath (str): The path where the data is stored or to be stored.
        _startDate (datetime): The start date of the trajectories.
        _extent (Optional[List[float]]): The axes extent for plotting.
            Defaulting to the entire globe ([-180, 180, -90, 90]).
        _projection (cartopy.crs.Projection): Map projection used for plotting.
            Defaulting to Mercator projection.
        _trajs (Optional[np.ndarray]): The trajectory data as a NumPy array.
        _k (Optional[float]): Empirical scaling parameter.
    """

    def __init__(self, dataPath: str, startDate: datetime):
        super().__init__(dataPath, startDate)
        self._trajs: Optional[np.ndarray] = None
        self._k : Optional[float] = None
        self._extent: Optional[List] = [-180, 180, -90, 90]
        self._projection: Optional[cartopy.crs] = cartopy.crs.Mercator()

    def __str__(self) -> str:
        parentStr = super().__str__()
        return ("Trajectory Data object; " + parentStr)

    def load(self) -> None:
        """
        Load trajectory data (npy) from file specified in dataPath

        Returns
        -------
        None.

        """
        self._trajs = np.load(self._dataPath)
        self._get_properties()

    def _get_properties(self) -> None:
        self._nTraj, self._nSteps = self._trajs.shape
        self._dt = self._trajs['time'][0,1] - self._trajs['time'][0,0]

    def save(self) -> None:
        """
        Saves trajectory data (npy) to file specified in dataPath

        Returns
        -------
        None.

        """
        np.save(self.dataPath, self.trajs)

    @property
    def extent(self):
        return self._extent

    @extent.setter
    def extent(self, newExtent: List) -> None:
        self._extent = newExtent

    @property
    def projection(self):
        return self._projection

    @projection.setter
    def projection(self, newProjection: cartopy.crs.Projection) -> None:
        self._projection = newProjection

    @property
    def trajs(self):
        return self._trajs

    @trajs.setter
    def trajs(self, newTrajs: np.ndarray) -> None:
        self._trajs = newTrajs
        self._get_properties()

    @property
    def k(self):
        """
        Scaling parameter. Assumes U and V are in m/s and Omega is in P/s.

        Returns
        -------
        float
            In km/hPa.

        """
        if self._k is None:
            self._k = cc.calc_k(self._trajs['U']/1000, self._trajs['V']/1000,
                                self._trajs['OMEGA']/100)
        return self._k

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
        fig, ax = self.plot_2d(**kwargs)
        return fig, ax

    def plot_2d(self, **kwargs):
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
        fig, ax = pp.plot_traj_2d(self.trajs, self._projection,
                                     self._extent,
                                     **kwargs)
        return fig, ax

# %%

class DistData(Data):
    """
    A class for handling pairwise distances of trajectories.

    Attributes:
        _dataPath (str): Path where the  data is stored or to be stored.
        _startDate (datetime): The start date of the trajectories.
        _r (float): The cut-off radius for distance calculations.
        _k (float): vertical scaling parameter
        _savePattern (str): The pattern used for saving distance matrix files,
            with datetime formatting symbols (e.g., "%Y%m%d_%H%M%S.npz").
        _mats (Dict[datetime, sps.csc_matrix]): A dictionary mapping each
            timestep to its corresponding sparse distance matrix triangle.
        _r (float): The cut-off radius for distance calculations.
        _matPaths (List[str]): The list of file paths for the matrices.
        _trajData (Optional[TrajData]): An optional TrajData object from which
            pairwise distances can be calculated if not loading.
    """

    def __init__(self, dataPath: str, r: float, k: float,
                 startDate: Optional[datetime] = None,
                 trajData: Optional[TrajData] = None,
                 savePattern: Optional[str] = "%Y%m%d_%H%M.npz"):
        self._savePattern = savePattern
        self._mats = {}  # Initializing here to ensure it's unique per instance
        self._r = r
        self._k = k
        if startDate is not None:
            super().__init__(dataPath, startDate)
            self._matPaths = os.listdir(self._dataPath)
            # Assuming matPaths are named according to savePattern
            self._nTraj = None
            self._nSteps = len(self._matPaths)
            self._dt = (datetime.strptime(self._matPaths[1], savePattern) -
                        datetime.strptime(self._matPaths[0], savePattern))
            self._trajData = None
        elif trajData is not None:
            super().__init__(dataPath, trajData.startDate)
            self._nTraj = trajData.nTraj
            self._nSteps = trajData.nSteps
            self._dt = trajData.dt
            self._trajData = trajData
            # Generate matPaths based on trajData timing
            self._matPaths = [d.astype(datetime).strftime(savePattern)
                              for d in np.unique(trajData._trajs['time'])]
        else:
            raise ValueError("Either startDate or trajData must be provided.")

    def __str__(self) -> str:
        parentStr = super().__str__()
        return ("Distance Data object; " + parentStr)

    def load(self) -> None:
        ''' Load all available distance matrices. Caution for large data!'''
        for mp in self._matPaths:
            fullPath = os.path.join(self._dataPath, mp)  # Ensure full path
            dateKey = datetime.strptime(mp, self._savePattern)
            self._mats[dateKey] = self.load_mat(fullPath)

    def load_mat(self, matPath: str) -> sps.csc_matrix:
        return sps.load_npz(matPath)

    def save(self) -> None:
        ''' Save distance matrix for all dates. Caution for large data!'''
        for mp in self._matPaths:
            dateKey = datetime.strptime(mp, self._savePattern)
            if dateKey in self._mats:
                self.save_mat(self._mats[dateKey], mp)
            else:
                distMat = self.calc_dist(dateKey)
                self.save_mat(distMat, mp)

    def save_mat(self, mat: sps.csc_matrix, matPath: str) -> None:
        fullPath = os.path.join(self._dataPath, matPath)  # Ensure full path
        sps.save_npz(fullPath, mat)

    def calc_dist(self, dateKey: Optional[datetime] = None,
                 timestep: Optional[int] = None) -> sps.csc_matrix:
        if dateKey is not None:
            dateKey = np.datetime64(dateKey)
            # Find the index where 'time' matches the timestep
            index = np.where(self._trajData.trajs['time'] == dateKey)[1]
            if index.size > 0:
                column = self._trajData.trajs[:, index[0]]
            else:
                print(f"No data found for {timestep}")
        elif timestep is not None:
            column = self._trajData.trajs[:, timestep]
        else:
            raise ValueError("Either a datetime or an integer index " +
                             "must be provided.")
        mat = cc.calc_dist(column['lon'], column['lat'], column['p'],
                    self._r, self._k)
        return mat

    @property
    def savePattern(self) -> str:
        return self._savePattern

    @savePattern.setter
    def savePattern(self, value: str) -> None:
        self._savePattern = value

    @property
    def r(self) -> float:
        return self._r

    @r.setter
    def r(self, value: float) -> None:
        self._r = value

    @property
    def k(self) -> float:
        return self._k

    @k.setter
    def k(self, value: float) -> None:
        self._k = value

    @property
    def mats(self) -> dict:
        return self._mats

    @property
    def matPaths(self) -> list:
        return self._matPaths

    @property
    def trajData(self) -> Optional[TrajData]:
        return self._trajData


    def plot(self, **kwargs):
        """
        Default Distances plot. Invokes plotDistHist.

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        fig : matplotlib figure
        ax : matplotlib ax

        """
        fig, ax = self.plot_dist_hist(**kwargs)
        return fig, ax

    def plot_dist_hist(self, binCount: Optional[int] = 100, **kwargs):
        """
        Plots histogram of distances.

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        fig : matplotlib figure
        ax : matplotlib ax

        """

        binEdges = np.linspace(0, self._r, binCount)
        histCounts = {}
        for mp in self._matPaths:
            dateKey = datetime.strptime(mp, self._savePattern)
            if dateKey in self._mats:
                distMat = self._mats[dateKey]
            else:
                distMat = self.calc_dist(dateKey)

            counts, _ = np.histogram(distMat.data, bins=binEdges)
            histCounts[dateKey] = counts

        fig, ax = pp.plot_dist_hist(histCounts, binEdges, **kwargs)

        return fig, ax