#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:10:26 2024

@author: schoelleh96
"""

from . import plot_lib as pp
from . import calc_lib as cc
from typing import Optional, List, Tuple, Dict
from trimesh.base import Trimesh
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.widgets import (Slider, CheckButtons, RadioButtons,
                                Button, TextBox)
from datetime import datetime
import numpy as np
import cartopy
from abc import ABC, abstractmethod
import os
import scipy.sparse as sps
import pickle

# %%

class Data(ABC):
    """
    Abstract base class for all kinds of data in this package.

    Attributes:
        _data_path (str): Path where the data is stored or to be stored.
        _start_date (datetime): The start date of the trajectories.
        _n_traj (Optional[int]): Number of trajectories, initialized to None.
        _n_steps (Optional[int]): Number of time steps, initialized to None.
        _dt (Optional[datetime]): Time step size, initialized to None.
    """
    def __init__(self, data_path: str, start_date: datetime):
        self._data_path = data_path
        self._start_date = start_date
        self._n_traj: Optional[int] = None
        self._n_steps: Optional[int] = None
        self._dt: Optional[datetime] = None

    def __str__(self) -> str:
        dateStr: str = self._start_date.strftime("%Y-%m-%d %H:%M")
        return (f"{dateStr}, Number of Trajectories: {self._n_traj}, "
                f"Number of Steps: {self._n_steps}, Stepsize: {self._dt}")

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
    def data_path(self) -> str:
        return self._data_path

    @data_path.setter
    def data_path(self, value: str) -> None:
        self._data_path = value

    @property
    def start_date(self) -> datetime:
        return self._start_date

    @start_date.setter
    def start_date(self, value: datetime) -> None:
        if not isinstance(value, datetime):
            raise TypeError("start_date must be a datetime object")
        self._start_date = value

    @property
    def dt(self):
        return self._dt

    @property
    def n_traj(self):
        return self._n_traj

    @property
    def n_steps(self):
        return self._n_steps

# %%

class Traj(Data):
    """
    A class for handling trajectory data.

    Attributes:
        _data_path (str): The path where the data is stored or to be stored.
        _start_date (datetime): The start date of the trajectories.
        _extent (Optional[List[float]]): The axes extent for plotting.
            Defaulting to the entire globe ([-180, 180, -90, 90]).
        _projection (cartopy.crs.Projection): Map projection used for plotting.
            Defaulting to Mercator projection.
        _trajs (Optional[np.ndarray]): The trajectory data as a NumPy array.
        _k (Optional[float]): Empirical scaling parameter.
    """

    def __init__(self, data_path: str, start_date: datetime):
        super().__init__(data_path, start_date)
        self._trajs: Optional[np.ndarray] = None
        self._k : Optional[float] = None
        self._extent: Optional[List] = [-180, 180, -90, 90]
        self._projection: Optional[cartopy.crs] = cartopy.crs.Mercator()

    def __str__(self) -> str:
        parentStr = super().__str__()
        return ("Trajectory Data object; " + parentStr +
                f" k: {self._k}")

    def load(self) -> None:
        """
        Load trajectory data (npy) from file specified in data_path

        Returns
        -------
        None.

        """
        self._trajs = np.load(self._data_path)
        self._get_properties()

    def _get_properties(self) -> None:
        self._n_traj, self._n_steps = self._trajs.shape
        self._dt = self._trajs['time'][0,1] - self._trajs['time'][0,0]

    def save(self) -> None:
        """
        Saves trajectory data (npy) to file specified in data_path

        Returns
        -------
        None.

        """
        np.save(self.data_path, self.trajs)

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

    def plot(self, **kwargs)-> Tuple[mpl.figure.Figure,
                                     cartopy.mpl.geoaxes.GeoAxes]:
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

    def plot_2d(self, **kwargs)-> Tuple[mpl.figure.Figure,
                                      cartopy.mpl.geoaxes.GeoAxes]:
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

class Dist(Data):
    """
    A class for handling pairwise distances of trajectories.

    Attributes:
        _data_path (str): Path where the  data is stored or to be stored.
        _start_date (datetime): The start date of the trajectories.
        _r (float): The cut-off radius for distance calculations.
        _k (float): vertical scaling parameter
        _save_pattern (str): The pattern used for saving distance matrix files,
            with datetime formatting symbols (e.g., "%Y%m%d_%H%M%S.npz").
        _mats (Dict[datetime, sps.csc_matrix]): A dictionary mapping each
            timestep to its corresponding sparse distance matrix triangle.
        _r (float): The cut-off radius for distance calculations.
        _mat_paths (List[str]): The list of file paths for the matrices.
        _traj_data (Optional[traj_data]): An optional traj_data object from which
            pairwise distances can be calculated if not loading.
    """

    def __init__(self, data_path: str, r: float, k: float,
                 start_date: Optional[datetime] = None,
                 traj_data: Optional[Traj] = None,
                 save_pattern: Optional[str] = "%Y%m%d_%H%M.npz"):
        self._save_pattern = save_pattern
        self._mats = {}  # Initializing here to ensure it's unique per instance
        self._r = r
        self._k = k
        if start_date is not None:
            super().__init__(data_path, start_date)
            self._mat_paths = os.listdir(self._data_path)
            # Assuming matPaths are named according to save_pattern
            self._n_traj = None
            self._n_steps = len(self._mat_paths)
            self._dt = (datetime.strptime(self._mat_paths[1], save_pattern) -
                        datetime.strptime(self._mat_paths[0], save_pattern))
            self._traj_data = None
        elif traj_data is not None:
            super().__init__(data_path, traj_data.start_date)
            self._n_traj = traj_data.n_traj
            self._n_steps = traj_data.n_steps
            self._dt = traj_data.dt
            self._traj_data = traj_data
            # Generate matPaths based on traj_data timing
            self._mat_paths = [d.astype(datetime).strftime(save_pattern)
                              for d in np.unique(traj_data._trajs['time'])]
        else:
            raise ValueError("Either start_date or traj_data must be provided.")

        if not os.path.exists(self.data_path):
                os.makedirs(self._data_path)

    def __str__(self) -> str:
        parentStr = super().__str__()
        return ("Distance Data object; " + parentStr +
                f" k: {self._k}, r: {self._r}, ")

    def load(self) -> None:
        ''' Load all available distance matrices. Caution for large data!'''
        for mp in self._mat_paths:
            fullPath = os.path.join(self._data_path, mp)  # Ensure full path
            date_key = datetime.strptime(mp, self._save_pattern)
            self._mats[date_key] = self.load_mat(fullPath)

    def load_mat(self, matPath: str) -> sps.csc_matrix:
        return sps.load_npz(matPath)

    def save(self) -> None:
        ''' Save distance matrix for all dates. Caution for large data!'''
        for mp in self._mat_paths:
            date_key = datetime.strptime(mp, self._save_pattern)
            if date_key in self._mats:
                self.save_mat(self._mats[date_key], mp)
            else:
                dist_mat = self.calc_dist(date_key)
                self.save_mat(dist_mat, mp)

    def save_mat(self, mat: sps.csc_matrix, matPath: str) -> None:
        fullPath = os.path.join(self._data_path, matPath)  # Ensure full path
        sps.save_npz(fullPath, mat)

    def calc_dist(self, date_key: Optional[datetime] = None,
                 timestep: Optional[int] = None) -> sps.csc_matrix:
        if date_key is not None:
            date_key = np.datetime64(date_key)
            # Find the index where 'time' matches the timestep
            index = np.where(self._traj_data.trajs['time'] == date_key)[1]
            if index.size > 0:
                column = self._traj_data.trajs[:, index[0]]
            else:
                print(f"No data found for {timestep}")
        elif timestep is not None:
            column = self._traj_data.trajs[:, timestep]
        else:
            raise ValueError("Either a datetime or an integer index " +
                             "must be provided.")
        mat = cc.calc_dist(column['lon'], column['lat'], column['p'],
                    self._r, self._k)
        return mat

    @property
    def save_pattern(self) -> str:
        return self._save_pattern

    @save_pattern.setter
    def save_pattern(self, value: str) -> None:
        self._save_pattern = value

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
    def mat_paths(self) -> list:
        return self._mat_paths

    @property
    def traj_data(self) -> Optional[Traj]:
        return self._traj_data


    def plot(self, **kwargs) -> Tuple[mpl.figure.Figure, mpl.axes._axes.Axes]:
        """
        Default Distances plot. Invokes plot_dist_hist.

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

    def plot_dist_hist(self, bin_count: Optional[int] = 100,
                       **kwargs) -> Tuple[mpl.figure.Figure,
                                          mpl.axes._axes.Axes]:
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

        bin_edges = np.linspace(0, self._r, bin_count)
        hist_counts = {}
        for mp in self._mat_paths:
            date_key = datetime.strptime(mp, self._save_pattern)
            if date_key in self._mats:
                dist_mat = self._mats[date_key]
            else:
                dist_mat = self.calc_dist(date_key)

            counts, _ = np.histogram(dist_mat.data, bins=bin_edges)
            hist_counts[date_key] = counts

        fig, ax = pp.plot_dist_hist(hist_counts, bin_edges, **kwargs)

        return fig, ax

# %%

class Bound(Data):

    def __init__(self, data_path: str, k: float, convex: bool,
                 alpha: Optional[float] = None,
                 start_date: Optional[datetime] = None,
                 traj_data: Optional[Traj] = None):
        self._hulls = {}
        self._is_bound = {}
        self._k = k
        self._convex = convex
        self._alpha = alpha
        self._projection: Optional[cartopy.crs] = cartopy.crs.Stereographic(
            central_latitude=90, true_scale_latitude=50)
        self._dict_path = (
                        f"{data_path}{'convex' if convex else 'concave'}"
                        f"{f'{alpha}' if alpha is not None else ''}"
                    ).strip()
        if start_date is not None:
            super().__init__(data_path, start_date)
            self._n_traj = None
            self._n_steps = None
            self._dt = None
            self._traj_data = None
        elif traj_data is not None:
            super().__init__(data_path, traj_data.start_date)
            self._n_traj = traj_data.n_traj
            self._n_steps = traj_data.n_steps
            self._dt = traj_data.dt
            self._traj_data = traj_data
            # transform horizontal coords
            transform = self._projection.transform_points(
                cartopy.crs.PlateCarree(), self._traj_data.trajs['lon'],
                self._traj_data.trajs['lat'])
            self._x, self._y = transform[:,:,0]/1e3, transform[:,:,1]/1e3
        else:
            raise ValueError(
                "Either start_date or traj_data must be provided.")

        if not os.path.exists(self._data_path):
                os.makedirs(self._data_path)

    def __str__(self) -> str:
        parentStr = super().__str__()
        return ("Boundary Data object; " + parentStr +
                f" k: {self._k}, convex: {self._convex}, alpha: {self._alpha}")

    def load(self) -> None:
        with open(self._dict_path, 'rb') as f:
            d = pickle.load(f)
            self._hulls = d['hulls']
            self._is_bound = d['is_bound']

    def save(self) -> None:
        if not self._hulls:
            is_bound, hulls = self.calc_bounds()
        else:
            is_bound, hulls =  self._is_bound, self._hulls

        with open(self._dict_path, 'wb') as f:
            pickle.dump({"hulls": hulls,
                         "is_bound": is_bound}, f)

    def calc_bounds(self) -> Tuple[
        Dict[np.datetime64, np.ndarray],
        Dict[np.datetime64, Trimesh]]:

        self._is_bound, self._hulls = cc.calc_bounds(self._x, self._y,
                                         self._traj_data.trajs['p'] * self._k,
                                         self._traj_data.trajs['time'][0,:],
                                         self._convex, self._alpha)
        return self._is_bound, self._hulls

    def calc_or_load(self, convex: bool, alpha: float) -> Tuple[
            Dict[np.datetime64, np.ndarray],
            Dict[np.datetime64, Trimesh]]:

        self._alpha=alpha
        self._convex = convex
        self._dict_path = (
                        f"{self._data_path}{'convex' if convex else 'concave'}"
                        f"{f'{alpha}' if alpha is not None else ''}"
                    ).strip()
        if os.path.exists(self._dict_path):
            self.load()
            return self._is_bound, self._hulls
        else:
            is_bound, hulls = self.calc_bounds()
            self.save()
            return is_bound, hulls

    @property
    def k(self) -> float:
        return self._k

    @k.setter
    def k(self, value: float) -> None:
        self._k = value

    @property
    def convex(self) -> bool:
        return self._convex

    @convex.setter
    def convex(self, value: bool) -> None:
        self._convex = value
        self._dict_path = (
                        f"{self._data_path}{'convex' if value else 'concave'}"
                        f"{f'{self._alpha}' if self._alpha is not None else ''}"
                    ).strip()

    @property
    def alpha(self) -> Optional[float]:
        return self._alpha

    @alpha.setter
    def alpha(self, value: Optional[float]) -> None:
        self._alpha = value
        self._dict_path = (
                        f"{self._data_path}"
                        f"{'convex' if self._convex else 'concave'}"
                        f"{f'{value}' if value is not None else ''}"
                    ).strip()

    @property
    def traj_data(self) -> Optional[Traj]:
        return self._traj_data

    @traj_data.setter
    def traj_data(self, value: Optional[Traj]) -> None:
        self._traj_data = value

        if value is not None:
            self._n_traj = value.n_traj
            self._n_steps = value.n_steps
            self._dt = value.dt
            # transform horizontal coords
            transform = self._projection.transform_points(
                cartopy.crs.PlateCarree(), self._traj_data.trajs['lon'],
                self._traj_data.trajs['lat'])
            self._x, self._y = transform[:,:,0]/1e3, transform[:,:,1]/1e3

    def plot(self, **kwargs) -> Tuple[mpl.figure.Figure, mpl.axes._axes.Axes]:
        """
        Default Boundary plot. Invokes plot_bound.

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        fig : matplotlib figure
        ax : matplotlib ax

        """
        fig, ax = self.plot_bound(**kwargs)
        return fig, ax

    def plot_bound(self, **kwargs) -> Tuple[mpl.figure.Figure,
                                            mpl.axes._axes.Axes]:

        self._BoundVisualizer = pp.BoundVisualizer(
            self._x, self._y, self._traj_data.trajs['p'] * self._k,
            self.calc_or_load, self._convex, alpha = self._alpha)

        return self._BoundVisualizer.fig, self._BoundVisualizer.ax

