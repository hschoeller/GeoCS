#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:10:26 2024

@author: schoelleh96
"""

from typing import Optional, List, Tuple, Dict
from abc import ABC, abstractmethod
import os
from datetime import datetime
import pickle
import numpy as np
import matplotlib as mpl
from trimesh.base import Trimesh
import cartopy
import scipy.sparse as sps
from . import plot_lib as pp
from . import calc_lib as cc

# %%


class Data(ABC):
    """
    Abstract base class for all kinds of data in this package.

    Attributes
    ----------
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
    def plot(self) -> Tuple[mpl.figure.Figure, mpl.axes._axes.Axes]:
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

    Attributes
    ----------
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
        self._k: Optional[float] = None
        self._extent: Optional[List] = [-180, 180, -90, 90]
        self._projection: Optional[cartopy.crs] = cartopy.crs.Mercator()

    def __str__(self) -> str:
        parentStr = super().__str__()
        return ("Trajectory Data object; " + parentStr +
                f" k: {self._k}")

    def load(self) -> None:
        """
        Load trajectory data (npy) from file specified in data_path.

        Returns
        -------
        None.

        """
        self._trajs = np.load(self._data_path)
        self._get_properties()

    def _get_properties(self) -> None:
        self._n_traj, self._n_steps = self._trajs.shape
        self._dt = (self._trajs['time'][0, 1] -
                    self._trajs['time'][0, 0]).astype(datetime)

    def save(self) -> None:
        """
        Save trajectory data (npy) to file specified in data_path.

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

    def plot(self, **kwargs) -> Tuple[mpl.figure.Figure,
                                      cartopy.mpl.geoaxes.GeoAxes]:
        """
        Plot default Trajectory plot. Invokes plot2D.

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

    def plot_2d(self, **kwargs) -> Tuple[mpl.figure.Figure,
                                         cartopy.mpl.geoaxes.GeoAxes]:
        """
        Plot simple 2D trajectory plot.

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
                                  self._extent, **kwargs)
        return fig, ax

# %%


class Dist(Data):
    """
    A class for handling pairwise distances of trajectories.

    Attributes
    ----------
        _data_path (str): Path where the  data is stored or to be stored.
        _start_date (datetime): The start date of the trajectories.
        _r (float): The cut-off radius for distance calculations.
        _k (float): vertical scaling parameter
        _save_pattern (str): The pattern used for saving distance matrix files,
            with datetime formatting symbols (e.g., "%Y%m%d_%H%M%S.npz").
        _mats (Dict[datetime, sps.csr_matrix]): A dictionary mapping each
            timestep to its corresponding sparse distance matrix triangle.
        _mat_paths (List[str]): The list of file paths for the matrices.
        _traj_data (Optional[traj_data]): An optional traj_data object from
            which pairwise distances can be calculated if not loading.
    """

    def __init__(self, data_path: str, r: float, k: float,
                 start_date: Optional[datetime] = None,
                 traj_data: Optional[Traj] = None,
                 save_pattern: Optional[str] = "%Y%m%d_%H%M.npz"):
        self._save_pattern = save_pattern
        self._mats = {}
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
            raise ValueError(
                "Either start_date or traj_data must be provided.")

        if not os.path.exists(self.data_path):
            os.makedirs(self._data_path)

    def __str__(self) -> str:
        parentStr = super().__str__()
        return ("Distance Data object; " + parentStr +
                f" k: {self._k}, r: {self._r}, ")

    def load(self) -> None:
        """
        Load all available distance matrices. Caution for large data.

        Returns
        -------
        None
            DESCRIPTION.

        """
        for mp in self._mat_paths:
            fullPath = os.path.join(self._data_path, mp)  # Ensure full path
            date_key = datetime.strptime(mp, self._save_pattern)
            self._mats[date_key] = self.load_mat(fullPath)

    def load_mat(self, full_path: str) -> sps.csr_matrix:
        return sps.load_npz(full_path)

    def save(self) -> None:
        """
        Save distance matrix for all dates. Caution for large data.

        Returns
        -------
        None
            DESCRIPTION.

        """
        for mp in self._mat_paths:
            date_key = datetime.strptime(mp, self._save_pattern)
            if date_key in self._mats:
                self.save_mat(self._mats[date_key], mp)
            else:
                dist_mat = self.calc_dist(date_key)
                self.save_mat(dist_mat, mp)

    def save_mat(self, mat: sps.csr_matrix, matPath: str) -> None:
        fullPath = os.path.join(self._data_path, matPath)  # Ensure full path
        sps.save_npz(fullPath, mat)

    def calc_dist(self, date_key: Optional[datetime] = None,
                  timestep: Optional[int] = None) -> sps.csr_matrix:
        if date_key is not None:
            # Find the index where 'time' matches the timestep
            index = np.where((self._traj_data.trajs['time']).astype(datetime)
                             == date_key)[1]
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

    def calc_or_load(self, date_key: Optional[datetime]) -> sps.csr_matrix:
        mat_path = date_key.strftime(self._save_pattern)
        if os.path.exists(os.path.join(self._data_path, mat_path)):
            mat = self.load_mat(os.path.join(self._data_path, mat_path))
        else:
            mat = self.calc_dist(date_key)
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
        Plot default Distances plot. Invokes plot_dist_hist.

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
        Plot histogram of distances.

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
    """
    Represents boundaries point clouds belonging to trajectory data.

    Uses either convex hulls or alpha shapes.

    Parameters
    ----------
    data_path : str
        The file path for storing and loading boundary data.
    k : float
        The scaling parameter used in distance calculations.
    convex : bool
        Specifies whether to use convex hulls (True) or alpha shapes (False).
    alpha : Optional[float], default=None
        The alpha parameter for alpha shape calculation. If None and using
        alpha shapes, an optimal alpha will be estimated.
    start_date : Optional[datetime], default=None
        The start date for the data analysis period.
    traj_data : Optional[Traj], default=None
        An instance of the Traj class containing trajectory data to be analyzed

    Attributes
    ----------
    _hulls : Dict[datetime, Trimesh]
        Stores the hull (either convex or alpha shape) for each timestep.
    _is_bound : Dict[datetime, np.ndarray]
        Indicates whether points are within the boundary for each timestep.
    _projection : Optional[cartopy.crs]
        The cartopy coordinate reference system used for data projection.
    """

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
            self._x, self._y = transform[:, :, 0]/1e3, transform[:, :, 1]/1e3
        else:
            raise ValueError(
                "Either start_date or traj_data must be provided.")

        if not os.path.exists(self._data_path):
            os.makedirs(self._data_path)

    def __str__(self) -> str:
        parent_str = super().__str__()
        return ("Boundary Data object; " + parent_str +
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
            is_bound, hulls = self._is_bound, self._hulls

        with open(self._dict_path, 'wb') as f:
            pickle.dump({"hulls": hulls,
                         "is_bound": is_bound}, f)

    def calc_bounds(self) -> Tuple[Dict[datetime, np.ndarray],
                                   Dict[datetime, Trimesh]]:

        self._is_bound, self._hulls = cc.calc_bounds(
            self._x, self._y, self._traj_data.trajs['p'] * self._k,
            (self._traj_data.trajs['time'][0, :]).astype(datetime),
            self._convex, self._alpha)
        return self._is_bound, self._hulls

    def calc_or_load(self, convex: bool, alpha: float) -> Tuple[
            Dict[datetime, np.ndarray],
            Dict[datetime, Trimesh]]:

        self._alpha = alpha
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
    def is_bound(self) -> Dict[datetime, np.typing.NDArray[bool]]:
        return self._is_bound

    @property
    def hulls(self) -> Dict[datetime, Trimesh]:
        return self._hulls

    @property
    def convex(self) -> bool:
        return self._convex

    @convex.setter
    def convex(self, value: bool) -> None:
        self._convex = value
        self._dict_path = (
            f"{self._data_path}{'convex' if value else 'concave'}"
            f"{f'{self._alpha}' if self._alpha is not None else ''}").strip()

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
            self._x, self._y = transform[:, :, 0]/1e3, transform[:, :, 1]/1e3

    @property
    def x(self) -> np.typing.NDArray[float]:
        return self._x

    @property
    def y(self) -> np.typing.NDArray[float]:
        return self._y

    def plot(self, **kwargs) -> Tuple[mpl.figure.Figure, mpl.axes._axes.Axes]:
        """
        Plot default Boundary plot. Invokes plot_bound.

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
        """
        Plot an interactive widget to demonstrate boundary detection.

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        fig : matplotlib figure
        ax : matplotlib ax

        """
        self._BoundVisualizer = pp.BoundVisualizer(
            self._x, self._y, self._traj_data.trajs['p'] * self._k,
            self.calc_or_load, self._convex, alpha=self._alpha)

        return self._BoundVisualizer.fig, self._BoundVisualizer.ax

# %%


class DiffMap(Data):
    """
    Diffusion maps for trajectory data to identify coherent sets.

    Parameters
    ----------
    data_path : str
        Path where the computed diffusion map results are stored or will be
        stored.
    eps : float
        The epsilon parameter controlling the diffusion process scale.
    N_v : Optional[int], default=20
        Number of eigenvectors (and corresponding eigenvalues) to compute.
    N_cs : Optional[int], default=6
        Number of coherent sets to identify from the diffusion map.
    start_date : Optional[datetime], default=None
        Starting date for analyzing the trajectory data.
    bound_data : Optional[Bound], default=None
        An instance of the Bound class containing boundary data.
    dist_data : Optional[Dist], default=None
        An instance of the Dist class containing distance data between points.

    Attributes
    ----------
    _E_vals : np.ndarray or None
        The eigenvalues computed from the diffusion map.
    _E_vecs : np.ndarray or None
        The eigenvectors computed from the diffusion map.
    """

    def __init__(self, data_path: str, eps: float, N_v: Optional[int] = 20,
                 N_cs: Optional[int] = 6,
                 start_date: Optional[datetime] = None,
                 bound_data: Optional[Bound] = None,
                 dist_data: Optional[Dist] = None):
        self._bound_data = bound_data
        self._dist_data = dist_data
        self._E_vals = None  # Placeholder for the calculation result
        self._E_vecs = None
        self._eps = eps
        self._N_v = N_v
        self._N_cs = N_cs
        if start_date is not None:
            super().__init__(data_path, start_date)
            self._n_traj = None
            self._n_steps = None
            self._dt = None
            self._traj_data = None
        elif (dist_data is not None) and (bound_data is not None):
            super().__init__(data_path, bound_data.start_date)
            self._n_traj = bound_data.n_traj
            self._n_steps = bound_data.n_steps
            self._dt = bound_data.dt
            self._x, self._y = bound_data._x, bound_data._y
        else:
            raise ValueError(
                "Either start_date or data must be provided.")

        if not os.path.exists(self._data_path):
            os.makedirs(self._data_path)

    def __str__(self) -> str:
        parent_str = super().__str__()
        return ("Diffusion Map Object; " + parent_str +
                f" epsilon: {self._eps}, N_v: {self._N_v}")

    @property
    def file_path(self):
        return (f"{self._data_path}{self._eps}.npz").strip()

    def save(self) -> None:
        np.savez(self.file_path, eigenvalues=self._E_vals,
                 eigenvectors=self._E_vecs)

    def load(self) -> None:
        data = np.load(self.file_path)
        self._E_vals, self._E_vecs = data['eigenvalues'], data['eigenvectors']

    def calc_diff_map(self, eps: float) -> Tuple[np.typing.NDArray[float],
                                                 np.typing.NDArray[float]]:
        if not self._bound_data.is_bound:
            is_bound, _ = self._bound_data.calc_or_load(
                self._bound_data.convex, self._bound_data.alpha)
        if self._dist_data.mats:
            # distance matrices are in RAM already
            self._E_vals, self._E_vecs = cc.calc_diff_map(
                eps, self._bound_data.is_bound,
                self._N_v, self._n_traj,
                (self._dist_data.traj_data._trajs['time'][0, :]
                 ).astype(datetime),
                dist_mats=self._dist_data.mats)
        else:
            self._E_vals, self._E_vecs = cc.calc_diff_map(
                eps, self._bound_data.is_bound,
                self._N_v, self._n_traj,
                (self._dist_data.traj_data._trajs['time'][0, :]
                 ).astype(datetime),
                calc_dist=self._dist_data.calc_or_load)
        self._eps = eps
        return self._E_vals, self._E_vecs

    def calc_or_load(self, eps: float) -> Tuple[np.typing.NDArray[float],
                                                np.typing.NDArray[float]]:
        self._eps = eps
        if os.path.exists(self.file_path):
            self.load()
        else:
            self._E_vals, self._E_vecs = self.calc_diff_map(eps)
        return self._E_vals, self._E_vecs

    def cluster_cs(self, N_cs: int) -> np.typing.NDArray[int]:
        self._kmeans = cc.kmeans(self._E_vecs, N_cs)
        self._N_cs = N_cs
        return self._kmeans.labels_

    def plot(self, **kwargs) -> Tuple[mpl.figure.Figure, mpl.axes._axes.Axes]:
        """
        Plot default coherent set plot. Invokes plot_cs.

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        fig : matplotlib figure
        ax : matplotlib ax

        """
        fig, ax = self.plot_cs(**kwargs)
        return fig, ax

    def plot_cs(self, **kwargs) -> Tuple[mpl.figure.Figure,
                                         mpl.axes._axes.Axes]:
        """
        Plot interactive widget to analyze behaviour of coherent set detection.

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        fig : matplotlib figure
        ax : matplotlib ax

        """
        self._CSVisualizer = pp.CSVisualizer(
            self._bound_data.x, self._bound_data.y,
            self._bound_data.traj_data.trajs['p'] * self._bound_data.k,
            self._bound_data.is_bound, self._eps, self._N_cs,
            self.calc_or_load, self.cluster_cs)

        return self._CSVisualizer.fig, self._CSVisualizer.ax
