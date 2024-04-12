#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:09:03 2024

@author: schoelleh96
"""

from typing import List, Tuple, Dict, Optional, Callable
from abc import ABC, abstractmethod
from datetime import datetime
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sb
import cartopy
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.widgets import Slider, CheckButtons, RadioButtons, TextBox
from trimesh import Trimesh
import numpy as np


class PointCloudVisualizer(ABC):
    """
    Abstract base class interactive point cloud visualizers.

    Attributes
    ----------
    x : np.ndarray
        The x-coordinates of the points in the point cloud.
    y : np.ndarray
        The y-coordinates of the points in the point cloud.
    z : np.ndarray
        The z-coordinates of the points in the point cloud.
    t_i : int
        The initial time index for the visualization.
    fig : plt.Figure
        The figure object for the plot.
    ax : plt.Axes
        The axes object for the plot.
    t_slider : Slider
        An interactive slider widget to control the time dimension.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 initial_time_index: int = 0):
        self.x, self.y, self.z = x, y, z
        self.t_i = initial_time_index
        self.fig, self.ax = self._setup_fig_axes()
        self._init_widgets()
        self.t_slider.on_changed(self._update_plot)

    def _setup_fig_axes(self) -> Tuple[mpl.figure.Figure, mpl.axes._axes.Axes]:
        """Set up the figure and axes for plotting."""
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'},
                               figsize=(14, 8))
        return fig, ax

    def _init_widgets(self):
        """Initialize interactive widgets for the plot."""
        t_slider_ax = self.fig.add_axes([0.05, 0.95, 0.15, 0.025])
        self.t_slider = Slider(t_slider_ax, 'T', 0, self.x.shape[1]-1,
                               valinit=self.t_i, valstep=1)

    @abstractmethod
    def _update_plot(self, val=None):
        """Update the plot."""
        pass

    @abstractmethod
    def _recalculate(self, event=None):
        """Recalculate data and update the plot."""
        pass

# %% Traj functions


def plot_traj_2d(trajs: np.ndarray, projection: cartopy.crs.Projection,
                 extent: List[float], **kwargs) -> Tuple[mpl.figure.Figure,
                                                         GeoAxes]:
    """
    Plot a 2D trajectory map with specified projection and extent.

    Parameters
    ----------
    trajs : numpy.ndarray
        A structured array containing 'lon', 'lat', and 'p' fields.
    projection : cartopy.crs.Projection
        The cartopy coordinate reference system to use for the plot.
    extent : List[float]
        A list of floats specifying the extent of the plot as
        [longitude_min, longitude_max, latitude_min, latitude_max].
    **kwargs : dict, optional
        Additional keyword arguments:
        - cmap (matplotlib.colors.Colormap): The colormap for the line plot.
          Default is a custom cmap.
        - norm (matplotlib.colors.Normalize): The normalization for the line plot.
        - figsize (tuple): Figure size as (width, height). Default is (3.5, 2).
        - every_n (int): Frequency of trajectories to plot. Default is 50.
        - linewidth (float): Width of the trajectory lines. Default is 0.4.
        - points (list): Indices of points to select for the scatter plot.
          Default is [0, -1] for the first and last points.
        - s (float): Size of the scatter plot markers. Default is 0.4.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : cartopy.mpl.geoaxes.GeoAxes
        The cartopy GeoAxes object.
    """
    def make_segments(x, y):
        """
        Create list of line segments from x and y coordinates.

        Parameters
        ----------
        x : float
            coordinate.
        y : float
            coordinate.

        Returns
        -------
        segments : line segment
            for LineCollection.

        """
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        return segments

    def color_line(x, y, z=None, cmap=None, norm=None, linewidth=3,
                   alpha=1.0, ax=None):
        """
        Plot a colored line with coordinates x and y.

        Parameters
        ----------
        x : float
            coordinate.
        y : float
            coordinate.
        z : array, optional
            specify colorspacing. The default is None.
        cmap : colorbar, optional
            colorbar. The default is None.
        norm : normalize.colors, optional
            normalize.colors. The default is None.
        linewidth : float, optional
            linewidth. The default is 3.
        alpha : float, optional
            alpha. The default is 1.0.

        Returns
        -------
        lc : lineCollection
            Collection of segments.

        """
        if cmap is None:
            cmap = plt.get_cmap('copper')

        if norm is None:
            norm = plt.Normalize(0.0, 1.0)

        # Default colors equally spaced on [0,1]:
        if z is None:
            z = np.linspace(0.0, 1.0, len(x))

        # Special case if a single number:
        if not hasattr(z, "__iter__"):  # to check for numerical input (hack)
            z = np.array([z])

        z = np.asarray(z)

        segments = make_segments(x, y)
        lc = mpl.collections.LineCollection(
            segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth,
            alpha=alpha, transform=cartopy.crs.PlateCarree())

        if ax is None:
            ax = plt.gca()

        ax.add_collection(lc)

        return lc

    # colormap for trajectories

    colors = [  # more detailed 18 colors
        [130, 0, 0],  # rot
        [160, 0, 0],
        [190, 0, 0],
        [220, 30, 0],
        [250, 60, 0],
        [250, 90, 0],
        [250, 120, 0],   # orange
        [250, 170, 30],   # yellow
        [250, 200, 90],
        [250, 220, 150],  # MITTE
        [0, 220, 250],   # cyan
        [0, 90, 250],  # blue
        [0, 60, 250],
        [0, 60, 220],
        [0, 30, 190],
        [0, 30, 160],
        [0, 30, 130],
    ]
    levels = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
              800, 850, 900, 950]
    # 17 levels
    # convert RGB values to range between 0 - 1
    colors = np.array(colors)/255
    # creat colormap
    cmap, norm = mpl.colors.from_levels_and_colors(levels, colors,
                                                   extend='both')

    # Extracting **kwargs
    cmap = kwargs.get('cmap', cmap)
    norm = kwargs.get('norm', norm)
    figsize = kwargs.get('figsize', (3.5, 2))
    every_n = kwargs.get('every_n', 50)
    linewidth = kwargs.get('linewidth', 0.4)
    points = kwargs.get('points', [0, -1])
    s = kwargs.get('s', 0.4)

    fig, ax = plt.subplots(1, 1, figsize=figsize,
                           subplot_kw={'projection': projection})

    ax.coastlines()
    ax.gridlines(linestyle='--', alpha=0.5, zorder=-1)
    if extent != [-180, 180, -90, 90]:
        ax.set_extent(extent, crs=cartopy.crs.PlateCarree())

    # plot trajectories
    Lon = trajs['lon']
    Lat = trajs['lat']
    P = trajs['p']
    n_tra = Lon.shape[0]
    # loop through trajectories
    for i in range(0, n_tra, every_n):  # plot only every nth trajectory

        # cosmetic: lines that cross the 180° longitude create ugly artefacts
        segment = np.vstack((Lon[i], Lat[i]))
        lon0 = 180  # center of map
        bleft = lon0-181.
        bright = lon0+181.
        segment[0, segment[0] > bright] -= 360.
        segment[0, segment[0] < bleft] += 360.
        threshold = 180.
        isplit = np.nonzero(np.abs(np.diff(segment[0])) > threshold)[0]
        subsegs = np.split(segment, isplit+1, axis=+1)

        # plot the tracks
        for seg in subsegs:
            x, y = seg[0], seg[1]
            cl = color_line(x, y, P[i], norm=norm,
                            linewidth=linewidth, cmap=cmap)

    ax.scatter(Lon[::every_n, points], Lat[::every_n, points], color='black',
               s=s, zorder=5, transform=cartopy.crs.PlateCarree())

    # add colorbar
    cbar = fig.colorbar(cl, ax=ax, orientation='horizontal',
                        fraction=0.1, pad=0.05)

    cbar.set_label('$p$ [hPa]')  # ,size=14)
    # cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()

    return fig, ax

# %% Dist functions


def plot_dist_hist(hist_counts: dict[str, list[int]],
                   bin_edges: np.ndarray,
                   **kwargs) -> Tuple[mpl.figure.Figure, mpl.axes._axes.Axes]:
    """
    Plot a heatmap of histogram counts over timesteps.

    Args:
        hist_counts (dict): A dictionary with timesteps as keys and
        histogram counts as values.
        bin_edges (np.ndarray): The edges of the bins used for the histograms.
        **kwargs: Additional keyword arguments to customize the plot:
            - cmap (str): Colormap for the heatmap. Default is "viridis".
            - figsize (Tuple[int, int]): Figure size. Default is (10, 6).

    Returns
    -------
        Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
            The figure and axes objects of the plot.
    """
    # Extracting **kwargs
    cmap = kwargs.get('cmap', "viridis")
    figsize = kwargs.get('figsize', (10, 6))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Sort the keys to ensure the order on the y-axis
    sorted_keys = sorted(hist_counts.keys())

    # Convert the dictionary values to a 2D array
    hist_counts_array = np.array([hist_counts[key] for key in sorted_keys])
    # Plotting the heatmap on the specified axes
    sb.heatmap(hist_counts_array, cmap=cmap, ax=ax)

    # Setting x-axis and y-axis tick positions and labels
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xticks(np.arange(len(bin_edges))[::10])
    ax.set_xticklabels([f"{int(edge)}" for edge in bin_edges[::10]])

    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_yticks(np.arange(len(sorted_keys))[::10])
    ax.set_yticklabels(sorted_keys[::10])

    ax.set_xlabel('Distance')
    ax.set_ylabel('Timestep')

    plt.tight_layout()
    return fig, ax

# %% Bound Functions


class BoundVisualizer(PointCloudVisualizer):
    """
    A visualizer for displaying point clouds with their boundaries.

    Attributes
    ----------
    convex : bool
        Indicates whether to use a convex hull or an alpha shape.
    alpha : Optional[float]
        The alpha parameter for the alpha shape. If None, an optimal alpha will
        be calculated.
    get_bound : Callable[[float, bool], Tuple[Dict[datetime, np.ndarray],
                  Dict[datetime, Trimesh]]]
        A function that, given the alpha parameter and a boolean indicating
        whether to use a convex hull, returns the boundary flags and hulls for
        each timestep.
    is_bound : Dict[datetime, np.ndarray]
        A dictionary mapping timesteps to arrays indicating whether each point
        is within the boundary.
    hulls : Dict[datetime, Trimesh]
        A dictionary mapping timesteps to Trimesh objects representing the
        hulls.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 get_bound: Callable[[bool, float], Tuple[
                                             Dict[datetime, np.ndarray],
                                             Dict[datetime, Trimesh]]],
                 convex: bool, initial_time_index: int = 0,
                 alpha: Optional[float] = None):
        self.convex = convex
        self.alpha = alpha

        super().__init__(x, y, z, initial_time_index)
        self.get_bound = get_bound
        self.is_bound, self.hulls = self.get_bound(convex, alpha)
        self._update_plot()
        self.alph_text.on_submit(self._recalculate)
        self.meth_check.on_clicked(self._recalculate)
        self.hull_check.on_clicked(self._update_plot)

    def _init_widgets(self):
        super()._init_widgets()

        self.meth_ax = self.fig.add_axes([0.25, 0.9, 0.15, 0.05])

        if self.convex:
            init_met = 0
        elif self.alpha is not None:
            init_met = 1
        else:
            init_met = 2
        self.meth_check = RadioButtons(self.meth_ax, ["Convex", "$\\alpha$",
                                                      "opt. $\\alpha$"],
                                       init_met)
        self.meth_ax.set_title("Hull Method")

        self.alph_ax = self.fig.add_axes([0.45, 0.9, 0.15, 0.05])
        self.alph_text = TextBox(self.alph_ax, "$\\alpha$")
        self.alph_text.set_val(self.alpha)
        self.alph_ax.set_title("$\\alpha$")

        self.hull_ax = self.fig.add_axes([0.65, 0.9, 0.15, 0.05])
        self.hull_check = CheckButtons(self.hull_ax, ["Plot hull?"], [False])

    def _update_plot(self, event=None):
        self.ax.clear()
        t = int(self.t_slider.val)
        date_key = sorted(self.is_bound.keys())[t]
        self.ax.scatter(self.x[:, t], self.y[:, t], self.z[:, t],
                        c=self.is_bound[date_key])
        self.ax.invert_zaxis()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        if self.hull_check.get_status()[0]:
            hull = self.hulls[date_key]
            self.ax.plot_trisurf(hull.vertices[:, 0], hull.vertices[:, 1],
                                 triangles=hull.faces,
                                 Z=hull.vertices[:, 2], alpha=0.5)
        plt.draw()

    def _recalculate(self, event=None):

        if self.meth_check.value_selected == "Convex":
            self.convex = True
            self.is_bound, self.hulls = self.get_bound(self.convex, None)
        elif self.meth_check.value_selected == "$\\alpha$":
            self.convex = False
            self.alpha = float(self.alph_text.text)
            self.is_bound, self.hulls = self.get_bound(self.convex, self.alpha)
        else:
            self.convex = False
            self.is_bound, self.hulls = self.get_bound(self.convex, None)
        self._update_plot()

# %% CS functions


class CSVisualizer(PointCloudVisualizer):
    """
    A visualizer for displaying coherent sets.

    Attributes
    ----------
    N_cs : int
        The number of coherent sets (clusters) to identify.
    eps : float
        The epsilon parameter used in the diffusion map calculation.
    get_E : Callable[[float], Tuple[np.typing.NDArray[float],
                                    np.typing.NDArray[float]]]
        A function to compute or retrieve the eigenvalues and eigenvectors.
    get_clust : Callable[[int], np.typing.NDArray[int]]
        A function to perform clustering on the eigenvectors.
    cluster_labels : np.typing.NDArray[int]
        The labels of each point indicating its cluster assignment.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 is_bound: Dict[datetime, np.ndarray], eps: float,
                 N_cs: int,
                 get_E: Callable[[float], Tuple[np.typing.NDArray[float],
                                                np.typing.NDArray[float]]],
                 get_clust: Callable[[int], np.typing.NDArray[int]],
                 initial_time_index: int = 0):
        self.N_cs = N_cs
        self.eps = eps
        super().__init__(x, y, z, initial_time_index)
        self.get_E = get_E
        self.get_clust = get_clust
        self.E_vals, self.E_vecs = self.get_E(eps)
        self.cluster_labels = self.get_clust(N_cs)
        self._update_plot()
        self.eps_text.on_submit(self._recalculate)
        self.N_cs_text.on_submit(self._recalculate)
        self.t0_check.on_clicked(self._update_plot)

    def _init_widgets(self):
        super()._init_widgets()

        self.N_cs_ax = self.fig.add_axes([0.25, 0.9, 0.15, 0.05])
        self.N_cs_text = TextBox(self.N_cs_ax, "$N_{cs}$")
        self.N_cs_text.set_val(self.N_cs)
        self.N_cs_ax.set_title("$N_{cs}$")

        self.eps_ax = self.fig.add_axes([0.45, 0.9, 0.15, 0.05])
        self.eps_text = TextBox(self.eps_ax, "$\\epsilon$")
        self.eps_text.set_val(self.eps)
        self.eps_ax.set_title("$\\epsilon$")

        self.t0_ax = self.fig.add_axes([0.65, 0.9, 0.15, 0.05])
        self.t0_check = CheckButtons(self.t0_ax, ["Plot $t_0$?"], [False])

    def _update_plot(self, event=None):
        self.ax.clear()
        t = int(self.t_slider.val)
        self.ax.scatter(self.x[:, t], self.y[:, t], self.z[:, t],
                        c=self.cluster_labels)
        self.ax.invert_zaxis()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        if self.t0_check.get_status()[0]:
            self.ax.scatter(self.x[:, 0], self.y[:, 0], self.z[:, 0],
                            c=self.cluster_labels)
        plt.draw()

    def _recalculate(self, event=None):
        self.N_cs = int(self.N_cs_text.text)
        self.eps = float(self.eps_text.text)
        self.E_vals, self.E_vecs = self.get_E(self.eps)
        self.cluster_labels = self.get_clust(self.N_cs)

        self._update_plot()
