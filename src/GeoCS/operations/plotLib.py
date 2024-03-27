#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:09:03 2024

@author: schoelleh96
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import cartopy
from typing import List, Tuple

def plotTraj2D(trajs: np.ndarray, projection: cartopy.crs.Projection,
               extent: List[float], **kwargs) -> Tuple[mpl.figure.Figure,
                                                  cartopy.mpl.geoaxes.GeoAxes]:
    """
    Plots a 2D trajectory map with specified projection and extent.

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
        - norm (matplotlib.colors.Normalize): The normalization for the line
          plot.
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
        '''
        Create list of line segments from x and y coordinates, in the correct
        format for LineCollection:
        an array of the form numlines x (points per line) x 2 (x and y) array

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

        '''

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        return segments

    def colorline(x, y, z=None, cmap=None, norm=None, linewidth=3,
                  alpha=1.0, ax=None):
        '''
        Plot a colored line with coordinates x and y
        Optionally specify colors in the array z
        Optionally specify a colormap, a norm function and a line width
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

        '''

        if cmap is None:
            cmap=plt.get_cmap('copper')

        if norm is None:
            norm=plt.Normalize(0.0, 1.0)

        # Default colors equally spaced on [0,1]:
        if z is None:
            z = np.linspace(0.0, 1.0, len(x))

        # Special case if a single number:
        if not hasattr(z, "__iter__"):  # to check for numerical input (hack)
            z = np.array([z])

        z = np.asarray(z)

        segments = make_segments(x, y)
        lc = mpl.collections.LineCollection(segments, array=z, cmap=cmap,
                norm=norm, linewidth=linewidth, alpha=alpha,
                transform=cartopy.crs.PlateCarree())

        if ax is None:
            ax = plt.gca()

        ax.add_collection(lc)

        return lc

    # colormap for trajectories

    colors = [#more detailed 18 colors
        [130,0,0], #rot
        [160,0,0],
        [190,0,0],
        [220,30,0],
        [250,60,0],
        [250,90,0],
        [250,120,0],   #orange
        [250,170,30],   #yellow
        [250,200,90],
        [250,220,150], #MITTE
        [0,220,250],   #cyan
        [0,90,250],  #blue
        [0,60,250],
        [0,60,220],
        [0,30,190],
        [0,30,160],
        [0,30,130],
    ]
    levels =  [200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950]
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
    ax.gridlines(linestyle='--', alpha=0.5)
    ax.set_extent(extent, crs=cartopy.crs.PlateCarree())

    # plot trajectories
    Lon = trajs['lon']
    Lat = trajs['lat']
    P = trajs['p']
    n_tra = Lon.shape[0]
    # loop through trajectories
    for i in range(0,n_tra,every_n): # plot only every nth trajectory

        # cosmetic: lines that cross the 180° longitude create ugly artefacts

        segment = np.vstack((Lon[i], Lat[i]))
        lon0 = 180 #center of map
        bleft = lon0-181.
        bright = lon0+181.
        segment[0,segment[0]> bright] -= 360.
        segment[0,segment[0]< bleft]  += 360.
        threshold = 180.
        isplit = np.nonzero(np.abs(np.diff(segment[0])) > threshold)[0]
        subsegs = np.split(segment,isplit+1,axis=+1)

        #plot the tracks
        for seg in subsegs:
            x,y = seg[0],seg[1]

            cl = colorline(x, y, P[i], norm=norm,
                           linewidth=linewidth, cmap=cmap)

    ax.scatter(Lon[::every_n,points], Lat[::every_n,points],
               color='black',
               s=s, zorder=5, transform=cartopy.crs.PlateCarree())

    # add colorbar
    cbar = fig.colorbar(cl, ax=ax, orientation='horizontal',
                        fraction=0.1, pad=0.05)

    cbar.set_label('$p$ [hPa]')#,size=14)
    # cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()

    return fig, ax