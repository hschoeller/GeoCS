#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:33:01 2024

@author: schoelleh96
"""

from GeoCS.dataLib import TrajData, DistData
from datetime import datetime
import cartopy

# %% Traj Test

startDate = datetime(2016, 5, 2, 0)
fPath = startDate.strftime("/net/scratch/schoelleh96/WP2/WP2.1/LAGRANTO/wp21/" +
                      "era5/traj/%Y/traj_%Y%m%d_%H.npy")

T = TrajData(fPath, startDate)

print(T)
T.load()
print(T)

T.save()
T.load()
print(T)

T.plot()

T.extent = [-210, -30, 30, 90]
T.projection = cartopy.crs.Stereographic(
     central_latitude=90.0, true_scale_latitude=50.0,
     central_longitude=-120)

f, ax = T.plot()
f, ax = T.plot_2d(figsize=(7,5))

# %% Dist Test

T.trajs = T.trajs[::10,:]
print(T)

D = DistData(dataPath="./dists/", r=1e5, k=15, trajData=T)

D.r=1e3

Dmat = D.calc_dist(timestep=0)

D.save_mat(Dmat, D.matPaths[0])

D.save()

D.load()

D.mats

D.plot()