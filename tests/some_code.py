#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:33:01 2024

@author: schoelleh96
"""
# This will move the console to the right working directory.
from os.path import dirname, abspath
from os import chdir
from GeoCS import Traj, Dist, Bound, DiffMap
from datetime import datetime
import cartopy
chdir(dirname(abspath(__file__)))

# %% Traj Test

start_date = datetime(2016, 5, 2, 0)
fPath = start_date.strftime("/net/scratch/schoelleh96/WP2/WP2.1/LAGRANTO/" +
                            "wp21/era5/traj/%Y/traj_%Y%m%d_%H.npy")

T = Traj(fPath, start_date)

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
f, ax = T.plot_2d(figsize=(7, 5))

# %% Dist Test

T.trajs = T.trajs[::10, :]
print(T)

D = Dist(data_path=start_date.strftime("./dists/%Y%m%d_%H/"), r=1e5, k=15,
         traj_data=T)

D.r = 1e4

D_mat = D.calc_dist(timestep=0)

D.save_mat(D_mat, D.mat_paths[0])

D.save()

D.load()

D.mats

D.plot()


# %% Bound Test

B = Bound(data_path=start_date.strftime("./bounds/%Y%m%d_%H/"), k=15,
          convex=True, traj_data=T)

B.save()

B.convex = False
B.alpha = 0.001

print(B)

B._dict_path

B.save()

B.plot()

# %% DiffMap Test

DM = DiffMap(data_path=start_date.strftime("./DiffMaps/%Y%m%d_%H"), eps=5e4,
             bound_data=B, dist_data=D)

vals, vecs = DM.calc_diff_map(5e4)

ll = DM.cluster_cs(5)
