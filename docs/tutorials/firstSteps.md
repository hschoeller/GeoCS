# First Steps
The package follows object orientation and is centered around classes handling trajectories (Traj), point-wise distances (Dist), point-cloud boundaries (Bound), and diffusion maps (DiffMap). Each class can be calculated, saved, loaded and plotted.

```
from GeoCS import Traj, Dist, Bound, DiffMap
from datetime import datetime

T = Traj(path_to_your_trajectories, datetime(Y, M, D, H))
T.load()

r = 1e5  # cut-off radius
k = 15  # scaling parameter

D = Dist(path_to_distances, r=r, k=k, traj_data=T)
D.save()

B = Bound(path_to_boundaries, k=k, convex=True, traj_data=T)
B.save()

eps = 1e5  # diffusion bandwidth
DM = DiffMap(path_to_diffusion_maps, eps=eps, bound_data=B, dist_data=D)

DM.save()

DM.plot()
```
