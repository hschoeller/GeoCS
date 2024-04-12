# Distances

Distance objects can by initiated with an existing trajectory object or by loading saved distance data from earlier calculations. We will use the Traj object from earlier.

Parameter r is a cut-off radius to limit computational cost of distance calculations and k is the scaling parameter (vertical distances will be scaled by k). Both can be changed later on, too.

```
from datetime import datetime
from GeoCS import Dist


D = Dist(data_path=start_date.strftime("./dists/%Y%m%d_%H/"), r=1e5, k=15, traj_data=T)

D.r = 1e4
```

Since distance calculations scale quadratically (worst-case) with the number of points, you can either calculate/load the distance and save them as an attribute or only calculate/load them once required. The calc_or_load function calculates the point-wise distances only if there is no data saved for the respective timestep. You can call it with an integer (timestep) or with a datetime (date).

```
D_mat = D.calc_or_load(timestep=0)

D.save_mat(D_mat, D.mat_paths[0])
```

To calculate/load all data at once, simply do

```
D.save()

D.load()

D.mats
```

The defaul plot option is a 2d histogram as a heatmap, showing the frequency of distances across the timesteps.

```
D.plot()
```