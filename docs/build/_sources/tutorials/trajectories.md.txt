# Trajectories

We will use the example data available on the [GitHub repository](https://github.com/hschoeller/GeoCS/tests). You can use any data you want.
The package works with saved numpy arrays that have fields for
- longitude and latitude: "lon" and "lat" in degrees
- pressure: "p" in hPa
- horizontal velocity: "U" and "V" in m/s
- vertical velocity: "OMEGA" in P/s
- time: "time" (in datetime or numpy.datetime64)

```
from datetime import datetime
from GeoCS import Traj

start_date = datetime(2016, 5, 2, 0)
fPath = start_date.strftime("traj_%Y%m%d_%H.npy")

# object created
T = Traj(fPath, start_date)

# load data
print(T)
T.load()
print(T)
```

The scaling parameter is k is used to calculate three dimensional distances according to rough average velocities. Calculate it empirically with

```
T.k
```

Now try plotting:

```
T.plot()
```
Several options exist for this plot. E.g. you can set the extent of the map proportion shown and the map projection used. The plot() method defaults to plot_2d().

``` 
T.extent = [-210, -30, 30, 90]
T.projection = cartopy.crs.Stereographic(
     central_latitude=90.0, true_scale_latitude=50.0,
     central_longitude=-120)

f, ax = T.plot()
f, ax = T.plot_2d(figsize=(7, 5))
```