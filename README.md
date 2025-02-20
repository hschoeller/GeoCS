<!-- SPHINX-START -->

[![DOI](https://zenodo.org/badge/777665015.svg)](https://doi.org/10.5281/zenodo.14899385)

# GeoCS

A package to calculate coherent sets from geospatial trajectory data.

## Installation

```bash
pip install GeoCS
```

## Quick Start

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

## Documentation

Full documentation is available on readthedocs: [https://geocs.readthedocs.io/](https://geocs.readthedocs.io/).

Project repository is at github: [https://github.com/hschoeller/GeoCS](https://github.com/hschoeller/GeoCS)

## Citation

If you use this package in your research, please cite it as:

**Schoeller, Henry (2025). GeoCS (Version 1.0.2). Zenodo.**  
[https://doi.org/10.5281/zenodo.14899385](https://doi.org/10.5281/zenodo.14899385)

## License:

Licensed under the [MIT License](https://github.com/hschoeller/GeoCS/blob/main/LICENSE).

## Credits:

Development has been financed by the DFG funded [CRC 1114](https://www.mi.fu-berlin.de/en/sfb1114/index.html).

Largely based on theory laid out in Banisch & Koltai, 2017. Application and extension in the context of atmospheric flow will be detailed in future publication (Schoeller et. al, 2025).

Banisch, Ralf and P ́eter Koltai (Mar. 2017). “Understanding the Geometry of Transport: Diffusion Maps for Lagrangian Trajectory Data Unravel Coherent Sets”. In: Chaos 27.3, p. 035804. issn: 1054-1500, 1089-7682. doi: 10.1063/1.4971788.