# Diffusion Maps

The point-wise distances and the boundary indications are used for calculating the diffusion maps. Parameter $\epsilon > 0$ controls the strength of the diffusion.

```
DM = DiffMap(data_path=start_date.strftime("./DiffMaps/%Y%m%d_%H"), eps=5e4,
             bound_data=B, dist_data=D)

vals, vecs = DM.calc_diff_map(5e4)
```

To find coherent sets, we perform a k-Means clustering algorithm on the data points in the space spanned by the diffusion maps.

```
labels = DM.cluster_cs(5)
```

labels is an array that assigns every data point (trajectory) a cluster.