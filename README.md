# MSA-Assessment

## Algorithm

It consists of two parts. The first part produces small dense clusters based on latitude and longitude pairs and identifies noise (left unclustered). The idea is to associate the noise with non-metropolitan areas. The second part merges these clusters based on a criterion which accounts for the population-count parameter.

**Part 1 - Hierarchical Clustering**

Clustering is done using HDBSCAN.

We set `min_samples` equal to `1`, which is the least possible value allowed for this parameter.
We choose a small value so that fewer points are flagged as outliers (reduces noise).

We set `min_cluster_size` equal to `2`, which is the least possible value allowed for this parameter.
We choose a small value because we want small dense clusters that can later be merged in Part 2.

We set `cluster_selection_epsilon` equal to `3.0`, which means that HDBSCAN will not merge clusters which are `3 miles` apart.

**Part 2 - Merging**

This uses population to merge the clusters from the first part.

Each cluster should have at least one city with population above a threshold `population_threshold`. If it doesn't, it should be merged to its nearest cluster. If the nearest cluster is farther than a given distance threshold `distance_threshold`, we mark the current cluster as non-metropolitan.

Note that merging this way does not ensure that the new cluster has a city with a population above `population_threshold`.

We set `population_threshold` equal to `50,000`, which we believe is a reasonable threshold.

We set `distance_threshold` equal to `10`, which means that if the closest cluster is farther than `10 miles`, the current cluster will be marked as non-metropolitan.

**Notes**

- We precompute the pairwise distances matrix, which uses Haversine distance. All distance-related parameters will be in miles.

- The merging criterion in part 2. is flexible and can be changed to accommodate for a better criterion or more parameters which indicate economic and social integration between clusters.

- Part 2 algorithm is bound to converge because, on every iteration, we reduce the number of clusters by 1.

- It took 1 hr. 9 minutes to run the algorithm on US cities.

- It took 11 hrs. 23 minutes to run on all cities.

## Setup Instructions

#### Move into top-level directory
```
cd MSA-Assessment
```

#### Install environment
```
conda env create -f environment.yml
```

#### Activate environment
```
conda activate rlabs
```

#### Install package
```
pip install -e src/rlabs
```

Including the optional -e flag will install the package in "editable" mode, meaning that instead of copying the files into your virtual environment, a symlink will be created to the files where they are.

#### Fetch data
```
python -m rlabs fetch
```

#### Run jupyter server
```
jupyter notebook notebooks/
```

You can now use the jupyter kernel to run notebooks.

#### Build clusters for US cities
```
pythom -m rlabs build-usa
```

#### Build clusters for all cities
```
pythom -m rlabs build-all
```

## Notebooks

The notebooks may be viewed in the following order:

1. *[analysis.ipynb](notebooks/analysis.ipynb)* - Includes data exploration and how the problem was approached

2. *[us_cities.ipynb](notebooks/us_cities.ipynb)* - Includes results on US cities

## Results

1. *[us_cities.csv](reports/us_cities.csv)* - CSV with clusters for US cities

2. *[all_cities.csv](reports/all_cities.csv)* - CSV with clusters for all cities