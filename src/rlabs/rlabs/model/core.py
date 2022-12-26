import math
import logging
from tabnanny import verbose

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN

from rlabs.config import RLabsConfig
from rlabs.utils import timing
from rlabs.utils.constants import (
    POPULATION,
    CLUSTER
)

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, config: RLabsConfig = RLabsConfig):
        self.config = config

    @timing
    def cluster(
        self,
        D: np.ndarray
    ) -> HDBSCAN:
        """
        Run Hierarchical clustering
        Args:
            D (np.ndarray): Pre-computed pairwise distance matrix
        Returns:
            clusters (HDBSCAN): HDBSCAN fit object
        """
        clusters = HDBSCAN(**self.config.hdbscan_params).fit(D)
        return clusters

    @timing
    def merge(
        self,
        df: pd.DataFrame,
        D: np.ndarray,
        population_threshold: int = 5e4,
        distance_threshold: int = 40,
        max_iter: int = 100,
        verbose: int = 10
    ) -> pd.DataFrame:
        """
        Merge clusters to ensure each cluster has at least one city with
        the population of at least population_threshold
        Args:
            df (pd.DataFrame): Dataframe with latitude, longitude, population and clusters.
            Clusters may be obtained from a clustering mechanism like HDBSCAN
            D (np.ndarray): Pre-computed pairwise distance matrix between lat/long pairs of df
            population_threshold (int): Maximum population of a city in a cluster should be
            at least `population_threshold` for that cluster to be an MSA
            distance_threshold (int): For a cluster that does not have a city with population of
            atleast `population_threshold`, if the closest cluster to it is farther than
            `distance_threshold` (in kilometres), then it will be marked as non-metropolitan
            max_iter (int): Maximum iterations to run for
            verbose (int): Display progress for every `verbose` iterations
        Returns:
            df (pd.DataFrame): Dataframe with updated clusters
        """
        grouped_df = \
            df[~df[CLUSTER].isin([-1])] \
            .groupby(by=[CLUSTER], as_index=False) \
            .agg({POPULATION: "max"}).copy()

        if grouped_df[POPULATION].max() < population_threshold:
            logger.info(f"Found 0 cities with population of {population_threshold}")
            logger.info(f"All cities will be marked as noise")
            logger.info(f"Consider lowering population_threshold")
            df[CLUSTER] = -1
            return df

        logger.info(f"Merging clusters ...")
        iter = max_iter
        while iter:
            grouped_df = \
                df[~df[CLUSTER].isin([-1])] \
                .groupby(by=[CLUSTER], as_index=False) \
                .agg({POPULATION: "max"}).copy()
            if grouped_df[POPULATION].min() >= population_threshold:
                break

            cluster = \
                grouped_df[
                    grouped_df[POPULATION].isin([grouped_df[POPULATION].min()])
                ][CLUSTER].values[0]
            D_cluster = D[df[CLUSTER].isin([cluster]), :]
            m, n = D_cluster.shape

            distances = [math.inf for _ in range(m)]
            clusters = [-1 for _ in range(m)]
            for i in range(m):
                for j in range(n):
                    if (df.iloc[j][CLUSTER] == cluster) or (df.iloc[j][CLUSTER] == -1):
                        pass
                    else:
                        if D_cluster[i, j] < distances[i]:
                            distances[i] = D_cluster[i, j]
                            clusters[i] = df.iloc[j][CLUSTER]

            arg = np.argmin(np.array(distances))
            closest_distance, closest_cluster = distances[arg], clusters[arg]
            if closest_distance >= distance_threshold:
                closest_cluster = -1
            df[CLUSTER] = df[CLUSTER].apply(lambda x: closest_cluster if x == cluster else x)

            iter -= 1
            if not (max_iter - iter) % verbose:
                num_clusters = df[CLUSTER].nunique() - 1
                logger.info(f"Ran for {max_iter - iter} iterations. # clusters {num_clusters}")

        if not iter:
            if df[~df[CLUSTER].isin([-1])] \
                .groupby(by=[CLUSTER], as_index=False) \
                .agg({POPULATION: "max"})[POPULATION].min() < population_threshold:
                logger.info(f"Maximum iterations {max_iter} reached without convergence")
                logger.info(f"Consider increasing max_iter")
            else:
                logger.info(f"Converged in {max_iter} iterations")
        else:
            logger.info(f"Converged in {max_iter - iter} iterations")

        remap = {c: i for i, c in enumerate(sorted(list(set(df[CLUSTER]) - set({-1}))))}
        df[CLUSTER] = df[CLUSTER].replace(remap)
        return df