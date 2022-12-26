import math
import logging
from tabnanny import verbose

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.metrics import pairwise_distances

from rlabs.config import RLabsConfig
from rlabs.utils import timing
from rlabs.utils.constants import (
    POPULATION,
    CLUSTER,
    COUNTRY,
    LATITUDE,
    LONGITUDE,
    R
)

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, config: RLabsConfig = RLabsConfig):
        self.config = config

    @timing
    def cluster(
        self,
        D: np.ndarray
    ):
        """
        Run Hierarchical clustering
        Args:
            D (np.ndarray): Pre-computed pairwise distance matrix
        Returns:
            clusters (HDBSCAN): HDBSCAN fit object
        """
        clusters = HDBSCAN(**self.config.hdbscan_params).fit(D)
        logger.info(f"Finished running hierarchical clustering")
        logger.info(f"Found {np.unique(clusters.labels_).shape[0]-1} clusters")
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
    ):
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
        logger.info(f"population_threshold: {population_threshold}")
        logger.info(f"distance_threshold: {distance_threshold}")
        logger.info(f"max_iter: {max_iter}")
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
                logger.info(f"({max_iter - iter}/{max_iter}) {num_clusters} clusters left")

        if not iter:
            if df[~df[CLUSTER].isin([-1])] \
                .groupby(by=[CLUSTER], as_index=False) \
                .agg({POPULATION: "max"})[POPULATION].min() < population_threshold:
                logger.info(f"Maximum iterations {max_iter} reached without convergence")
                logger.info(f"Consider increasing max_iter")
            else:
                logger.info(f"Converged after {max_iter} iterations")
        else:
            logger.info(f"Converged after {max_iter - iter} iterations")

        remap = {c: i for i, c in enumerate(sorted(list(set(df[CLUSTER]) - set({-1}))))}
        df[CLUSTER] = df[CLUSTER].replace(remap)
        logger.info(f"Number of clusters reduced to {df[CLUSTER].nunique()-1}")
        return df

    @timing
    def _build(
        self,
        df: pd.DataFrame
    ):
        if df.shape[0] == 1:
            logger.info("Found only one city")
            df[CLUSTER] = 0
            return df
        X = df[[LONGITUDE, LATITUDE]].to_numpy()
        X_radians = np.radians(X)
        D_pairwise = pairwise_distances(X_radians, X_radians, metric="haversine")
        D_pairwise *= R  # Multiply by Earth radius to get miles
        clusters = self.cluster(D_pairwise)
        df[CLUSTER] = clusters.labels_
        df = self.merge(df=df.copy(), D=D_pairwise, **self.config.merge_params)
        return df

    @timing
    def build(
        self,
        df: pd.DataFrame,
        countries_list: list[str] = None
    ):
        if countries_list is not None:
            df = df[df[COUNTRY].isin(countries_list)].copy()
            df.reset_index(drop=True, inplace=True)
        countries_list = df[COUNTRY].unique().tolist()
        result = None
        for country_name in countries_list:
            logger.info(f"Processing {country_name} ...")
            country = df[df[COUNTRY].isin([country_name])].copy()
            country.reset_index(drop=True, inplace=True)
            country = self._build(df=country.copy())
            country_name_ = country_name.replace(" ", "_")
            country[CLUSTER] = country[CLUSTER].apply(lambda x: f"{country_name_}__{x}")
            if result is not None:
                result = pd.concat([result, country], ignore_index=True).copy()
            else:
                result = country.copy()
        return result
