from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from rlabs.config import RLabsConfig
from rlabs.utils.constants import (
    CLUSTER,
    MAJORITY,
    SECOND_MAJORITY
)


class CustomEvaluation():
    def __init__(self, config: RLabsConfig = RLabsConfig):
        pass

    def elbow_method(self, X: np.ndarray, k_values: List):
        inertias = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
            inertias.append(kmeans.inertia_)
        return inertias

    def compute_purity_scores(
        self,
        clusters: np.ndarray,
        y_true: np.ndarray
    ):
        rows = []
        labels_which_got_assigned = []
        for c in set(clusters.tolist()):
            labels = y_true[clusters == c]
            labels_sorted = sorted(
                list(set(labels.tolist())),
                key=labels.tolist().count,
                reverse=True
            )
            most_common = labels_sorted[0]
            second_most_common = \
                labels_sorted[1] if len(labels_sorted) > 1 else -1
            max_purity = sum(labels == most_common) / len(labels)
            second_max_purity = \
                sum(labels == second_most_common) / len(labels) \
                if second_most_common != -1 else 0.0
            most_common_label = most_common
            second_most_common_label = second_most_common
            rows.append([
                (most_common_label, f"{max_purity:.3}"),
                (second_most_common_label, f"{second_max_purity:.3}"),
                c
            ])
            if c != -1:
                labels_which_got_assigned.append(most_common_label)
        rows = sorted(rows, key=lambda x: x[0][0], reverse=False)
        df = pd.DataFrame(rows, columns=[MAJORITY, SECOND_MAJORITY, CLUSTER])
        counts = {}
        for label in labels_which_got_assigned:
            counts[label] = 1 if label not in counts else (counts[label] + 1)
        missing = set(list(y_true)) - set(counts.keys())
        multiple = [(label, counts[label]) for label in counts if counts[label] > 1]
        return df, missing, multiple
