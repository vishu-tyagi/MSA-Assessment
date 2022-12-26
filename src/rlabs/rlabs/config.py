class RLabsConfig():
    DATA_URL = "https://reveliolabs.s3.us-east-2.amazonaws.com/RevelioLabs_LocationSample_Nov22.csv"
    CURRENT_PATH = None

    # HDBSCAN parameters
    hdbscan_params = {
        "cluster_selection_epsilon": 3.0,
        "min_cluster_size": 2,
        "min_samples": 1,
        "metric": "precomputed",
        "algorithm": "best",
        "cluster_selection_method": "eom"
    }

    # Merge parameters
    merge_params = {
        "population_threshold": 5e4,
        "distance_threshold": 10.0,
        "max_iter": 6000,
        "verbose": 100
    }