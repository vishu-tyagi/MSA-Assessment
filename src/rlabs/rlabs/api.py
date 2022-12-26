import os
from pathlib import Path
import logging

import pandas as pd

from rlabs.config import RLabsConfig
from rlabs.data_access import DataClass
from rlabs.model import Model
from rlabs.utils import timing
from rlabs.utils.constants import (DATA_CSV)

logger = logging.getLogger(__name__)


@timing
def fetch(config: RLabsConfig = RLabsConfig) -> None:
    logger.info("Fetching data ...")
    data = DataClass(config)
    data.make_dirs()
    data.fetch()
    return


@timing
def build(
    target_fname: str,
    config: RLabsConfig = RLabsConfig,
    countries_list: list[str] = None
):
    logger.info("Building clusters ...")
    data = DataClass(config)
    model = Model(config)
    df = pd.read_csv(os.path.join(data.data_path, DATA_CSV), index_col=0)
    df = model.build(df=df, countries_list=countries_list)
    save_to = os.path.join(data.reports_path, target_fname)
    df.to_csv(save_to, index=False)
    return