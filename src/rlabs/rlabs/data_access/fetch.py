import os
from pathlib import Path
import logging

import pandas as pd

from rlabs.config import RLabsConfig
from rlabs.data_access.helpers import (download_data)
from rlabs.utils import timing
from rlabs.utils.constants import (
    DATA_DIR,
    MODEL_DIR,
    REPORTS_DIR,
    DATA_CSV
)

logger = logging.getLogger(__name__)


class DataClass():
    def __init__(self, config: RLabsConfig = RLabsConfig):
        self.config = config
        self.data_url = config.DATA_URL

        self.current_path = Path(os.getcwd()) if not config.CURRENT_PATH else config.CURRENT_PATH
        self.data_path = Path(os.path.join(self.current_path, DATA_DIR))
        self.model_path = Path(os.path.join(self.current_path, MODEL_DIR))
        self.reports_path = Path(os.path.join(self.current_path, REPORTS_DIR))

    def make_dirs(self):
        dirs = [
            self.data_path,
            self.model_path,
            self.reports_path
        ]
        for dir in dirs:
            dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created data directory {self.data_path}")
        logger.info(f"Created model directory {self.model_path}")
        logger.info(f"Created reports directory {self.reports_path}")

    @timing
    def fetch(self):
        save_to = Path(os.path.join(self.data_path, DATA_CSV))
        download_data(url=self.data_url, save_to=save_to)
        logger.info(f"Downloaded {DATA_CSV} to {self.data_path}")

