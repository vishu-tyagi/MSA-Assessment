import logging

from rlabs.config import RLabsConfig
from rlabs.data_access import DataClass
from rlabs.utils import timing

logger = logging.getLogger(__name__)


@timing
def fetch(config: RLabsConfig = RLabsConfig) -> None:
    logger.info("Fetching data...")
    data = DataClass(config)
    data.make_dirs()
    data.fetch()
    return