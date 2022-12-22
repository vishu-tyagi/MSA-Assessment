import logging
import logging.config
from pathlib import Path

import requests

from rlabs.utils import timing

logger = logging.getLogger(__name__)


@timing
def download_data(url, save_to: Path) -> None:
    """
    Download data from URL
    Args:
        data_url (_type_): URL to download from
        to_ (Path): Destination for downloaded data
    """
    logger.info(f"Downloading {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(str(save_to), 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
