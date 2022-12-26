import sys
import logging

from rlabs.api import (fetch, build)
from rlabs.utils.constants import USA


if __name__ == "__main__":
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    try:
        if sys.argv[1] == "fetch":
            fetch()
        elif sys.argv[1] == "build-usa":
            build(target_fname="us_cities.csv", countries_list=[USA])
        elif sys.argv[1] == "build-all":
            build(target_fname="all_cities.csv")
    except IndexError:
        raise ValueError("Call to API requires an endpoint")