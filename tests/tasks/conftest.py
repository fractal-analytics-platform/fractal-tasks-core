import logging
import sys
from pathlib import Path

new_path = str(Path(__file__).parents[1])
logging.info(f"Adding {new_path} to sys.path, and importing * from conftest")
sys.path = [new_path] + sys.path

from conftest import *  # noqa
