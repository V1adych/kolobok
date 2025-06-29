import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("initializer")

logger.info("Initializing service...")

from utils import *

logger.info("Service initialized successfully!")
