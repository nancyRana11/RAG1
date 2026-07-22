"""
logger.py
---------
Central logging setup. Import `logger` anywhere in the project:

    from src.utils.logger import logger
    logger.info("Something happened")

Logs go to console AND to a rotating file under data/processed/logs/
so you can debug issues after the fact.
"""

import sys
from loguru import logger

from config import PROCESSED_DIR, LOG_LEVEL

LOG_DIR = PROCESSED_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Remove the default handler and configure our own so we control format/level.
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add(
    LOG_DIR / "app.log",
    level="DEBUG",
    rotation="5 MB",
    retention=5,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
)

__all__ = ["logger"]
