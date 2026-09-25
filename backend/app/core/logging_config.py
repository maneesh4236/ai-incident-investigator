"""
Production-style logging setup using loguru.

Keeps a single, consistent log format across the whole backend and exposes
`get_logger(name)` so every module logs with its own context tag.
"""
import sys

from loguru import logger

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[component]}</cyan> | "
    "<level>{message}</level>"
)


def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.configure(extra={"component": "app"})
    logger.add(sys.stdout, format=_LOG_FORMAT, level=level, backtrace=False, diagnose=False)


def get_logger(component: str):
    return logger.bind(component=component)
