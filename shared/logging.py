"""Centralized logging setup using loguru."""

import sys
from loguru import logger


def setup_logger(name: str, level: str = "INFO") -> None:
    """Configure loguru logger for an agent or the orchestrator."""
    logger.remove()  # remove default handler
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            f"<cyan>{name}</cyan> | "
            "<level>{message}</level>"
        ),
    )
    logger.add(
        f"logs/{name}.log",
        level=level,
        rotation="10 MB",
        retention="7 days",
        format="{time} | {level} | {name} | {message}",
    )


__all__ = ["logger", "setup_logger"]
