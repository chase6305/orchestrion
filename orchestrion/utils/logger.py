import asyncio
import logging

try:
    import colorlog
except ImportError:  # Color is optional; standard logging remains functional.
    colorlog = None

logger = logging.getLogger("Orchestrion")
if not logger.handlers:
    handler = logging.StreamHandler()
    if colorlog is not None:
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_error(message: str, *args: object) -> None:
    """Report asynchronous failures without letting a log sink break completion."""
    try:
        logger.error(message, *args)
    except (Exception, asyncio.CancelledError):
        # Application-owned handlers may fail; error state remains in the result.
        return
