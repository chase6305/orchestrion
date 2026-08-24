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
