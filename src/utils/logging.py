import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取带默认格式的 Logger。
    """
    logger = logging.getLogger(name if name is not None else __name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
