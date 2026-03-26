import logging
import os
import sys
import io
from logging.handlers import RotatingFileHandler

# Set to True by interface.dashboard.start_dashboard() to suppress new StreamHandlers
_dashboard_active: bool = False


def get_logger(name: str, cfg: dict = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = logging.INFO
    log_file = "logs/mt5_bot_mk2.log"
    max_bytes = 10 * 1024 * 1024
    backup_count = 5

    if cfg:
        level = getattr(logging, cfg.get("level", "INFO").upper(), logging.INFO)
        log_file = cfg.get("file", log_file)
        max_bytes = cfg.get("max_bytes", max_bytes)
        backup_count = cfg.get("backup_count", backup_count)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count,
                             encoding="utf-8")
    fh.setFormatter(fmt)
    logger.setLevel(level)
    logger.addHandler(fh)

    # Only add console output when the Rich dashboard is not active
    if not _dashboard_active:
        # Force UTF-8 so Unicode chars (—, →, etc.) don't crash on cp1252 Windows terminals
        if hasattr(sys.stdout, "buffer"):
            stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace", line_buffering=True)
        else:
            stream = sys.stdout
        ch = logging.StreamHandler(stream=stream)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger
