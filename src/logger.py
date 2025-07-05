"""Logging configuration for the scraper."""

import logging
import structlog
from pathlib import Path
from typing import Optional
from .config import Config

def setup_logging(log_level: Optional[str] = None, log_file: Optional[Path] = None) -> None:
    """Set up structured logging for the application."""
    
    # Use config defaults if not provided
    if log_level is None:
        log_level = Config.LOG_LEVEL
    
    if log_file is None:
        log_file = Config.LOG_FILE
    
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger for the given name."""
    return structlog.get_logger(name)