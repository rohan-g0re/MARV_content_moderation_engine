import logging
import logging.handlers
import json
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        # Safely access extra attributes using getattr
        extra_data = getattr(record, "extra", None)
        if extra_data:
            log_record.update(extra_data)
        return json.dumps(log_record)

def get_logger(name="moderation", log_file="moderation.log"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.INFO)
    
    # Create rotating file handler
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    
    # Create console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Use JSON formatter for file, simple formatter for console
    json_formatter = JsonFormatter()
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    handler.setFormatter(json_formatter)
    console_handler.setFormatter(simple_formatter)
    
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger