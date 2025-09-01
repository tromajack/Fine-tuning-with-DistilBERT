# utils/logger.py
import logging
import sys

def setup_logger():
    """Configures and returns a logger for the application."""
    logger = logging.getLogger("bert-classifier")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler("app.log")

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger for other modules to import
app_logger = setup_logger()
