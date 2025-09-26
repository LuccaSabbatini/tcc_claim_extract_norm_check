"""Utility functions for file operations."""

# Imports
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Utilitary Functions
def move_file_to_directory(source_path, destination_path):
    """Move a file from source_path to destination_path."""
    try:
        os.rename(source_path, destination_path)
        logging.info("Moved file to %s", destination_path)
    except Exception as e:
        logging.error("Error moving file to %s: %s", destination_path, e)
