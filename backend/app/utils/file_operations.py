"""
File operations utility module for content moderation engine.
Handles reading and processing of data files.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_words_from_json(file_path: str) -> Optional[List[str]]:
    """
    Load words from JSON file and return as list.
    
    Args:
        file_path (str): Path to the JSON file containing words
        
    Returns:
        Optional[List[str]]: List of words if successful, None if failed
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
    """
    try:
        # Convert to absolute path for better error handling
        abs_path = Path(file_path).resolve()
        
        if not abs_path.exists():
            logger.error(f"File not found: {abs_path}")
            raise FileNotFoundError(f"JSON file not found at: {abs_path}")
        
        logger.info(f"Loading words from: {abs_path}")
        
        with open(abs_path, 'r', encoding='utf-8') as file:
            words_data = json.load(file)
        
        # Validate that loaded data is a list
        if not isinstance(words_data, list):
            logger.error("JSON file should contain a list of words")
            raise ValueError("JSON file should contain a list of words")
        
        # Filter out empty strings and None values
        words_list = [word.strip() for word in words_data if word and isinstance(word, str) and word.strip()]
        
        logger.info(f"Successfully loaded {len(words_list)} words from JSON file")
        
        # Log some statistics
        avg_length = sum(len(word) for word in words_list) / len(words_list) if words_list else 0
        logger.info(f"Average word length: {avg_length:.2f} characters")
        
        return words_list
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading JSON file: {e}")
        raise

def validate_words_list(words: List[str]) -> dict:
    """
    Validate the loaded words list and return statistics.
    
    Args:
        words (List[str]): List of words to validate
        
    Returns:
        dict: Dictionary containing validation results and statistics
    """
    if not words:
        return {
            "is_valid": False,
            "error": "Empty words list",
            "count": 0,
            "statistics": {}
        }
    
    statistics = {
        "total_count": len(words),
        "unique_count": len(set(words)),
        "duplicates": len(words) - len(set(words)),
        "avg_length": sum(len(word) for word in words) / len(words),
        "min_length": min(len(word) for word in words),
        "max_length": max(len(word) for word in words),
        "empty_count": sum(1 for word in words if not word.strip())
    }
    
    return {
        "is_valid": True,
        "count": len(words),
        "statistics": statistics
    } 