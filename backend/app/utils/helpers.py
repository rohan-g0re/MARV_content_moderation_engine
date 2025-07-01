"""
GuardianAI Utility Functions

Consolidated helper functions for the moderation system, including:
- Keyword management  
- Data loading utilities
"""

import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

def load_keywords_from_json(file_path: str = "data/external/words.json") -> List[str]:
    """
    Load keywords from JSON file
    
    Args:
        file_path: Path to the keywords JSON file
        
    Returns:
        List of keywords
    """
    try:
        path = Path(file_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                keywords = json.load(f)
            logger.info(f"Loaded {len(keywords)} keywords from {file_path}")
            return keywords
        else:
            logger.warning(f"Keywords file not found: {file_path}")
            return ["scammer", "fraud", "hate", "violence", "spam"]
    except Exception as e:
        logger.error(f"Error loading keywords: {e}")
        return ["scammer", "fraud", "hate", "violence", "spam"]

def update_keywords_json(keywords: List[str], file_path: str = "data/external/words.json") -> bool:
    """
    Update keywords JSON file
    
    Args:
        keywords: List of keywords to save
        file_path: Path to save the keywords
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(keywords, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated keywords file with {len(keywords)} keywords")
        return True
    except Exception as e:
        logger.error(f"Error updating keywords: {e}")
        return False

def add_keywords(new_keywords: List[str], file_path: str = "data/external/words.json") -> bool:
    """
    Add new keywords to existing list
    
    Args:
        new_keywords: List of new keywords to add
        file_path: Path to the keywords file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        existing_keywords = load_keywords_from_json(file_path)
        combined_keywords = list(set(existing_keywords + new_keywords))  # Remove duplicates
        return update_keywords_json(combined_keywords, file_path)
    except Exception as e:
        logger.error(f"Error adding keywords: {e}")
        return False

 