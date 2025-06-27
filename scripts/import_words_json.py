#!/usr/bin/env python3
"""
Import words.json into moderation_rules table (simplified version)
"""

import sys
import json
import logging
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import SessionLocal
from app.models.post import ModerationRule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_words_json(file_path: str) -> list:
    """Load keywords from words.json file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            keywords = json.load(f)
        logger.info(f"Loaded {len(keywords)} keywords from {file_path}")
        return keywords
    except Exception as e:
        logger.error(f"Could not load {file_path}: {e}")
        return []


def create_rule_from_keyword(keyword: str, rule_id: str) -> ModerationRule:
    """Create a ModerationRule from a keyword"""
    # Determine severity based on keyword characteristics
    severity = 5  # Default medium severity
    
    # Adjust severity based on keyword type
    if any(word in keyword.lower() for word in ['kill', 'bomb', 'rape', 'murder', 'terror']):
        severity = 9  # Critical
    elif any(word in keyword.lower() for word in ['hate', 'violence', 'attack', 'assault']):
        severity = 8  # High
    elif any(word in keyword.lower() for word in ['fraud', 'scam', 'steal', 'hack']):
        severity = 7  # High
    elif any(word in keyword.lower() for word in ['spam', 'click', 'money']):
        severity = 4  # Medium
    elif any(word in keyword.lower() for word in ['damn', 'hell']):
        severity = 3  # Low
    
    return ModerationRule(
        id=rule_id,
        rule_type="keyword",
        pattern=keyword,
        severity=severity,
        is_regex=False,
        is_active=True,
        description=f"Keyword: {keyword}",
        category="keywords"
    )


def import_words_json(force_reset: bool = False):
    """Import words.json into the moderation_rules table"""
    logger.info("Starting words.json import...")
    
    db = SessionLocal()
    
    try:
        # Check if rules already exist
        existing_rules = db.query(ModerationRule).filter(
            ModerationRule.category == "keywords"
        ).count()
        
        if existing_rules > 0 and not force_reset:
            logger.info(f"Found {existing_rules} existing keyword rules. Use --force-reset to recreate.")
            return
        
        # Clear existing keyword rules if force reset
        if force_reset and existing_rules > 0:
            logger.info("Clearing existing keyword rules...")
            db.query(ModerationRule).filter(
                ModerationRule.category == "keywords"
            ).delete()
            db.commit()
        
        # Find words.json file
        words_json_paths = [
            "backend/words.json",
            "data/words.json"
        ]
        
        keywords = []
        for json_path in words_json_paths:
            if Path(json_path).exists():
                keywords = load_words_json(json_path)
                logger.info(f"Importing {len(keywords)} keywords from {json_path}")
                break
        
        if not keywords:
            logger.error("No words.json file found!")
            return
        
        # Import keywords
        rule_id_counter = 1
        for keyword in keywords:
            if keyword.strip():  # Skip empty keywords
                rule_id = f"keyword_{rule_id_counter}"
                rule = create_rule_from_keyword(keyword, rule_id)
                db.add(rule)
                rule_id_counter += 1
        
        db.commit()
        logger.info(f"Imported {len(keywords)} keywords")
        
        # Add some basic regex patterns
        regex_patterns = [
            {
                "id": "regex_url",
                "pattern": r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "severity": 3,
                "description": "URL detection",
                "category": "links"
            },
            {
                "id": "regex_email",
                "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "severity": 2,
                "description": "Email detection",
                "category": "personal_info"
            },
            {
                "id": "regex_phone",
                "pattern": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                "severity": 2,
                "description": "Phone number detection",
                "category": "personal_info"
            }
        ]
        
        # Check for existing regex patterns and only add if they don't exist
        for pattern_data in regex_patterns:
            existing_rule = db.query(ModerationRule).filter(
                ModerationRule.id == pattern_data["id"]
            ).first()
            
            if not existing_rule:
                rule = ModerationRule(
                    id=pattern_data["id"],
                    rule_type="regex",
                    pattern=pattern_data["pattern"],
                    severity=pattern_data["severity"],
                    is_regex=True,
                    is_active=True,
                    description=pattern_data["description"],
                    category=pattern_data["category"]
                )
                db.add(rule)
                logger.info(f"Added regex pattern: {pattern_data['id']}")
            else:
                logger.info(f"Regex pattern already exists: {pattern_data['id']}")
        
        db.commit()
        logger.info(f"Processed {len(regex_patterns)} regex patterns")
        
        # Final statistics
        total_rules = db.query(ModerationRule).count()
        active_rules = db.query(ModerationRule).filter(ModerationRule.is_active == True).count()
        
        logger.info("Import completed successfully!")
        logger.info(f"Total rules in database: {total_rules}")
        logger.info(f"Active rules: {active_rules}")
        
        # Show categories
        categories = db.query(ModerationRule.category).distinct().all()
        logger.info(f"Rule categories: {', '.join([cat[0] for cat in categories if cat[0]])}")
        
    except Exception as e:
        logger.error(f"Import failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Import words.json into moderation_rules table")
    parser.add_argument(
        "--force-reset",
        action="store_true",
        help="Force reset and recreate all keyword rules"
    )
    
    args = parser.parse_args()
    
    try:
        import_words_json(force_reset=args.force_reset)
        logger.info("Words.json import completed successfully!")
    except Exception as e:
        logger.error(f"Import failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 