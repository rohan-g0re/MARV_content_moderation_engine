#!/usr/bin/env python3
"""
Import legacy rules from words.json and dictionary CSVs into moderation_rules table
"""

import sys
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import SessionLocal
from app.models.post import ModerationRule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_words_json(file_path: str) -> List[str]:
    """Load keywords from words.json file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            keywords = json.load(f)
        logger.info(f"Loaded {len(keywords)} keywords from {file_path}")
        return keywords
    except Exception as e:
        logger.warning(f"Could not load {file_path}: {e}")
        return []


def load_dictionary_csv(file_path: str) -> List[Dict[str, Any]]:
    """Load words and ratings from dictionary CSV file"""
    try:
        words_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                words_data.append({
                    'word': row.get('word', '').strip(),
                    'category': row.get('category', '').strip(),
                    'mean_rating': float(row.get('mean_rating', 0))
                })
        logger.info(f"Loaded {len(words_data)} words from {file_path}")
        return words_data
    except Exception as e:
        logger.warning(f"Could not load {file_path}: {e}")
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
        description=f"Legacy keyword: {keyword}",
        category="legacy_import"
    )


def create_rule_from_dictionary_word(word_data: Dict[str, Any], rule_id: str) -> ModerationRule:
    """Create a ModerationRule from dictionary word data"""
    word = word_data['word']
    category = word_data['category']
    rating = word_data['mean_rating']
    
    # Convert rating (0-10 scale) to severity (1-10 scale)
    severity = max(1, min(10, int(rating)))
    
    # Determine rule type based on category
    rule_type = "keyword"
    if category in ["deadline", "desperation", "pressure"]:
        rule_type = "stress"
    elif category in ["violence", "threat"]:
        rule_type = "threat"
    elif category in ["fraud", "financial"]:
        rule_type = "fraud"
    
    return ModerationRule(
        id=rule_id,
        rule_type=rule_type,
        pattern=word,
        severity=severity,
        is_regex=False,
        is_active=True,
        description=f"Dictionary word: {word} (category: {category}, rating: {rating})",
        category=category
    )


def import_legacy_rules(force_reset: bool = False):
    """Import all legacy rules into the moderation_rules table"""
    logger.info("Starting legacy rules import...")
    
    db = SessionLocal()
    
    try:
        # Check if rules already exist
        existing_rules = db.query(ModerationRule).filter(
            ModerationRule.category.in_(["legacy_import", "deadline", "desperation", "pressure"])
        ).count()
        
        if existing_rules > 0 and not force_reset:
            logger.info(f"Found {existing_rules} existing legacy rules. Use --force-reset to recreate.")
            return
        
        # Clear existing legacy rules if force reset
        if force_reset and existing_rules > 0:
            logger.info("Clearing existing legacy rules...")
            db.query(ModerationRule).filter(
                ModerationRule.category.in_(["legacy_import", "deadline", "desperation", "pressure"])
            ).delete()
            db.commit()
        
        rule_id_counter = 1
        
        # Import words.json
        words_json_paths = [
            "backend/words.json",
            "data/words.json"
        ]
        
        for json_path in words_json_paths:
            if Path(json_path).exists():
                keywords = load_words_json(json_path)
                logger.info(f"Importing {len(keywords)} keywords from {json_path}")
                
                for keyword in keywords:
                    if keyword.strip():  # Skip empty keywords
                        rule_id = f"legacy_keyword_{rule_id_counter}"
                        rule = create_rule_from_keyword(keyword, rule_id)
                        db.add(rule)
                        rule_id_counter += 1
                
                db.commit()
                logger.info(f"Imported {len(keywords)} keywords from {json_path}")
                break  # Use first found words.json
        
        # Import dictionary CSVs
        csv_paths = [
            "backend/dictionary_5plus.csv",
            "backend/dictionary_7plus.csv"
        ]
        
        for csv_path in csv_paths:
            if Path(csv_path).exists():
                words_data = load_dictionary_csv(csv_path)
                logger.info(f"Importing {len(words_data)} words from {csv_path}")
                
                for word_data in words_data:
                    if word_data['word'].strip():  # Skip empty words
                        rule_id = f"dict_{word_data['category']}_{rule_id_counter}"
                        rule = create_rule_from_dictionary_word(word_data, rule_id)
                        db.add(rule)
                        rule_id_counter += 1
                
                db.commit()
                logger.info(f"Imported {len(words_data)} words from {csv_path}")
        
        # Add some regex patterns from the legacy system
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
            },
            {
                "id": "regex_violence",
                "pattern": r"\b(kill|bash|hack|steal|threat|attack|rape|murder|shoot|stab|destroy|burn|harass|stalk|blackmail|assault|abuse|bully|rob|terror|terrorist|explosive|bomb|kidnap|extort)\b",
                "severity": 8,
                "description": "Violence/threat detection",
                "category": "violence"
            }
        ]
        
        for pattern_data in regex_patterns:
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
        
        db.commit()
        logger.info(f"Imported {len(regex_patterns)} regex patterns")
        
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
    
    parser = argparse.ArgumentParser(description="Import legacy rules into moderation_rules table")
    parser.add_argument(
        "--force-reset",
        action="store_true",
        help="Force reset and recreate all legacy rules"
    )
    
    args = parser.parse_args()
    
    try:
        import_legacy_rules(force_reset=args.force_reset)
        logger.info("Legacy rules import completed successfully!")
    except Exception as e:
        logger.error(f"Import failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 