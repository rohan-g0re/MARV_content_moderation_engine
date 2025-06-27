"""
Database setup script for MARV Content Moderation Engine
"""

import sys
import os
import argparse
import logging
import asyncio
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import create_tables, get_database_info
from app.models.post import ModerationRule
from app.core.database import SessionLocal
from app.services.rule_service import RuleService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_rules():
    """Create sample moderation rules"""
    sample_rules = [
        # Profanity rules
        {
            "id": "profanity_1",
            "rule_type": "profanity",
            "pattern": "damn",
            "severity": 3,
            "is_regex": False,
            "description": "Mild profanity",
            "category": "language"
        },
        {
            "id": "profanity_2",
            "rule_type": "profanity",
            "pattern": "hell",
            "severity": 2,
            "is_regex": False,
            "description": "Mild profanity",
            "category": "language"
        },
        {
            "id": "profanity_3",
            "rule_type": "profanity",
            "pattern": "shit",
            "severity": 5,
            "is_regex": False,
            "description": "Strong profanity",
            "category": "language"
        },
        
        # Threat rules
        {
            "id": "threat_1",
            "rule_type": "threat",
            "pattern": "kill",
            "severity": 8,
            "is_regex": False,
            "description": "Violent threat",
            "category": "violence"
        },
        {
            "id": "threat_2",
            "rule_type": "threat",
            "pattern": "hack",
            "severity": 5,
            "is_regex": False,
            "description": "Cyber threat",
            "category": "cybersecurity"
        },
        {
            "id": "threat_3",
            "rule_type": "threat",
            "pattern": "bomb",
            "severity": 9,
            "is_regex": False,
            "description": "Explosive threat",
            "category": "violence"
        },
        
        # Fraud rules
        {
            "id": "fraud_1",
            "rule_type": "fraud",
            "pattern": "credit card",
            "severity": 4,
            "is_regex": False,
            "description": "Financial fraud",
            "category": "financial"
        },
        {
            "id": "fraud_2",
            "rule_type": "fraud",
            "pattern": "social security",
            "severity": 5,
            "is_regex": False,
            "description": "Identity theft",
            "category": "financial"
        },
        {
            "id": "fraud_3",
            "rule_type": "fraud",
            "pattern": "bank account",
            "severity": 4,
            "is_regex": False,
            "description": "Financial fraud",
            "category": "financial"
        },
        
        # Spam rules
        {
            "id": "spam_1",
            "rule_type": "spam",
            "pattern": "click here",
            "severity": 3,
            "is_regex": False,
            "description": "Spam link",
            "category": "spam"
        },
        {
            "id": "spam_2",
            "rule_type": "spam",
            "pattern": "make money fast",
            "severity": 4,
            "is_regex": False,
            "description": "Spam content",
            "category": "spam"
        },
        
        # Regex patterns
        {
            "id": "regex_url",
            "rule_type": "url",
            "pattern": r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "severity": 3,
            "is_regex": True,
            "description": "URL detection",
            "category": "links"
        },
        {
            "id": "regex_email",
            "rule_type": "email",
            "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "severity": 2,
            "is_regex": True,
            "description": "Email detection",
            "category": "personal_info"
        },
        {
            "id": "regex_phone",
            "rule_type": "phone",
            "pattern": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "severity": 2,
            "is_regex": True,
            "description": "Phone number detection",
            "category": "personal_info"
        }
    ]
    
    return sample_rules


def setup_database(force_reset=False):
    """Setup database with tables and sample data"""
    try:
        logger.info("🚀 Setting up MARV Content Moderation Engine database...")
        
        # Get database info
        db_info = get_database_info()
        logger.info(f"Database: {db_info['url']}")
        logger.info(f"Type: {'SQLite' if db_info['is_sqlite'] else 'PostgreSQL'}")
        
        # Create tables
        logger.info("Creating database tables...")
        create_tables()
        logger.info("✅ Database tables created successfully")
        
        # Check if rules already exist
        db = SessionLocal()
        existing_rules = db.query(ModerationRule).count()
        
        if existing_rules > 0 and not force_reset:
            logger.info(f"Database already contains {existing_rules} rules. Skipping sample data creation.")
            logger.info("Use --force-reset to recreate all sample data.")
            return
        
        # Clear existing rules if force reset
        if force_reset and existing_rules > 0:
            logger.info("Clearing existing rules...")
            db.query(ModerationRule).delete()
            db.commit()
        
        # Create sample rules
        logger.info("Creating sample moderation rules...")
        sample_rules = create_sample_rules()
        
        for rule_data in sample_rules:
            rule = ModerationRule(
                id=rule_data["id"],
                rule_type=rule_data["rule_type"],
                pattern=rule_data["pattern"],
                severity=rule_data["severity"],
                is_regex=rule_data["is_regex"],
                description=rule_data["description"],
                category=rule_data["category"]
            )
            db.add(rule)
        
        db.commit()
        logger.info(f"✅ Created {len(sample_rules)} sample rules")
        
        # Verify setup
        total_rules = db.query(ModerationRule).count()
        active_rules = db.query(ModerationRule).filter(ModerationRule.is_active == True).count()
        
        logger.info(f"📊 Database setup complete:")
        logger.info(f"   - Total rules: {total_rules}")
        logger.info(f"   - Active rules: {active_rules}")
        
        # Show rule categories
        categories = db.query(ModerationRule.category).distinct().all()
        logger.info(f"   - Rule categories: {', '.join([cat[0] for cat in categories if cat[0]])}")
        
        db.close()
        
        logger.info("🎉 Database setup completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        raise


async def test_database():
    """Test database functionality"""
    try:
        logger.info("🧪 Testing database functionality...")
        
        db = SessionLocal()
        
        # Test rule service
        rule_service = RuleService()
        
        # Test content analysis
        test_content = "This is a test message with some concerning words like damn and hack."
        result = await rule_service.analyze(test_content)
        
        logger.info(f"Test analysis result:")
        logger.info(f"  - Total severity: {result['total_severity']}")
        logger.info(f"  - Confidence: {result['confidence']}")
        logger.info(f"  - Matches: {len(result['matches'])}")
        
        for match in result['matches']:
            logger.info(f"    - {match['type']}: '{match['matched_text']}' (severity: {match['severity']})")
        
        db.close()
        logger.info("✅ Database test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Setup MARV Content Moderation Engine database")
    parser.add_argument(
        "--force-reset",
        action="store_true",
        help="Force reset and recreate all sample data"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run database tests after setup"
    )
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="Use PostgreSQL instead of SQLite"
    )
    
    args = parser.parse_args()
    
    try:
        # Setup database
        setup_database(force_reset=args.force_reset)
        
        # Run tests if requested
        if args.test:
            asyncio.run(test_database())
        
        logger.info("🎯 Setup completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Start the API: uvicorn app.main:app --reload")
        logger.info("2. Visit: http://localhost:8000/docs")
        logger.info("3. Test moderation: POST /api/v1/moderate")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 