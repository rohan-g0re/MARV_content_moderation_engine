#!/usr/bin/env python3
"""
Setup script for GuardianAI Database
Initializes the database with sample moderation rules
"""

import sqlite3
import os
from guardian_core import initialize_guardian_db

def setup_guardian_database():
    """Initialize GuardianAI database with sample rules"""
    print("🛡️ Setting up GuardianAI Database...")
    
    # Initialize database schema
    initialize_guardian_db("guardian.db")
    
    # Connect to database
    conn = sqlite3.connect("guardian.db")
    cursor = conn.cursor()
    
    # Sample moderation rules
    sample_rules = [
        # Profanity rules
        ("profanity_1", "profanity", "damn", 3, 0, 1),
        ("profanity_2", "profanity", "hell", 2, 0, 1),
        ("profanity_3", "profanity", "crap", 2, 0, 1),
        
        # Threat rules
        ("threat_1", "threat", "hack", 5, 0, 1),
        ("threat_2", "threat", "kill", 8, 0, 1),
        ("threat_3", "threat", "destroy", 6, 0, 1),
        ("threat_4", "threat", "attack", 7, 0, 1),
        
        # Financial fraud rules
        ("fraud_1", "fraud", "credit card", 4, 0, 1),
        ("fraud_2", "fraud", "social security", 5, 0, 1),
        ("fraud_3", "fraud", "bank account", 4, 0, 1),
        ("fraud_4", "fraud", "password", 3, 0, 1),
        
        # Spam rules
        ("spam_1", "spam", "buy now", 2, 0, 1),
        ("spam_2", "spam", "click here", 2, 0, 1),
        ("spam_3", "spam", "limited time", 2, 0, 1),
        
        # Violence rules
        ("violence_1", "violence", "punch", 4, 0, 1),
        ("violence_2", "violence", "stab", 7, 0, 1),
        ("violence_3", "violence", "shoot", 8, 0, 1),
        
        # Regex patterns
        ("regex_1", "url", r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", 3, 1, 1),
        ("regex_2", "email", r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b", 2, 1, 1),
        ("regex_3", "phone", r"\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b", 2, 1, 1),
    ]
    
    try:
        # Insert sample rules
        cursor.executemany("""
            INSERT OR REPLACE INTO moderation_rules 
            (id, rule_type, pattern, severity, is_regex, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
        """, sample_rules)
        
        # Commit changes
        conn.commit()
        
        # Verify rules were inserted
        cursor.execute("SELECT COUNT(*) FROM moderation_rules WHERE is_active = 1")
        rule_count = cursor.fetchone()[0]
        
        print(f"✅ Successfully inserted {rule_count} moderation rules")
        
        # Show rule categories
        cursor.execute("""
            SELECT rule_type, COUNT(*) as count 
            FROM moderation_rules 
            WHERE is_active = 1 
            GROUP BY rule_type
        """)
        
        print("\n📊 Rule Categories:")
        for rule_type, count in cursor.fetchall():
            print(f"  {rule_type}: {count} rules")
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        conn.rollback()
    finally:
        conn.close()
    
    print("\n✅ GuardianAI Database setup completed!")

def verify_database():
    """Verify the database setup"""
    print("\n🔍 Verifying database setup...")
    
    conn = sqlite3.connect("guardian.db")
    cursor = conn.cursor()
    
    try:
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"📋 Tables found: {', '.join(tables)}")
        
        # Check rule count
        cursor.execute("SELECT COUNT(*) FROM moderation_rules WHERE is_active = 1")
        rule_count = cursor.fetchone()[0]
        print(f"📊 Active rules: {rule_count}")
        
        # Show sample rules
        cursor.execute("""
            SELECT rule_type, pattern, severity 
            FROM moderation_rules 
            WHERE is_active = 1 
            LIMIT 5
        """)
        
        print("\n📝 Sample Rules:")
        for rule_type, pattern, severity in cursor.fetchall():
            print(f"  {rule_type}: '{pattern}' (severity: {severity})")
        
    except Exception as e:
        print(f"❌ Error verifying database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    setup_guardian_database()
    verify_database()
    
    print("\n🎉 Database is ready for GuardianAI Core Pipeline!")
    print("You can now run: python guardian_core.py") 