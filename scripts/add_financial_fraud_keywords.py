#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add financial fraud keywords to the moderation_rules database
"""

import json
import logging
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.post import ModerationRule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Financial fraud keywords to add
FINANCIAL_FRAUD_KEYWORDS = [
    # Pump and dump patterns
    "guarantee", "guaranteed", "guarantees", "guaranteeing",
    "insider", "insiders", "insider sources", "insider information",
    "parabolic", "parabolically", "parabolic move",
    "pump", "pumping", "pumped", "pumps",
    "dump", "dumping", "dumped", "dumps",
    "moon", "mooning", "to the moon", "rocket", "rocketing",
    "explode", "exploding", "exploded", "explosion",
    
    # Unrealistic returns
    "10x", "20x", "50x", "100x", "1000x",
    "10x return", "20x return", "50x return", "100x return",
    "get rich", "get rich quick", "get rich fast",
    "overnight millionaire", "instant wealth", "quick money",
    "easy money", "fast money", "quick profit",
    
    # Market manipulation
    "breakthrough", "revolutionary", "game changer",
    "next big thing", "once in a lifetime", "don't miss out",
    "limited time", "act now", "buy now", "buy as much as you can",
    "loading up", "loading up big", "massive gains",
    "massive returns", "huge profits", "enormous gains",
    
    # Investment fraud
    "investment opportunity", "amazing opportunity", "incredible opportunity",
    "hot stock", "hot tip", "stock tip", "investment tip",
    "penny stock", "penny stocks", "penny stock alert",
    "stock alert", "stock alerts", "trading alert",
    
    # Urgency and pressure
    "do not miss", "don't miss", "miss out", "missing out",
    "last chance", "final chance", "limited opportunity",
    "act fast", "hurry", "urgent", "emergency",
    "time sensitive", "time limited", "deadline",
    
    # Social proof manipulation
    "everyone is buying", "everyone buying", "everyone knows",
    "tell everyone", "share with friends", "spread the word",
    "viral", "going viral", "trending", "hot trending",
    
    # Crypto/stock confusion
    "crypto", "cryptocurrency", "bitcoin", "ethereum",
    "blockchain", "token", "ico", "initial coin offering",
    "stock market", "trading", "day trading", "swing trading",
    
    # Emotional manipulation
    "fomo", "fear of missing out", "fear of missing",
    "regret", "regretting", "wish you had", "should have",
    "lucky", "lucky break", "lucky opportunity",
    
    # Authority claims
    "expert", "experts", "professional", "professionals",
    "analyst", "analysts", "trader", "traders",
    "guru", "gurus", "master", "masters",
    
    # Specific fraud patterns
    "pump and dump", "pump & dump", "pump-and-dump",
    "market manipulation", "price manipulation",
    "insider trading", "illegal trading", "fraudulent",
    "scam", "scams", "scamming", "scammer",
    "ponzi", "ponzi scheme", "pyramid", "pyramid scheme"
]

# Financial fraud regex patterns
FINANCIAL_FRAUD_REGEX = [
    {
        "id": "regex_unrealistic_returns",
        "pattern": r"\b\d{1,3}x\s*(?:return|gain|profit|increase)\b",
        "severity": 8,
        "description": "Unrealistic return promises",
        "category": "financial_fraud"
    },
    {
        "id": "regex_percentage_gains",
        "pattern": r"\b(?:guaranteed|promised)\s+\d{1,4}%\s*(?:return|gain|profit)\b",
        "severity": 9,
        "description": "Guaranteed percentage returns",
        "category": "financial_fraud"
    },
    {
        "id": "regex_urgency_pressure",
        "pattern": r"\b(?:act\s+now|hurry|urgent|limited\s+time|don't\s+miss)\b",
        "severity": 6,
        "description": "Urgency and pressure tactics",
        "category": "financial_fraud"
    },
    {
        "id": "regex_insider_claims",
        "pattern": r"\b(?:insider\s+(?:source|information|tip)|guaranteed\s+(?:profit|return|gain))\b",
        "severity": 9,
        "description": "Insider trading claims",
        "category": "financial_fraud"
    },
    {
        "id": "regex_pump_language",
        "pattern": r"\b(?:pump|moon|rocket|explode|parabolic|breakthrough)\b",
        "severity": 7,
        "description": "Pump and dump language",
        "category": "financial_fraud"
    }
]

def add_financial_fraud_keywords():
    """Add financial fraud keywords to the database"""
    logger.info("Adding financial fraud keywords to database...")
    
    db = next(get_db())
    
    try:
        # Add keywords
        keyword_count = 0
        for i, keyword in enumerate(FINANCIAL_FRAUD_KEYWORDS, 1):
            rule_id = f"financial_fraud_keyword_{i}"
            
            # Check if rule already exists
            existing = db.query(ModerationRule).filter(ModerationRule.id == rule_id).first()
            if existing:
                logger.info(f"Rule already exists: {rule_id}")
                continue
            
            rule = ModerationRule(
                id=rule_id,
                rule_type="keyword",
                pattern=keyword.lower(),
                severity=8,  # High severity for financial fraud
                is_regex=False,
                is_active=True,
                description=f"Financial fraud keyword: {keyword}",
                category="financial_fraud"
            )
            
            db.add(rule)
            keyword_count += 1
            
            if keyword_count % 50 == 0:
                logger.info(f"Added {keyword_count} keywords...")
        
        # Add regex patterns
        regex_count = 0
        for pattern_data in FINANCIAL_FRAUD_REGEX:
            # Check if rule already exists
            existing = db.query(ModerationRule).filter(ModerationRule.id == pattern_data["id"]).first()
            if existing:
                logger.info(f"Regex rule already exists: {pattern_data['id']}")
                continue
            
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
            regex_count += 1
        
        db.commit()
        
        logger.info(f"Successfully added {keyword_count} keywords and {regex_count} regex patterns")
        logger.info("Financial fraud detection enhanced!")
        
    except Exception as e:
        logger.error(f"Error adding financial fraud keywords: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    add_financial_fraud_keywords() 