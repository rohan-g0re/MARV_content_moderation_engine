"""
Rule-based filtering service for content moderation
"""

import re
import logging
import unicodedata
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.database import get_db
from app.models.post import ModerationRule
from app.core.config import settings

logger = logging.getLogger(__name__)


class RuleService:
    """Service for rule-based content filtering"""
    
    def __init__(self):
        """Initialize rule service"""
        self.rules_cache = None
        self.cache_timestamp = 0
        self.cache_ttl = 300  # 5 minutes cache
        
        logger.info("RuleService initialized")
    
    async def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analyze content using rule-based filtering
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Normalize content
            normalized_content = self._normalize_text(content)
            
            # Get rules (with caching)
            rules = await self._get_rules()
            
            # Apply rules
            matches = []
            total_severity = 0
            
            for rule in rules:
                if not rule["is_active"]:
                    continue
                
                rule_matches = self._apply_rule(rule, content, normalized_content)
                if rule_matches:
                    matches.extend(rule_matches)
                    total_severity += rule["severity"] * len(rule_matches)
            
            # Calculate confidence based on matches
            confidence = min(0.9 + (len(matches) * 0.1), 1.0) if matches else 0.8
            
            return {
                "total_severity": total_severity,
                "confidence": confidence,
                "matches": matches,
                "rule_count": len(rules),
                "active_rules": len([r for r in rules if r["is_active"]])
            }
            
        except Exception as e:
            logger.error(f"Error in rule analysis: {e}")
            return {
                "total_severity": 0,
                "confidence": 0.0,
                "matches": [],
                "error": str(e)
            }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent matching"""
        # Convert to lowercase
        text = text.lower()
        
        # Normalize unicode characters
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        
        return text
    
    def _apply_rule(self, rule: Dict[str, Any], original_content: str, normalized_content: str) -> List[Dict[str, Any]]:
        """Apply a single rule to content"""
        matches = []
        
        try:
            if rule["is_regex"]:
                # Regex pattern matching
                pattern_matches = re.finditer(rule["pattern"], original_content, re.IGNORECASE)
                for match in pattern_matches:
                    matches.append({
                        "type": rule["rule_type"],
                        "pattern": rule["pattern"],
                        "matched_text": match.group(0),
                        "severity": rule["severity"],
                        "rule_id": rule["id"],
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                        "is_regex": True
                    })
            else:
                # Simple keyword matching
                keyword = rule["pattern"].lower()
                if keyword in normalized_content:
                    # Find all occurrences
                    start_pos = 0
                    while True:
                        pos = normalized_content.find(keyword, start_pos)
                        if pos == -1:
                            break
                        
                        matches.append({
                            "type": rule["rule_type"],
                            "pattern": rule["pattern"],
                            "matched_text": original_content[pos:pos+len(keyword)],
                            "severity": rule["severity"],
                            "rule_id": rule["id"],
                            "start_pos": pos,
                            "end_pos": pos + len(keyword),
                            "is_regex": False
                        })
                        
                        start_pos = pos + 1
                        
        except Exception as e:
            logger.error(f"Error applying rule {rule.get('id', 'unknown')}: {e}")
        
        return matches
    
    async def _get_rules(self) -> List[Dict[str, Any]]:
        """Get moderation rules from database (with caching)"""
        import time
        
        current_time = time.time()
        
        # Check if cache is still valid
        if (self.rules_cache is not None and 
            current_time - self.cache_timestamp < self.cache_ttl):
            return self.rules_cache
        
        try:
            # Get database session
            db = next(get_db())
            
            # Query active rules
            rules = db.query(ModerationRule).filter(ModerationRule.is_active == True).all()
            
            # Convert to dictionary format
            rules_dict = []
            for rule in rules:
                rules_dict.append({
                    "id": rule.id,
                    "rule_type": rule.rule_type,
                    "pattern": rule.pattern,
                    "severity": rule.severity,
                    "is_regex": rule.is_regex,
                    "is_active": rule.is_active,
                    "description": rule.description,
                    "category": rule.category
                })
            
            # Update cache
            self.rules_cache = rules_dict
            self.cache_timestamp = current_time
            
            logger.info(f"Loaded {len(rules_dict)} active rules")
            return rules_dict
            
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
            # Return empty list if database error
            return []
    
    async def get_rule_count(self) -> int:
        """Get total number of rules"""
        try:
            db = next(get_db())
            return db.query(ModerationRule).count()
        except Exception as e:
            logger.error(f"Error getting rule count: {e}")
            return 0
    
    async def add_rule(self, rule_data: Dict[str, Any]) -> bool:
        """Add a new moderation rule"""
        try:
            db = next(get_db())
            
            rule = ModerationRule(
                id=rule_data["rule_id"],
                rule_type=rule_data["rule_type"],
                pattern=rule_data["pattern"],
                severity=rule_data["severity"],
                is_regex=rule_data.get("is_regex", False),
                description=rule_data.get("description"),
                category=rule_data.get("category")
            )
            
            db.add(rule)
            db.commit()
            
            # Invalidate cache
            self.rules_cache = None
            
            logger.info(f"Added rule: {rule_data['rule_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding rule: {e}")
            return False
    
    async def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing rule"""
        try:
            db = next(get_db())
            
            rule = db.query(ModerationRule).filter(ModerationRule.id == rule_id).first()
            if not rule:
                return False
            
            # Update fields
            for field, value in updates.items():
                if hasattr(rule, field):
                    setattr(rule, field, value)
            
            db.commit()
            
            # Invalidate cache
            self.rules_cache = None
            
            logger.info(f"Updated rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating rule: {e}")
            return False
    
    async def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule (soft delete by setting is_active=False)"""
        try:
            db = next(get_db())
            
            rule = db.query(ModerationRule).filter(ModerationRule.id == rule_id).first()
            if not rule:
                return False
            
            rule.is_active = False
            db.commit()
            
            # Invalidate cache
            self.rules_cache = None
            
            logger.info(f"Deleted rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting rule: {e}")
            return False
    
    async def get_rules_by_type(self, rule_type: str) -> List[Dict[str, Any]]:
        """Get rules by type"""
        try:
            db = next(get_db())
            
            rules = db.query(ModerationRule).filter(
                ModerationRule.rule_type == rule_type,
                ModerationRule.is_active == True
            ).all()
            
            return [
                {
                    "id": rule.id,
                    "rule_type": rule.rule_type,
                    "pattern": rule.pattern,
                    "severity": rule.severity,
                    "is_regex": rule.is_regex,
                    "description": rule.description,
                    "category": rule.category
                }
                for rule in rules
            ]
            
        except Exception as e:
            logger.error(f"Error getting rules by type: {e}")
            return []
    
    def _load_default_rules(self) -> List[Dict[str, Any]]:
        """Load default moderation rules"""
        return [
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
                "pattern": r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b",
                "severity": 2,
                "is_regex": True,
                "description": "Email detection",
                "category": "personal_info"
            }
        ] 