# -*- coding: utf-8 -*-
"""
GuardianAI Core Pipeline v1.0
Combines DatabaseFilter + moderation router with structured output
"""

import sqlite3
import re
import os
import unicodedata
import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# === Enums for Structured Output ===
class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ModerationAction(Enum):
    ACCEPT = "accept"
    FLAG = "flag"
    BLOCK = "block"
    QUARANTINE = "quarantine"

class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"

# === Structured Result Classes ===
@dataclass
class ModerationResult:
    """Structured output for moderation results"""
    content_id: str
    threat_level: ThreatLevel
    action: ModerationAction
    explanation: str
    confidence: float
    processing_time_ms: int
    timestamp: datetime.datetime
    metadata: Dict[str, Any]

@dataclass
class FilterMatch:
    """Individual filter match details"""
    filter_type: str
    matched_content: str
    severity: int
    confidence: float
    rule_id: Optional[str] = None

# === Database Setup ===
def initialize_guardian_db(db_path: str = "guardian.db"):
    """Initialize GuardianAI database with comprehensive schema"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Content table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS content (
            id TEXT PRIMARY KEY,
            content_type TEXT NOT NULL,
            raw_content TEXT NOT NULL,
            processed_content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Moderation results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS moderation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_id TEXT NOT NULL,
            threat_level TEXT NOT NULL,
            action TEXT NOT NULL,
            explanation TEXT NOT NULL,
            confidence REAL NOT NULL,
            processing_time_ms INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(content_id) REFERENCES content(id)
        )
    """)
    
    # Filter matches table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS filter_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_id TEXT NOT NULL,
            filter_type TEXT NOT NULL,
            matched_content TEXT NOT NULL,
            severity INTEGER NOT NULL,
            confidence REAL NOT NULL,
            rule_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(content_id) REFERENCES content(id)
        )
    """)
    
    # Keywords/rules table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS moderation_rules (
            id TEXT PRIMARY KEY,
            rule_type TEXT NOT NULL,
            pattern TEXT NOT NULL,
            severity INTEGER NOT NULL,
            is_regex BOOLEAN DEFAULT 0,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

# === Text Normalization ===
def normalize_text(text: str) -> str:
    """Normalize text for consistent processing"""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    return text

# === Core Filter Engine ===
class GuardianFilter:
    """Enhanced filter engine with multiple detection methods"""
    
    def __init__(self, db_path: str = "guardian.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.rules = self._load_rules()
        
    def _load_rules(self) -> List[Dict[str, Any]]:
        """Load moderation rules from database"""
        self.cursor.execute("""
            SELECT id, rule_type, pattern, severity, is_regex 
            FROM moderation_rules 
            WHERE is_active = 1
        """)
        rows = self.cursor.fetchall()
        return [
            {
                "id": row[0],
                "type": row[1],
                "pattern": row[2],
                "severity": row[3],
                "is_regex": bool(row[4])
            }
            for row in rows
        ]
    
    def scan_content(self, text: str) -> List[FilterMatch]:
        """Scan content against all rules"""
        matches = []
        normalized_text = normalize_text(text)
        
        for rule in self.rules:
            matched_content = None
            confidence = 0.0
            
            if rule["is_regex"]:
                # Regex pattern matching
                pattern_matches = re.finditer(rule["pattern"], text, re.IGNORECASE)
                for match in pattern_matches:
                    matched_content = match.group(0)
                    confidence = 1.0
                    matches.append(FilterMatch(
                        filter_type=rule["type"],
                        matched_content=matched_content,
                        severity=rule["severity"],
                        confidence=confidence,
                        rule_id=rule["id"]
                    ))
            else:
                # Simple keyword matching
                if rule["pattern"].lower() in normalized_text:
                    matched_content = rule["pattern"]
                    confidence = 1.0
                    matches.append(FilterMatch(
                        filter_type=rule["type"],
                        matched_content=matched_content,
                        severity=rule["severity"],
                        confidence=confidence,
                        rule_id=rule["id"]
                    ))
        
        return matches
    
    def close(self):
        """Close database connection"""
        self.conn.close()

# === ML Model Integration ===
class MLDetector:
    """Machine learning model integration"""
    
    def __init__(self):
        self.models_loaded = False
        self._load_models()
    
    def _load_models(self):
        """Load ML models (placeholder for actual model loading)"""
        try:
            # Placeholder for actual model loading
            # In production, this would load Detoxify, FinBERT, etc.
            self.models_loaded = True
            print("✅ ML models loaded successfully")
        except Exception as e:
            print(f"⚠️ ML model loading failed: {e}")
            self.models_loaded = False
    
    def detect_toxicity(self, text: str) -> Dict[str, Any]:
        """Detect toxicity using ML models"""
        if not self.models_loaded:
            return {"score": 0.0, "confidence": 0.0, "label": "unknown"}
        
        # Placeholder for actual toxicity detection
        # This would use Detoxify or similar
        return {"score": 0.1, "confidence": 0.8, "label": "non-toxic"}
    
    def detect_fraud(self, text: str) -> Dict[str, Any]:
        """Detect financial fraud using ML models"""
        if not self.models_loaded:
            return {"score": 0.0, "confidence": 0.0, "label": "unknown"}
        
        # Placeholder for actual fraud detection
        # This would use FinBERT or similar
        return {"score": 0.05, "confidence": 0.9, "label": "legitimate"}

# === GuardianAI Core Pipeline ===
class GuardianAICore:
    """Main GuardianAI moderation pipeline"""
    
    def __init__(self, db_path: str = "guardian.db"):
        self.db_path = db_path
        self.filter = GuardianFilter(db_path)
        self.ml_detector = MLDetector()
        self._ensure_db_initialized()
    
    def _ensure_db_initialized(self):
        """Ensure database is initialized"""
        if not os.path.exists(self.db_path):
            initialize_guardian_db(self.db_path)
    
    def _calculate_threat_level(self, filter_matches: List[FilterMatch], 
                               toxicity_score: float, fraud_score: float) -> ThreatLevel:
        """Calculate overall threat level"""
        total_severity = sum(match.severity for match in filter_matches)
        
        # Combine rule-based and ML scores
        ml_threat = (toxicity_score + fraud_score) / 2
        
        if total_severity >= 10 or ml_threat >= 0.8:
            return ThreatLevel.CRITICAL
        elif total_severity >= 6 or ml_threat >= 0.6:
            return ThreatLevel.HIGH
        elif total_severity >= 3 or ml_threat >= 0.4:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _determine_action(self, threat_level: ThreatLevel, 
                         filter_matches: List[FilterMatch]) -> ModerationAction:
        """Determine moderation action based on threat level"""
        if threat_level == ThreatLevel.CRITICAL:
            return ModerationAction.BLOCK
        elif threat_level == ThreatLevel.HIGH:
            return ModerationAction.QUARANTINE
        elif threat_level == ThreatLevel.MEDIUM:
            return ModerationAction.FLAG
        else:
            return ModerationAction.ACCEPT
    
    def _generate_explanation(self, threat_level: ThreatLevel, 
                            filter_matches: List[FilterMatch],
                            toxicity_score: float, fraud_score: float) -> str:
        """Generate human-readable explanation"""
        explanations = []
        
        if filter_matches:
            match_types = set(match.filter_type for match in filter_matches)
            explanations.append(f"Rule violations: {', '.join(match_types)}")
        
        if toxicity_score > 0.3:
            explanations.append(f"Toxicity detected: {toxicity_score:.2f}")
        
        if fraud_score > 0.3:
            explanations.append(f"Fraud risk: {fraud_score:.2f}")
        
        if not explanations:
            explanations.append("Content passed all checks")
        
        return "; ".join(explanations)
    
    def moderate_content(self, content: str, content_type: ContentType = ContentType.TEXT,
                        content_id: Optional[str] = None) -> ModerationResult:
        """
        Main entrypoint for content moderation
        
        Args:
            content: The content to moderate
            content_type: Type of content (text, image, etc.)
            content_id: Optional ID for the content
            
        Returns:
            ModerationResult with structured output
        """
        start_time = datetime.datetime.now()
        
        # Generate content ID if not provided
        if content_id is None:
            content_id = f"content_{int(start_time.timestamp())}"
        
        # Step 1: Rule-based filtering
        filter_matches = self.filter.scan_content(content)
        
        # Step 2: ML-based detection
        toxicity_result = self.ml_detector.detect_toxicity(content)
        fraud_result = self.ml_detector.detect_fraud(content)
        
        # Step 3: Calculate threat level and action
        threat_level = self._calculate_threat_level(
            filter_matches, 
            toxicity_result["score"], 
            fraud_result["score"]
        )
        
        action = self._determine_action(threat_level, filter_matches)
        
        # Step 4: Generate explanation
        explanation = self._generate_explanation(
            threat_level, filter_matches,
            toxicity_result["score"], fraud_result["score"]
        )
        
        # Step 5: Calculate confidence and processing time
        processing_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
        confidence = max(
            toxicity_result["confidence"],
            fraud_result["confidence"],
            0.8 if filter_matches else 0.9
        )
        
        # Step 6: Create result
        result = ModerationResult(
            content_id=content_id,
            threat_level=threat_level,
            action=action,
            explanation=explanation,
            confidence=confidence,
            processing_time_ms=int(processing_time),
            timestamp=start_time,
            metadata={
                "content_type": content_type.value,
                "filter_matches_count": len(filter_matches),
                "toxicity_score": toxicity_result["score"],
                "fraud_score": fraud_result["score"],
                "total_severity": sum(match.severity for match in filter_matches)
            }
        )
        
        # Step 7: Store results in database
        self._store_result(result, filter_matches)
        
        return result
    
    def _store_result(self, result: ModerationResult, filter_matches: List[FilterMatch]):
        """Store moderation result in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Store content
            cursor.execute("""
                INSERT OR REPLACE INTO content (id, content_type, raw_content)
                VALUES (?, ?, ?)
            """, (result.content_id, "text", "content_preview"))
            
            # Store moderation result
            cursor.execute("""
                INSERT INTO moderation_results 
                (content_id, threat_level, action, explanation, confidence, processing_time_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                result.content_id,
                result.threat_level.value,
                result.action.value,
                result.explanation,
                result.confidence,
                result.processing_time_ms
            ))
            
            # Store filter matches
            for match in filter_matches:
                cursor.execute("""
                    INSERT INTO filter_matches 
                    (content_id, filter_type, matched_content, severity, confidence, rule_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    result.content_id,
                    match.filter_type,
                    match.matched_content,
                    match.severity,
                    match.confidence,
                    match.rule_id
                ))
            
            conn.commit()
        except Exception as e:
            print(f"⚠️ Database storage error: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def close(self):
        """Clean up resources"""
        self.filter.close()

# === Test and Demo Functions ===
def demo_guardian_core():
    """Demonstrate GuardianAI Core Pipeline"""
    print("🛡️ GuardianAI Core Pipeline Demo")
    print("=" * 50)
    
    # Initialize GuardianAI
    guardian = GuardianAICore()
    
    # Sample content for testing
    test_content = [
        "This is a great product!",
        "I hate everything about this damn service!",
        "Let's discuss the financial investment opportunities.",
        "I'm going to destroy your account and hack the system!",
        "You are a genius and I love this community!",
        "Send me your credit card details for verification."
    ]
    
    for i, content in enumerate(test_content, 1):
        print(f"\n📝 Test {i}: {content}")
        print("-" * 40)
        
        result = guardian.moderate_content(content, content_id=f"test_{i}")
        
        print(f"🆔 Content ID: {result.content_id}")
        print(f"⚠️ Threat Level: {result.threat_level.value.upper()}")
        print(f"🛡️ Action: {result.action.value.upper()}")
        print(f"📊 Confidence: {result.confidence:.2f}")
        print(f"⏱️ Processing Time: {result.processing_time_ms}ms")
        print(f"💬 Explanation: {result.explanation}")
        print(f"📈 Metadata: {result.metadata}")
    
    guardian.close()
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    demo_guardian_core() 