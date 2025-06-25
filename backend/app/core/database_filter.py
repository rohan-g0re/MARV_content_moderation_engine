"""
Database-Driven Rule Filter for GuardianAI

This module implements the database-backed profanity and severity lookup system
as specified in the project brief for Day 2-3. It provides fast keyword lookup,
regex pattern matching, and scoring capabilities.

Features:
- Fast keyword lookup using SQLite database
- Regex pattern matching for complex rules
- Severity scoring based on multiple factors
- Caching for performance optimization
- Explainable results with detailed reasoning
"""

import re
import json
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import time
from collections import defaultdict

from .config import DatabaseConfig

@dataclass
class FilterResult:
    """Result from database filter analysis"""
    threat_score: float
    confidence: float
    explanation: str
    matched_patterns: List[str] = field(default_factory=list)
    matched_keywords: List[str] = field(default_factory=list)
    severity_level: str = "SAFE"
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            "threat_score": self.threat_score,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "matched_patterns": self.matched_patterns,
            "matched_keywords": self.matched_keywords,
            "severity_level": self.severity_level,
            "metadata": self.metadata
        }

class DatabaseFilter:
    """
    Database-Driven Rule Filter for fast keyword lookup and pattern matching
    
    This implements the core filtering logic as specified in the project brief:
    - Fast keyword lookup using SQLite
    - Regex pattern matching
    - Severity scoring
    - Explainable results
    """
    
    def __init__(self, config: DatabaseConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.keywords_cache = {}
        self.patterns_cache = {}
        self.severity_weights = {}
        
        # Load rules and initialize database
        self._load_rules()
        self._initialize_database()
        
        self.logger.info("DatabaseFilter initialized successfully")
    
    def _load_rules(self) -> None:
        """Load profanity patterns and severity keywords from JSON files"""
        try:
            # Load profanity patterns
            if Path(self.config.rules_file).exists():
                with open(self.config.rules_file, 'r', encoding='utf-8') as f:
                    rules_data = json.load(f)
                    self.profanity_patterns = rules_data.get('patterns', [])
                    self.severity_weights = rules_data.get('severity_weights', {})
            else:
                self.logger.warning(f"Rules file not found: {self.config.rules_file}")
                self.profanity_patterns = self._get_default_patterns()
                self.severity_weights = self._get_default_severity_weights()
            
            # Load severity keywords
            if Path(self.config.keywords_file).exists():
                with open(self.config.keywords_file, 'r', encoding='utf-8') as f:
                    keywords_data = json.load(f)
                    self.severity_keywords = keywords_data.get('keywords', {})
            else:
                self.logger.warning(f"Keywords file not found: {self.config.keywords_file}")
                self.severity_keywords = self._get_default_keywords()
            
            self.logger.info(f"Loaded {len(self.profanity_patterns)} patterns and {len(self.severity_keywords)} keyword categories")
            
        except Exception as e:
            self.logger.error(f"Failed to load rules: {e}")
            # Fallback to defaults
            self.profanity_patterns = self._get_default_patterns()
            self.severity_keywords = self._get_default_keywords()
            self.severity_weights = self._get_default_severity_weights()
    
    def _get_default_patterns(self) -> List[Dict]:
        """Get default profanity patterns if no file is provided"""
        return [
            {
                "pattern": r"\b(fuck|shit|bitch|asshole|dick|pussy)\b",
                "severity": "HIGH",
                "weight": 0.8,
                "description": "Strong profanity"
            },
            {
                "pattern": r"\b(damn|hell|crap|bullshit)\b",
                "severity": "MEDIUM",
                "weight": 0.4,
                "description": "Mild profanity"
            },
            {
                "pattern": r"\b(kill|murder|suicide|bomb|terrorist)\b",
                "severity": "CRITICAL",
                "weight": 1.0,
                "description": "Violent content"
            },
            {
                "pattern": r"\b(hack|crack|steal|fraud|scam)\b",
                "severity": "HIGH",
                "weight": 0.7,
                "description": "Illegal activities"
            },
            {
                "pattern": r"\b(hate|racist|sexist|homophobic)\b",
                "severity": "HIGH",
                "weight": 0.8,
                "description": "Hate speech"
            }
        ]
    
    def _get_default_keywords(self) -> Dict[str, List[str]]:
        """Get default severity keywords if no file is provided"""
        return {
            "CRITICAL": [
                "kill yourself", "commit suicide", "bomb threat", "terrorist attack",
                "shoot up", "mass shooting", "hate crime", "lynch"
            ],
            "HIGH": [
                "fuck you", "go to hell", "piece of shit", "worthless",
                "hack into", "steal money", "fraud scheme", "scam people"
            ],
            "MEDIUM": [
                "stupid", "idiot", "moron", "dumb", "annoying",
                "hate", "disgusting", "terrible", "awful"
            ],
            "LOW": [
                "bad", "wrong", "mistake", "problem", "issue",
                "concern", "worry", "upset", "angry"
            ]
        }
    
    def _get_default_severity_weights(self) -> Dict[str, float]:
        """Get default severity weights"""
        return {
            "CRITICAL": 1.0,
            "HIGH": 0.8,
            "MEDIUM": 0.5,
            "LOW": 0.2
        }
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database for fast keyword lookup"""
        try:
            # Extract database path from URL
            db_path = self.config.database_url.replace("sqlite:///", "")
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            # Create connection
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row
            
            # Create tables
            self._create_tables()
            
            # Populate database with keywords
            self._populate_keywords()
            
            self.logger.info(f"Database initialized: {db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            # Fallback to in-memory storage
            self.conn = None
    
    def _create_tables(self) -> None:
        """Create database tables for keyword lookup"""
        cursor = self.conn.cursor()
        
        # Keywords table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT UNIQUE NOT NULL,
                severity TEXT NOT NULL,
                weight REAL NOT NULL,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                severity TEXT NOT NULL,
                weight REAL NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON keywords(keyword)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_keywords_severity ON keywords(severity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_severity ON patterns(severity)")
        
        self.conn.commit()
    
    def _populate_keywords(self) -> None:
        """Populate database with keywords from loaded data"""
        cursor = self.conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM keywords")
        cursor.execute("DELETE FROM patterns")
        
        # Insert keywords
        for severity, keywords in self.severity_keywords.items():
            weight = self.severity_weights.get(severity, 0.5)
            for keyword in keywords:
                cursor.execute("""
                    INSERT OR IGNORE INTO keywords (keyword, severity, weight, category)
                    VALUES (?, ?, ?, ?)
                """, (keyword.lower(), severity, weight, "manual"))
        
        # Insert patterns
        for pattern_data in self.profanity_patterns:
            cursor.execute("""
                INSERT OR IGNORE INTO patterns (pattern, severity, weight, description)
                VALUES (?, ?, ?, ?)
            """, (
                pattern_data["pattern"],
                pattern_data["severity"],
                pattern_data["weight"],
                pattern_data["description"]
            ))
        
        self.conn.commit()
        self.logger.info("Database populated with keywords and patterns")
    
    def analyze(self, content: str) -> FilterResult:
        """
        Analyze content using database-driven rule filtering
        
        Args:
            content: Text content to analyze
            
        Returns:
            FilterResult with threat score, confidence, and explanation
        """
        start_time = time.time()
        
        # Normalize content
        content_lower = content.lower()
        words = content_lower.split()
        
        # Initialize results
        matched_keywords = []
        matched_patterns = []
        severity_scores = defaultdict(float)
        
        # 1. Keyword matching (fast database lookup)
        if self.config.enable_keyword_matching and self.conn:
            keyword_matches = self._match_keywords(content_lower)
            matched_keywords.extend(keyword_matches)
            
            # Calculate severity scores from keywords
            for keyword, severity, weight in keyword_matches:
                severity_scores[severity] += weight
        
        # 2. Pattern matching (regex)
        if self.config.enable_pattern_matching:
            pattern_matches = self._match_patterns(content_lower)
            matched_patterns.extend(pattern_matches)
            
            # Calculate severity scores from patterns
            for pattern, severity, weight in pattern_matches:
                severity_scores[severity] += weight
        
        # 3. Calculate final threat score
        threat_score = self._calculate_threat_score(severity_scores, len(words))
        
        # 4. Determine confidence and explanation
        confidence = self._calculate_confidence(matched_keywords, matched_patterns, threat_score)
        explanation = self._generate_explanation(matched_keywords, matched_patterns, threat_score)
        
        # 5. Determine severity level
        severity_level = self._determine_severity_level(threat_score)
        
        processing_time = (time.time() - start_time) * 1000
        
        result = FilterResult(
            threat_score=threat_score,
            confidence=confidence,
            explanation=explanation,
            matched_keywords=[kw[0] for kw in matched_keywords],
            matched_patterns=[pat[0] for pat in matched_patterns],
            severity_level=severity_level,
            metadata={
                "processing_time_ms": processing_time,
                "word_count": len(words),
                "severity_breakdown": dict(severity_scores)
            }
        )
        
        self.logger.debug(f"Analysis completed in {processing_time:.2f}ms: {threat_score:.3f}")
        
        return result
    
    def _match_keywords(self, content: str) -> List[Tuple[str, str, float]]:
        """Match keywords using database lookup"""
        matches = []
        
        try:
            cursor = self.conn.cursor()
            
            # Split content into words and phrases
            words = content.split()
            phrases = []
            
            # Generate phrases of different lengths (1-4 words)
            for i in range(len(words)):
                for j in range(i + 1, min(i + 5, len(words) + 1)):
                    phrase = " ".join(words[i:j])
                    phrases.append(phrase)
            
            # Check each phrase against database
            for phrase in phrases:
                cursor.execute("""
                    SELECT keyword, severity, weight 
                    FROM keywords 
                    WHERE keyword = ?
                """, (phrase,))
                
                result = cursor.fetchone()
                if result:
                    matches.append((result['keyword'], result['severity'], result['weight']))
            
        except Exception as e:
            self.logger.error(f"Keyword matching failed: {e}")
        
        return matches
    
    def _match_patterns(self, content: str) -> List[Tuple[str, str, float]]:
        """Match content against regex patterns"""
        matches = []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT pattern, severity, weight FROM patterns")
            patterns = cursor.fetchall()
            
            for pattern_row in patterns:
                pattern = pattern_row['pattern']
                severity = pattern_row['severity']
                weight = pattern_row['weight']
                
                try:
                    regex = re.compile(pattern, re.IGNORECASE)
                    if regex.search(content):
                        matches.append((pattern, severity, weight))
                except re.error as e:
                    self.logger.warning(f"Invalid regex pattern: {pattern} - {e}")
        
        except Exception as e:
            self.logger.error(f"Pattern matching failed: {e}")
        
        return matches
    
    def _calculate_threat_score(self, severity_scores: Dict[str, float], word_count: int) -> float:
        """Calculate final threat score from severity scores"""
        if not severity_scores:
            return 0.0
        
        # Weighted sum of severity scores
        total_score = 0.0
        max_possible_score = 0.0
        
        for severity, score in severity_scores.items():
            weight = self.severity_weights.get(severity, 0.5)
            total_score += score * weight
            max_possible_score += score
        
        # Normalize by word count to avoid bias towards longer content
        if word_count > 0:
            normalized_score = total_score / (word_count ** 0.5)  # Square root scaling
        else:
            normalized_score = total_score
        
        # Cap at 1.0
        return min(normalized_score, 1.0)
    
    def _calculate_confidence(self, keyword_matches: List, pattern_matches: List, threat_score: float) -> float:
        """Calculate confidence in the analysis result"""
        total_matches = len(keyword_matches) + len(pattern_matches)
        
        if total_matches == 0:
            return 1.0 if threat_score == 0.0 else 0.5
        
        # Higher confidence with more matches and higher threat scores
        base_confidence = min(0.8 + (threat_score * 0.2), 1.0)
        match_confidence = min(0.1 * total_matches, 0.2)
        
        return min(base_confidence + match_confidence, 1.0)
    
    def _generate_explanation(self, keyword_matches: List, pattern_matches: List, threat_score: float) -> str:
        """Generate human-readable explanation of the analysis"""
        if threat_score == 0.0:
            return "Content appears safe with no concerning patterns detected."
        
        explanations = []
        
        if keyword_matches:
            keyword_list = [kw[0] for kw in keyword_matches[:3]]  # Top 3 matches
            explanations.append(f"Matched concerning keywords: {', '.join(keyword_list)}")
        
        if pattern_matches:
            pattern_list = [pat[0] for pat in pattern_matches[:3]]  # Top 3 patterns
            explanations.append(f"Matched concerning patterns: {', '.join(pattern_list)}")
        
        if threat_score > 0.8:
            severity_desc = "highly concerning"
        elif threat_score > 0.5:
            severity_desc = "concerning"
        elif threat_score > 0.2:
            severity_desc = "potentially concerning"
        else:
            severity_desc = "slightly concerning"
        
        explanations.append(f"Overall threat level: {severity_desc} (score: {threat_score:.3f})")
        
        return ". ".join(explanations)
    
    def _determine_severity_level(self, threat_score: float) -> str:
        """Determine severity level based on threat score"""
        if threat_score >= 0.8:
            return "CRITICAL"
        elif threat_score >= 0.6:
            return "HIGH"
        elif threat_score >= 0.4:
            return "MEDIUM"
        elif threat_score >= 0.2:
            return "LOW"
        else:
            return "SAFE"
    
    def add_keyword(self, keyword: str, severity: str, weight: float = None) -> bool:
        """Add a new keyword to the database"""
        try:
            if weight is None:
                weight = self.severity_weights.get(severity, 0.5)
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO keywords (keyword, severity, weight, category)
                VALUES (?, ?, ?, ?)
            """, (keyword.lower(), severity, weight, "manual"))
            
            self.conn.commit()
            self.logger.info(f"Added keyword: {keyword} (severity: {severity})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add keyword: {e}")
            return False
    
    def remove_keyword(self, keyword: str) -> bool:
        """Remove a keyword from the database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM keywords WHERE keyword = ?", (keyword.lower(),))
            self.conn.commit()
            
            if cursor.rowcount > 0:
                self.logger.info(f"Removed keyword: {keyword}")
                return True
            else:
                self.logger.warning(f"Keyword not found: {keyword}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to remove keyword: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            cursor = self.conn.cursor()
            
            # Count keywords by severity
            cursor.execute("""
                SELECT severity, COUNT(*) as count 
                FROM keywords 
                GROUP BY severity
            """)
            keyword_stats = {row['severity']: row['count'] for row in cursor.fetchall()}
            
            # Count patterns by severity
            cursor.execute("""
                SELECT severity, COUNT(*) as count 
                FROM patterns 
                GROUP BY severity
            """)
            pattern_stats = {row['severity']: row['count'] for row in cursor.fetchall()}
            
            return {
                "keywords": keyword_stats,
                "patterns": pattern_stats,
                "total_keywords": sum(keyword_stats.values()),
                "total_patterns": sum(pattern_stats.values())
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close() 