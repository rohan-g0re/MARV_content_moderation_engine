"""
Pattern model for storing generated regex patterns from keywords.
This table stores all the auto-generated regex patterns for each keyword.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base

class Pattern(Base):
    """
    Patterns table - stores generated regex patterns from keywords.
    
    Schema matches the specification:
    - id (Primary Key)
    - keyword_id (Foreign Key to keywords table)
    - regex_pattern (The actual regex string)
    - pattern_type ("exact", "leetspeak", "spaced", "contextual")
    - tier (Inherited from parent keyword)
    - confidence (0.0-1.0 - how reliable this pattern is)
    """
    
    __tablename__ = "patterns"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Foreign key to keywords table
    keyword_id = Column(Integer, ForeignKey("keywords.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Pattern data
    regex_pattern = Column(String(1000), nullable=False, index=True)  # Longer length for complex patterns
    pattern_type = Column(String(50), nullable=False, index=True)
    
    # Classification (inherited from parent keyword)
    tier = Column(Integer, nullable=False, index=True)  # 1, 2, or 3
    
    # Pattern quality metrics
    confidence = Column(Float, nullable=False, default=1.0)  # 0.0 to 1.0
    
    # Control flags
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Performance metrics (for future optimization)
    match_count = Column(Integer, default=0, nullable=False)  # How many times this pattern matched
    false_positive_count = Column(Integer, default=0, nullable=False)  # Tracked false positives
    
    # Relationship back to keyword
    keyword = relationship("Keyword", back_populates="patterns")
    
    # Additional indexes for query optimization
    __table_args__ = (
        Index('idx_tier_type', 'tier', 'pattern_type'),        # Fast filtering by tier and type
        Index('idx_keyword_active', 'keyword_id', 'is_active'),# Fast lookup of active patterns for keyword
        Index('idx_tier_confidence', 'tier', 'confidence'),    # Filter by tier and confidence
        Index('idx_type_confidence', 'pattern_type', 'confidence'),  # Pattern type performance
    )
    
    def __repr__(self):
        return f"<Pattern(id={self.id}, keyword_id={self.keyword_id}, type='{self.pattern_type}', tier={self.tier})>"
    
    def __str__(self):
        return f"{self.pattern_type} pattern for keyword ID {self.keyword_id} (Tier {self.tier})"
    
    @property
    def tier_name(self) -> str:
        """Get human-readable tier name."""
        tier_names = {1: "High", 2: "Medium", 3: "Low"}
        return tier_names.get(self.tier, "Unknown")
    
    @property
    def confidence_level(self) -> str:
        """Get confidence level description."""
        if self.confidence >= 0.9:
            return "Very High"
        elif self.confidence >= 0.7:
            return "High"
        elif self.confidence >= 0.5:
            return "Medium"
        elif self.confidence >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    @property
    def effectiveness_ratio(self) -> float:
        """Calculate pattern effectiveness (true positives / total matches)."""
        if self.match_count == 0:
            return 0.0
        return max(0.0, (self.match_count - self.false_positive_count) / self.match_count)
    
    def to_dict(self) -> dict:
        """Convert pattern to dictionary representation."""
        return {
            'id': self.id,
            'keyword_id': self.keyword_id,
            'regex_pattern': self.regex_pattern,
            'pattern_type': self.pattern_type,
            'tier': self.tier,
            'tier_name': self.tier_name,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level,
            'is_active': self.is_active,
            'match_count': self.match_count,
            'false_positive_count': self.false_positive_count,
            'effectiveness_ratio': self.effectiveness_ratio,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'keyword_word': self.keyword.word if self.keyword else None
        }
    
    @staticmethod
    def validate_pattern_type(pattern_type: str) -> bool:
        """Validate pattern type value."""
        valid_types = ["exact", "leetspeak", "spaced", "contextual", "substitution", "boundary"]
        return pattern_type in valid_types
    
    @staticmethod
    def validate_tier(tier: int) -> bool:
        """Validate tier value."""
        return tier in [1, 2, 3]
    
    @staticmethod
    def validate_confidence(confidence: float) -> bool:
        """Validate confidence value."""
        return 0.0 <= confidence <= 1.0
    
    @classmethod
    def create_from_generation(cls, keyword_id: int, regex_pattern: str, pattern_type: str, 
                             tier: int, confidence: float = 1.0):
        """
        Create Pattern instance from pattern generation.
        
        Args:
            keyword_id (int): ID of parent keyword
            regex_pattern (str): Generated regex pattern
            pattern_type (str): Type of pattern generation used
            tier (int): Tier inherited from keyword
            confidence (float): Confidence score for this pattern
            
        Returns:
            Pattern: New pattern instance
        """
        return cls(
            keyword_id=keyword_id,
            regex_pattern=regex_pattern,
            pattern_type=pattern_type,
            tier=tier,
            confidence=confidence,
            is_active=True,
            match_count=0,
            false_positive_count=0
        )
    
    def record_match(self, is_false_positive: bool = False):
        """
        Record a pattern match for performance tracking.
        
        Args:
            is_false_positive (bool): Whether this match was a false positive
        """
        self.match_count += 1
        if is_false_positive:
            self.false_positive_count += 1
    
    def update_confidence(self, new_confidence: float):
        """
        Update pattern confidence based on performance.
        
        Args:
            new_confidence (float): New confidence score
        """
        if self.validate_confidence(new_confidence):
            self.confidence = new_confidence 