"""
Keyword model for storing words and their severity classifications.
This is the master table containing original keywords from JSON with ML-assigned tiers.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base

class Keyword(Base):
    """
    Keywords table - stores original words with ML-assigned severity scores and tiers.
    
    Schema matches the specification:
    - id (Primary Key)
    - word (Original keyword from JSON)
    - tier (1, 2, or 3 - assigned by ML model)
    - severity_score (0-100 - assigned by ML model)
    - category (optional: "violence", "hate", "sexual", "drugs", etc.)
    - is_active (boolean - for easy enable/disable)
    """
    
    __tablename__ = "keywords"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Core word data
    word = Column(String(255), nullable=False, unique=True, index=True)
    
    # ML-assigned classifications
    tier = Column(Integer, nullable=False, index=True)  # 1, 2, or 3
    severity_score = Column(Integer, nullable=False, index=True)  # 0-100
    
    # Optional categorization
    category = Column(String(50), nullable=True, index=True)
    
    # Control flags
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationship to patterns (one keyword can have many patterns)
    patterns = relationship("Pattern", back_populates="keyword", cascade="all, delete-orphan")
    
    # Additional indexes for query optimization
    __table_args__ = (
        Index('idx_tier_severity', 'tier', 'severity_score'),  # Composite index for filtering
        Index('idx_word_active', 'word', 'is_active'),         # Fast lookup for active words
        Index('idx_category_tier', 'category', 'tier'),        # Category-based filtering
    )
    
    def __repr__(self):
        return f"<Keyword(id={self.id}, word='{self.word}', tier={self.tier}, score={self.severity_score})>"
    
    def __str__(self):
        return f"{self.word} (Tier {self.tier}, Score: {self.severity_score})"
    
    @property
    def tier_name(self) -> str:
        """Get human-readable tier name."""
        tier_names = {1: "High", 2: "Medium", 3: "Low"}
        return tier_names.get(self.tier, "Unknown")
    
    @property
    def severity_level(self) -> str:
        """Get severity level description based on score."""
        if self.severity_score >= 80:
            return "Critical"
        elif self.severity_score >= 60:
            return "High"
        elif self.severity_score >= 40:
            return "Medium"
        elif self.severity_score >= 20:
            return "Low"
        else:
            return "Minimal"
    
    def to_dict(self) -> dict:
        """Convert keyword to dictionary representation."""
        return {
            'id': self.id,
            'word': self.word,
            'tier': self.tier,
            'tier_name': self.tier_name,
            'severity_score': self.severity_score,
            'severity_level': self.severity_level,
            'category': self.category,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'pattern_count': len(self.patterns) if self.patterns else 0
        }
    
    @staticmethod
    def validate_tier(tier: int) -> bool:
        """Validate tier value."""
        return tier in [1, 2, 3]
    
    @staticmethod
    def validate_severity_score(score: int) -> bool:
        """Validate severity score value."""
        return 0 <= score <= 100
    
    @classmethod
    def create_from_ml_result(cls, ml_result: dict):
        """
        Create Keyword instance from ML classification result.
        
        Args:
            ml_result (dict): Result from ML classifier containing word, tier, severity_score
            
        Returns:
            Keyword: New keyword instance
        """
        return cls(
            word=ml_result['word'],
            tier=ml_result['tier'],
            severity_score=ml_result['severity_score'],
            category=ml_result.get('category'),  # Optional
            is_active=True
        ) 