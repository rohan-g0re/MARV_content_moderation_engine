"""
Post model for content storage and moderation results
"""

from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Float, JSON
from sqlalchemy.sql import func
from app.core.database import Base


class Post(Base):
    """Post model for storing content and moderation results"""
    
    __tablename__ = "posts"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Content information
    content = Column(Text, nullable=False, index=True)
    content_type = Column(String(50), default="text", nullable=False)
    user_id = Column(String(100), nullable=True, index=True)
    
    # Moderation results
    action = Column(String(20), nullable=False, index=True)  # ACCEPT, FLAG, BLOCK, QUARANTINE
    threat_level = Column(String(20), nullable=False, index=True)  # LOW, MEDIUM, HIGH, CRITICAL
    confidence = Column(Float, nullable=False, default=0.0)
    explanation = Column(Text, nullable=False)
    
    # Processing information
    processing_time_ms = Column(Integer, nullable=False, default=0)
    post_metadata = Column(JSON, nullable=True)  # Store additional data as JSON (renamed from 'metadata')
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<Post(id={self.id}, action='{self.action}', threat_level='{self.threat_level}')>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "content_type": self.content_type,
            "user_id": self.user_id,
            "action": self.action,
            "threat_level": self.threat_level,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "processing_time_ms": self.processing_time_ms,
            "post_metadata": self.post_metadata,  # renamed
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ModerationRule(Base):
    """Moderation rules for filtering content"""
    
    __tablename__ = "moderation_rules"
    
    # Primary key
    id = Column(String(100), primary_key=True)
    
    # Rule information
    rule_type = Column(String(50), nullable=False, index=True)  # profanity, threat, fraud, spam, etc.
    pattern = Column(Text, nullable=False)
    severity = Column(Integer, nullable=False, default=1)  # 1-10 scale
    
    # Configuration
    is_regex = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True, index=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<ModerationRule(id='{self.id}', type='{self.rule_type}', severity={self.severity})>"


class ModerationLog(Base):
    """Detailed moderation logs for analysis"""
    
    __tablename__ = "moderation_logs"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Reference to post
    post_id = Column(Integer, nullable=False, index=True)
    
    # Layer results
    rule_matches = Column(JSON, nullable=True)  # Rule-based filter results
    ml_scores = Column(JSON, nullable=True)     # ML model scores
    llm_feedback = Column(JSON, nullable=True)  # LLM analysis results
    
    # Processing details
    layer_processing_times = Column(JSON, nullable=True)  # Time per layer
    final_decision = Column(String(20), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    def __repr__(self):
        return f"<ModerationLog(id={self.id}, post_id={self.post_id}, decision='{self.final_decision}')>" 