"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ThreatLevel(str, Enum):
    """Threat level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModerationAction(str, Enum):
    """Moderation action enumeration"""
    ACCEPT = "accept"
    FLAG = "flag"
    BLOCK = "block"
    QUARANTINE = "quarantine"


class ContentType(str, Enum):
    """Content type enumeration"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


# Request Schemas
class ModerationRequest(BaseModel):
    """Request schema for content moderation"""
    content: str = Field(..., min_length=1, max_length=10000, description="Content to moderate")
    content_type: ContentType = Field(default=ContentType.TEXT, description="Type of content")
    user_id: Optional[str] = Field(None, max_length=100, description="User identifier")
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()


class BatchModerationRequest(BaseModel):
    """Request schema for batch content moderation"""
    posts: List[ModerationRequest] = Field(..., min_items=1, max_items=100)
    
    @validator('posts')
    def validate_posts(cls, v):
        if len(v) > 100:
            raise ValueError('Maximum 100 posts per batch')
        return v


class RuleCreateRequest(BaseModel):
    """Request schema for creating moderation rules"""
    rule_id: str = Field(..., max_length=100, description="Unique rule identifier")
    rule_type: str = Field(..., max_length=50, description="Type of rule")
    pattern: str = Field(..., description="Pattern to match")
    severity: int = Field(..., ge=1, le=10, description="Severity level (1-10)")
    is_regex: bool = Field(default=False, description="Whether pattern is regex")
    description: Optional[str] = Field(None, description="Rule description")
    category: Optional[str] = Field(None, max_length=50, description="Rule category")


class RuleUpdateRequest(BaseModel):
    """Request schema for updating moderation rules"""
    pattern: Optional[str] = Field(None, description="Pattern to match")
    severity: Optional[int] = Field(None, ge=1, le=10, description="Severity level (1-10)")
    is_regex: Optional[bool] = Field(None, description="Whether pattern is regex")
    is_active: Optional[bool] = Field(None, description="Whether rule is active")
    description: Optional[str] = Field(None, description="Rule description")
    category: Optional[str] = Field(None, max_length=50, description="Rule category")


# Response Schemas
class ModerationResult(BaseModel):
    """Response schema for moderation results"""
    post_id: int = Field(..., description="Post identifier")
    action: ModerationAction = Field(..., description="Moderation action")
    threat_level: ThreatLevel = Field(..., description="Threat level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    explanation: str = Field(..., description="Human-readable explanation")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    created_at: datetime = Field(..., description="Timestamp of moderation")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PostResponse(BaseModel):
    """Response schema for post details"""
    id: int = Field(..., description="Post identifier")
    content: str = Field(..., description="Post content")
    content_type: ContentType = Field(..., description="Content type")
    user_id: Optional[str] = Field(None, description="User identifier")
    action: ModerationAction = Field(..., description="Moderation action")
    threat_level: ThreatLevel = Field(..., description="Threat level")
    confidence: float = Field(..., description="Confidence score")
    explanation: str = Field(..., description="Explanation")
    processing_time_ms: int = Field(..., description="Processing time")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchModerationResponse(BaseModel):
    """Response schema for batch moderation"""
    results: List[ModerationResult] = Field(..., description="Moderation results")
    total_posts: int = Field(..., description="Total number of posts processed")
    total_processing_time_ms: int = Field(..., description="Total processing time")
    summary: Dict[str, int] = Field(..., description="Summary by action")


class RuleResponse(BaseModel):
    """Response schema for moderation rules"""
    id: str = Field(..., description="Rule identifier")
    rule_type: str = Field(..., description="Rule type")
    pattern: str = Field(..., description="Pattern")
    severity: int = Field(..., description="Severity level")
    is_regex: bool = Field(..., description="Is regex pattern")
    is_active: bool = Field(..., description="Is active")
    description: Optional[str] = Field(None, description="Description")
    category: Optional[str] = Field(None, description="Category")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemStats(BaseModel):
    """Response schema for system statistics"""
    total_posts: int = Field(..., description="Total posts processed")
    posts_by_action: Dict[str, int] = Field(..., description="Posts by action")
    posts_by_threat_level: Dict[str, int] = Field(..., description="Posts by threat level")
    average_processing_time_ms: float = Field(..., description="Average processing time")
    total_rules: int = Field(..., description="Total moderation rules")
    active_rules: int = Field(..., description="Active moderation rules")
    system_uptime_seconds: int = Field(..., description="System uptime")


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str = Field(..., description="System status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    database_status: str = Field(..., description="Database status")
    ml_models_status: Dict[str, str] = Field(..., description="ML models status")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Response schema for errors"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 