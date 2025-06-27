"""
Moderation API endpoints
"""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.schemas import (
    ModerationRequest, ModerationResult, 
    BatchModerationRequest, BatchModerationResponse
)
from app.models.post import Post
from app.services.moderation_service import ModerationService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/moderate", response_model=ModerationResult)
async def moderate_content(
    request: ModerationRequest,
    db: Session = Depends(get_db),
    moderation_service: ModerationService = Depends(lambda: ModerationService())
):
    """
    Moderate a single piece of content
    
    This endpoint processes content through the multi-layer moderation pipeline:
    1. Rule-based filtering
    2. ML/AI detection (Detoxify + FinBERT)
    3. LLM escalation (if needed)
    
    Returns structured moderation results with threat level, action, and explanation.
    """
    try:
        logger.info(f"Moderation request received for content: {request.content[:100]}...")
        
        # Run moderation pipeline
        result = await moderation_service.moderate_content(request)
        
        # Save to database
        post = Post(
            content=request.content,
            content_type=request.content_type.value,
            user_id=request.user_id,
            action=result.action.value,
            threat_level=result.threat_level.value,
            confidence=result.confidence,
            explanation=result.explanation,
            processing_time_ms=result.processing_time_ms,
            post_metadata=result.metadata
        )
        
        db.add(post)
        db.commit()
        db.refresh(post)
        
        # Update result with post ID
        result.post_id = post.id
        
        logger.info(f"Moderation completed: {result.action} ({result.threat_level}) - Post ID: {post.id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in moderation endpoint: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Moderation failed: {str(e)}")


@router.post("/moderate/batch", response_model=BatchModerationResponse)
async def moderate_batch(
    request: BatchModerationRequest,
    db: Session = Depends(get_db),
    moderation_service: ModerationService = Depends(lambda: ModerationService())
):
    """
    Moderate multiple pieces of content in batch
    
    This endpoint processes multiple content items efficiently through the moderation pipeline.
    Maximum 100 posts per batch.
    """
    try:
        logger.info(f"Batch moderation request received for {len(request.posts)} posts")
        
        results = []
        total_processing_time = 0
        
        for post_request in request.posts:
            # Run moderation for each post
            result = await moderation_service.moderate_content(post_request)
            
            # Save to database
            post = Post(
                content=post_request.content,
                content_type=post_request.content_type.value,
                user_id=post_request.user_id,
                action=result.action.value,
                threat_level=result.threat_level.value,
                confidence=result.confidence,
                explanation=result.explanation,
                processing_time_ms=result.processing_time_ms,
                post_metadata=result.metadata
            )
            
            db.add(post)
            db.commit()
            db.refresh(post)
            
            # Update result with post ID
            result.post_id = post.id
            results.append(result)
            total_processing_time += result.processing_time_ms
        
        # Calculate summary
        summary = {}
        for result in results:
            action = result.action.value
            summary[action] = summary.get(action, 0) + 1
        
        logger.info(f"Batch moderation completed: {len(results)} posts processed")
        
        return BatchModerationResponse(
            results=results,
            total_posts=len(results),
            total_processing_time_ms=total_processing_time,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error in batch moderation endpoint: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Batch moderation failed: {str(e)}")


@router.get("/moderate/{post_id}", response_model=ModerationResult)
async def get_moderation_result(
    post_id: int,
    db: Session = Depends(get_db)
):
    """
    Get moderation result for a specific post by ID
    """
    try:
        post = db.query(Post).filter(Post.id == post_id).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Convert to ModerationResult format
        from app.models.schemas import ModerationAction, ThreatLevel
        
        result = ModerationResult(
            post_id=post.id,
            action=ModerationAction(post.action),
            threat_level=ThreatLevel(post.threat_level),
            confidence=post.confidence,
            explanation=post.explanation,
            processing_time_ms=post.processing_time_ms,
            metadata=getattr(post, 'post_metadata', None),  # Use getattr to avoid SQLAlchemy metadata conflict
            created_at=post.created_at
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting moderation result: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve moderation result")


@router.post("/moderate/{post_id}/feedback")
async def submit_feedback(
    post_id: int,
    feedback: dict,
    db: Session = Depends(get_db)
):
    """
    Submit feedback for a moderation decision
    
    This allows users to provide feedback on moderation decisions,
    which can be used to improve the system.
    """
    try:
        post = db.query(Post).filter(Post.id == post_id).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Store feedback in metadata
        if not post.post_metadata:
            post.post_metadata = {}
        
        post.post_metadata["feedback"] = {
            "user_feedback": feedback,
            "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
        }
        
        db.commit()
        
        logger.info(f"Feedback submitted for post {post_id}")
        
        return {"message": "Feedback submitted successfully", "post_id": post_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to submit feedback") 