"""
Posts API endpoints for managing moderated content
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc

from app.core.database import get_db
from app.models.schemas import PostResponse
from app.models.post import Post

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/posts", response_model=List[PostResponse])
async def get_posts(
    skip: int = Query(0, ge=0, description="Number of posts to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of posts to return"),
    action: Optional[str] = Query(None, description="Filter by moderation action"),
    threat_level: Optional[str] = Query(None, description="Filter by threat level"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    db: Session = Depends(get_db)
):
    """
    Get list of moderated posts with filtering and pagination
    
    Supports filtering by action, threat level, and user ID.
    Results can be sorted by various fields.
    """
    try:
        # Build query
        query = db.query(Post)
        
        # Apply filters
        if action:
            query = query.filter(Post.action == action)
        
        if threat_level:
            query = query.filter(Post.threat_level == threat_level)
        
        if user_id:
            query = query.filter(Post.user_id == user_id)
        
        # Apply sorting
        if hasattr(Post, sort_by):
            sort_field = getattr(Post, sort_by)
            if sort_order.lower() == "desc":
                query = query.order_by(desc(sort_field))
            else:
                query = query.order_by(asc(sort_field))
        else:
            # Default sorting by created_at desc
            query = query.order_by(desc(Post.created_at))
        
        # Apply pagination
        posts = query.offset(skip).limit(limit).all()
        
        # Convert to response format
        response_posts = []
        for post in posts:
            from app.models.schemas import ModerationAction, ThreatLevel, ContentType
            
            response_posts.append(PostResponse(
                id=post.id,
                content=post.content,
                content_type=ContentType(post.content_type),
                user_id=post.user_id,
                action=ModerationAction(post.action),
                threat_level=ThreatLevel(post.threat_level),
                confidence=post.confidence,
                explanation=post.explanation,
                processing_time_ms=post.processing_time_ms,
                metadata=getattr(post, 'post_metadata', None),
                created_at=post.created_at,
                updated_at=post.updated_at
            ))
        
        logger.info(f"Retrieved {len(response_posts)} posts (skip={skip}, limit={limit})")
        
        return response_posts
        
    except Exception as e:
        logger.error(f"Error retrieving posts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve posts")


@router.get("/posts/{post_id}", response_model=PostResponse)
async def get_post(
    post_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific post by ID
    """
    try:
        post = db.query(Post).filter(Post.id == post_id).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Convert to response format
        from app.models.schemas import ModerationAction, ThreatLevel, ContentType
        
        return PostResponse(
            id=post.id,
            content=post.content,
            content_type=ContentType(post.content_type),
            user_id=post.user_id,
            action=ModerationAction(post.action),
            threat_level=ThreatLevel(post.threat_level),
            confidence=post.confidence,
            explanation=post.explanation,
            processing_time_ms=post.processing_time_ms,
            metadata=getattr(post, 'post_metadata', None),
            created_at=post.created_at,
            updated_at=post.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve post")


@router.delete("/posts/{post_id}")
async def delete_post(
    post_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a post (soft delete by updating metadata)
    """
    try:
        post = db.query(Post).filter(Post.id == post_id).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Soft delete by updating metadata
        if not post.post_metadata:
            post.post_metadata = {}
        
        post.post_metadata["deleted"] = True
        post.post_metadata["deleted_at"] = "2024-01-01T00:00:00Z"  # Would use actual timestamp
        
        db.commit()
        
        logger.info(f"Post {post_id} marked as deleted")
        
        return {"message": "Post deleted successfully", "post_id": post_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting post {post_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete post")


@router.get("/posts/search")
async def search_posts(
    q: str = Query(..., min_length=1, description="Search query"),
    skip: int = Query(0, ge=0, description="Number of posts to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of posts to return"),
    db: Session = Depends(get_db)
):
    """
    Search posts by content
    
    Performs a simple text search in post content.
    """
    try:
        # Simple text search (could be enhanced with full-text search)
        query = db.query(Post).filter(Post.content.ilike(f"%{q}%"))
        
        # Apply pagination
        posts = query.offset(skip).limit(limit).all()
        
        # Convert to response format
        response_posts = []
        for post in posts:
            from app.models.schemas import ModerationAction, ThreatLevel, ContentType
            
            response_posts.append(PostResponse(
                id=post.id,
                content=post.content,
                content_type=ContentType(post.content_type),
                user_id=post.user_id,
                action=ModerationAction(post.action),
                threat_level=ThreatLevel(post.threat_level),
                confidence=post.confidence,
                explanation=post.explanation,
                processing_time_ms=post.processing_time_ms,
                metadata=getattr(post, 'post_metadata', None),
                created_at=post.created_at,
                updated_at=post.updated_at
            ))
        
        logger.info(f"Search for '{q}' returned {len(response_posts)} posts")
        
        return {
            "query": q,
            "results": response_posts,
            "total": len(response_posts),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error searching posts: {e}")
        raise HTTPException(status_code=500, detail="Failed to search posts")


@router.get("/posts/stats/summary")
async def get_posts_summary(db: Session = Depends(get_db)):
    """
    Get summary statistics for posts
    """
    try:
        # Total posts
        total_posts = db.query(Post).count()
        
        # Posts by action
        posts_by_action = {}
        for action in ["accept", "flag", "block", "quarantine"]:
            count = db.query(Post).filter(Post.action == action).count()
            posts_by_action[action] = count
        
        # Posts by threat level
        posts_by_threat_level = {}
        for level in ["low", "medium", "high", "critical"]:
            count = db.query(Post).filter(Post.threat_level == level).count()
            posts_by_threat_level[level] = count
        
        # Average processing time
        avg_time = db.query(Post.processing_time_ms).filter(
            Post.processing_time_ms > 0
        ).scalar() or 0
        
        # Recent activity (last 24 hours)
        from datetime import datetime, timedelta
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_posts = db.query(Post).filter(Post.created_at >= yesterday).count()
        
        return {
            "total_posts": total_posts,
            "posts_by_action": posts_by_action,
            "posts_by_threat_level": posts_by_threat_level,
            "average_processing_time_ms": float(avg_time),
            "recent_posts_24h": recent_posts
        }
        
    except Exception as e:
        logger.error(f"Error getting posts summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get posts summary") 