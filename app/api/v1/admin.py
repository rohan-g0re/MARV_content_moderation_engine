"""
Admin API endpoints for system administration
"""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.schemas import RuleCreateRequest, RuleUpdateRequest, RuleResponse
from app.models.post import ModerationRule
from app.services.rule_service import RuleService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/stats")
async def get_admin_stats(db: Session = Depends(get_db)):
    """
    Get comprehensive system statistics for admin dashboard
    """
    try:
        from app.models.post import Post
        
        # Post statistics
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
        
        # Rule statistics
        total_rules = db.query(ModerationRule).count()
        active_rules = db.query(ModerationRule).filter(ModerationRule.is_active == True).count()
        
        # Rules by type
        rules_by_type = {}
        rule_types = db.query(ModerationRule.rule_type).distinct().all()
        for rule_type in rule_types:
            count = db.query(ModerationRule).filter(
                ModerationRule.rule_type == rule_type[0],
                ModerationRule.is_active == True
            ).count()
            rules_by_type[rule_type[0]] = count
        
        # Performance metrics
        avg_processing_time = db.query(Post.processing_time_ms).filter(
            Post.processing_time_ms > 0
        ).scalar() or 0
        
        return {
            "posts": {
                "total": total_posts,
                "by_action": posts_by_action,
                "by_threat_level": posts_by_threat_level,
                "average_processing_time_ms": float(avg_processing_time)
            },
            "rules": {
                "total": total_rules,
                "active": active_rules,
                "by_type": rules_by_type
            },
            "system": {
                "status": "healthy",
                "version": "1.0.0"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get admin statistics")


@router.get("/rules", response_model=List[RuleResponse])
async def get_rules(
    rule_type: str = None,
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """
    Get moderation rules with optional filtering
    """
    try:
        query = db.query(ModerationRule)
        
        if rule_type:
            query = query.filter(ModerationRule.rule_type == rule_type)
        
        if active_only:
            query = query.filter(ModerationRule.is_active == True)
        
        rules = query.all()
        
        response_rules = []
        for rule in rules:
            response_rules.append(RuleResponse(
                id=rule.id,
                rule_type=rule.rule_type,
                pattern=rule.pattern,
                severity=rule.severity,
                is_regex=rule.is_regex,
                is_active=rule.is_active,
                description=rule.description,
                category=rule.category,
                created_at=rule.created_at,
                updated_at=rule.updated_at
            ))
        
        return response_rules
        
    except Exception as e:
        logger.error(f"Error getting rules: {e}")
        raise HTTPException(status_code=500, detail="Failed to get rules")


@router.post("/rules", response_model=RuleResponse)
async def create_rule(
    request: RuleCreateRequest,
    db: Session = Depends(get_db)
):
    """
    Create a new moderation rule
    """
    try:
        # Check if rule ID already exists
        existing_rule = db.query(ModerationRule).filter(ModerationRule.id == request.rule_id).first()
        if existing_rule:
            raise HTTPException(status_code=400, detail="Rule ID already exists")
        
        # Create new rule
        rule = ModerationRule(
            id=request.rule_id,
            rule_type=request.rule_type,
            pattern=request.pattern,
            severity=request.severity,
            is_regex=request.is_regex,
            description=request.description,
            category=request.category
        )
        
        db.add(rule)
        db.commit()
        db.refresh(rule)
        
        logger.info(f"Created rule: {request.rule_id}")
        
        return RuleResponse(
            id=rule.id,
            rule_type=rule.rule_type,
            pattern=rule.pattern,
            severity=rule.severity,
            is_regex=rule.is_regex,
            is_active=rule.is_active,
            description=rule.description,
            category=rule.category,
            created_at=rule.created_at,
            updated_at=rule.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating rule: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create rule")


@router.put("/rules/{rule_id}", response_model=RuleResponse)
async def update_rule(
    rule_id: str,
    request: RuleUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    Update an existing moderation rule
    """
    try:
        rule = db.query(ModerationRule).filter(ModerationRule.id == rule_id).first()
        
        if not rule:
            raise HTTPException(status_code=404, detail="Rule not found")
        
        # Update fields
        if request.pattern is not None:
            rule.pattern = request.pattern
        if request.severity is not None:
            rule.severity = request.severity
        if request.is_regex is not None:
            rule.is_regex = request.is_regex
        if request.is_active is not None:
            rule.is_active = request.is_active
        if request.description is not None:
            rule.description = request.description
        if request.category is not None:
            rule.category = request.category
        
        db.commit()
        db.refresh(rule)
        
        logger.info(f"Updated rule: {rule_id}")
        
        return RuleResponse(
            id=rule.id,
            rule_type=rule.rule_type,
            pattern=rule.pattern,
            severity=rule.severity,
            is_regex=rule.is_regex,
            is_active=rule.is_active,
            description=rule.description,
            category=rule.category,
            created_at=rule.created_at,
            updated_at=rule.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating rule: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update rule")


@router.delete("/rules/{rule_id}")
async def delete_rule(
    rule_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a moderation rule (soft delete)
    """
    try:
        rule = db.query(ModerationRule).filter(ModerationRule.id == rule_id).first()
        
        if not rule:
            raise HTTPException(status_code=404, detail="Rule not found")
        
        # Soft delete
        rule.is_active = False
        db.commit()
        
        logger.info(f"Deleted rule: {rule_id}")
        
        return {"message": "Rule deleted successfully", "rule_id": rule_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting rule: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete rule")


@router.post("/rules/import")
async def import_rules(
    rules: List[RuleCreateRequest],
    db: Session = Depends(get_db)
):
    """
    Import multiple rules at once
    """
    try:
        imported_count = 0
        skipped_count = 0
        
        for rule_request in rules:
            # Check if rule already exists
            existing_rule = db.query(ModerationRule).filter(ModerationRule.id == rule_request.rule_id).first()
            if existing_rule:
                skipped_count += 1
                continue
            
            # Create new rule
            rule = ModerationRule(
                id=rule_request.rule_id,
                rule_type=rule_request.rule_type,
                pattern=rule_request.pattern,
                severity=rule_request.severity,
                is_regex=rule_request.is_regex,
                description=rule_request.description,
                category=rule_request.category
            )
            
            db.add(rule)
            imported_count += 1
        
        db.commit()
        
        logger.info(f"Imported {imported_count} rules, skipped {skipped_count}")
        
        return {
            "message": "Rules import completed",
            "imported": imported_count,
            "skipped": skipped_count
        }
        
    except Exception as e:
        logger.error(f"Error importing rules: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to import rules")


@router.get("/logs")
async def get_system_logs(
    limit: int = 100,
    level: str = None
):
    """
    Get system logs (placeholder - would integrate with proper logging system)
    """
    try:
        # This is a placeholder - in a real system, you'd integrate with
        # a proper logging system like ELK stack or similar
        return {
            "message": "System logs endpoint - would integrate with logging system",
            "limit": limit,
            "level": level,
            "logs": []  # Would contain actual log entries
        }
        
    except Exception as e:
        logger.error(f"Error getting system logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system logs") 