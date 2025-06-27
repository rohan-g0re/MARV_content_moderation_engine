"""
Main FastAPI application for MARV Content Moderation Engine
"""

import logging
import time
import json
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.config import settings, get_allowed_origins
from app.core.database import get_db, create_tables
from app.models.schemas import (
    ModerationRequest, ModerationResult, PostResponse, 
    BatchModerationRequest, BatchModerationResponse,
    HealthResponse, SystemStats, ErrorResponse
)
from app.services.moderation_service import ModerationService
from app.api.v1 import moderation, posts, admin

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting MARV Content Moderation Engine...")
    
    # Create database tables
    try:
        create_tables()
        logger.info("✅ Database tables created")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
    
    # Initialize services
    app.state.moderation_service = ModerationService()
    logger.info("✅ Services initialized")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down MARV Content Moderation Engine...")


# Create FastAPI app
app = FastAPI(
    title="MARV Content Moderation Engine",
    description="Production-ready content moderation system with multi-layer filtering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    
    error_response = ErrorResponse(
        error="Internal server error",
        detail=str(exc) if settings.DEBUG else "An unexpected error occurred"
    )
    
    return JSONResponse(
        status_code=500,
        content=json.loads(error_response.json())
    )


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database
        db = next(get_db())
        db.execute(text("SELECT 1"))
        db.close()
        database_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        database_status = "unhealthy"
    
    # Check ML models
    moderation_service = app.state.moderation_service
    ml_status = await moderation_service.get_system_stats()
    
    return HealthResponse(
        status="healthy" if database_status == "healthy" else "degraded",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        database_status=database_status,
        ml_models_status={
            "rule_service": "loaded",
            "ml_service": "loaded" if ml_status["ml_models_loaded"] else "not loaded",
            "llm_service": "enabled" if ml_status["llm_enabled"] else "disabled"
        }
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MARV Content Moderation Engine",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# Include API routes
app.include_router(moderation.router, prefix="/api/v1", tags=["moderation"])
app.include_router(posts.router, prefix="/api/v1", tags=["posts"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])


# System statistics endpoint
@app.get("/api/v1/stats", response_model=SystemStats)
async def get_system_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    try:
        from app.models.post import Post, ModerationRule
        
        # Get post statistics
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
        
        # Rule statistics
        total_rules = db.query(ModerationRule).count()
        active_rules = db.query(ModerationRule).filter(ModerationRule.is_active == True).count()
        
        # System uptime (placeholder - would need to track startup time)
        system_uptime = int(time.time())  # Placeholder
        
        return SystemStats(
            total_posts=total_posts,
            posts_by_action=posts_by_action,
            posts_by_threat_level=posts_by_threat_level,
            average_processing_time_ms=float(avg_time),
            total_rules=total_rules,
            active_rules=active_rules,
            system_uptime_seconds=system_uptime
        )
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system statistics")


# Test endpoint
@app.post("/api/v1/test")
async def test_moderation():
    """Test endpoint for moderation pipeline"""
    try:
        test_request = ModerationRequest(
            content="This is a test message to verify the moderation system is working correctly.",
            content_type="text"
        )
        
        moderation_service = app.state.moderation_service
        result = await moderation_service.moderate_content(test_request)
        
        return {
            "status": "success",
            "test_content": test_request.content,
            "result": result.dict()
        }
        
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    ) 