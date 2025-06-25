"""
FastAPI Backend for GuardianAI Content Moderation Engine

This module provides the main FastAPI application with endpoints for:
- Content moderation
- System statistics
- Configuration management
- Health checks

This implements the backend requirements as specified in the project brief.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import uvicorn
from pathlib import Path

from .core.guardian_ai import GuardianAI, ModerationResult
from .core.config import ModerationConfig, get_default_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/backend.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GuardianAI Content Moderation Engine",
    description="A cost-efficient, scalable content moderation system using multiple filtering layers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global GuardianAI instance
guardian_ai: Optional[GuardianAI] = None

# Pydantic models for API requests/responses
class ModerationRequest(BaseModel):
    content: str = Field(..., description="Text content to moderate", min_length=1, max_length=10000)
    content_id: Optional[str] = Field(None, description="Optional unique identifier for the content")

class ModerationResponse(BaseModel):
    content_id: str
    threat_level: str
    action: str
    confidence: float
    explanation: str
    layer_results: Dict[str, Any]
    processing_time_ms: float
    timestamp: str

class StatisticsResponse(BaseModel):
    total_processed: int
    layer_usage: Dict[str, int]
    action_distribution: Dict[str, int]
    threat_distribution: Dict[str, int]
    action_percentages: Dict[str, float]
    threat_percentages: Dict[str, float]
    layer_percentages: Dict[str, float]

class ConfigUpdateRequest(BaseModel):
    thresholds: Optional[Dict[str, float]] = None
    layers: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    guardian_ai_status: str

# Dependency to get GuardianAI instance
def get_guardian_ai() -> GuardianAI:
    if guardian_ai is None:
        raise HTTPException(status_code=503, detail="GuardianAI not initialized")
    return guardian_ai

@app.on_event("startup")
async def startup_event():
    """Initialize GuardianAI on startup"""
    global guardian_ai
    
    try:
        # Load configuration
        config = get_default_config()
        
        # Initialize GuardianAI
        guardian_ai = GuardianAI(config, logger)
        
        logger.info("GuardianAI Content Moderation Engine started successfully")
        logger.info(f"Configuration: {config.to_dict()}")
        
    except Exception as e:
        logger.error(f"Failed to initialize GuardianAI: {e}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "GuardianAI Content Moderation Engine",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.post("/moderate", response_model=ModerationResponse)
async def moderate_content(
    request: ModerationRequest,
    guardian: GuardianAI = Depends(get_guardian_ai)
):
    """
    Moderate content using the GuardianAI system
    
    This endpoint implements the main moderation functionality as specified in the project brief.
    It uses the unified moderation controller to ensure explainability and tunable thresholds.
    """
    try:
        # Perform moderation
        result = guardian.moderate_content(request.content, request.content_id)
        
        # Convert to response format
        response = ModerationResponse(
            content_id=result.content_id,
            threat_level=result.threat_level.name,
            action=result.action.value,
            confidence=result.confidence,
            explanation=result.explanation,
            layer_results=result.layer_results,
            processing_time_ms=result.processing_time_ms,
            timestamp=result.timestamp.isoformat()
        )
        
        logger.info(f"Moderation completed for {result.content_id}: {result.threat_level.name} -> {result.action.value}")
        
        return response
        
    except Exception as e:
        logger.error(f"Moderation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Moderation failed: {str(e)}")

@app.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(guardian: GuardianAI = Depends(get_guardian_ai)):
    """
    Get system statistics and performance metrics
    
    This provides insights into the moderation system's performance and usage patterns.
    """
    try:
        stats = guardian.get_statistics()
        return StatisticsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.get("/config", response_model=Dict[str, Any])
async def get_config(guardian: GuardianAI = Depends(get_guardian_ai)):
    """
    Get current system configuration
    
    This returns the current configuration including thresholds and layer settings.
    """
    try:
        return guardian.config.to_dict()
        
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")

@app.put("/config")
async def update_config(
    request: ConfigUpdateRequest,
    guardian: GuardianAI = Depends(get_guardian_ai)
):
    """
    Update system configuration dynamically
    
    This allows runtime configuration updates for thresholds and layer settings.
    """
    try:
        config_dict = {}
        
        if request.thresholds:
            config_dict["thresholds"] = request.thresholds
        
        if request.layers:
            config_dict["layers"] = request.layers
        
        if config_dict:
            success = guardian.config.update_from_dict(config_dict)
            if success:
                logger.info(f"Configuration updated: {config_dict}")
                return {"message": "Configuration updated successfully", "changes": config_dict}
            else:
                raise HTTPException(status_code=400, detail="Invalid configuration values")
        else:
            raise HTTPException(status_code=400, detail="No configuration changes provided")
        
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    This provides system health information for monitoring and deployment.
    """
    import time
    
    try:
        guardian_status = "healthy" if guardian_ai is not None else "unhealthy"
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime=time.time(),  # Simplified - should track actual uptime
            guardian_ai_status=guardian_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="System unhealthy")

@app.post("/test")
async def test_moderation():
    """
    Test endpoint with sample content
    
    This provides a quick way to test the moderation system with predefined content.
    """
    test_cases = [
        {
            "content": "This is a safe and normal post about finance.",
            "expected": "SAFE"
        },
        {
            "content": "I hate this stupid system and want to kill it.",
            "expected": "HIGH"
        },
        {
            "content": "Just sharing some investment advice - buy low, sell high!",
            "expected": "SAFE"
        }
    ]
    
    results = []
    guardian = get_guardian_ai()
    
    for i, test_case in enumerate(test_cases):
        try:
            result = guardian.moderate_content(test_case["content"], f"test_{i}")
            results.append({
                "test_id": i,
                "content": test_case["content"],
                "expected": test_case["expected"],
                "actual": result.threat_level.name,
                "action": result.action.value,
                "confidence": result.confidence,
                "explanation": result.explanation
            })
        except Exception as e:
            results.append({
                "test_id": i,
                "content": test_case["content"],
                "error": str(e)
            })
    
    return {
        "message": "Test completed",
        "results": results
    }

if __name__ == "__main__":
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 