from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
import os
import logging
import socket
import sys

# Import the consolidated moderation engine
from app.core.moderation import GuardianModerationEngine, ModerationResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_port_available(host, port):
    """Check if a port is available for binding"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError as e:
        logger.error(f"Port {port} is not available: {e}")
        return False

def find_available_port(host, start_port, max_port=9000):
    """Find an available port starting from start_port"""
    for port in range(start_port, max_port + 1):
        if is_port_available(host, port):
            return port
    return None

# Initialize FastAPI
app = FastAPI(
    title="GuardianAI Content Moderation API", 
    version="2.0.0",
    description="Consolidated three-stage content moderation pipeline"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = "sqlite:///./moderation.db"
logger.info(f"Using database at: {os.path.abspath('moderation.db')}")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model
class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    accepted = Column(Boolean, nullable=False)
    reason = Column(Text, nullable=False)
    threat_level = Column(String, default="low")
    confidence = Column(String, default="1.0")
    stage = Column(String, default="unknown")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class ModerationRequest(BaseModel):
    content: str

class ModerationResponse(BaseModel):
    accepted: bool
    reason: str
    id: int
    threat_level: str = "low"
    confidence: float = 1.0
    stage: str = "unknown"
    action: str = "unknown"
    explanation: str = ""

# Initialize the consolidated moderation engine
logger.info("Initializing GuardianAI Moderation Engine...")
moderation_engine = GuardianModerationEngine()

# API endpoints
@app.get("/")
def root():
    return {
        "message": "GuardianAI Content Moderation API", 
        "status": "running",
        "version": "2.0.0 - Consolidated Pipeline",
        "models": moderation_engine.get_model_status()
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models": moderation_engine.get_model_status()
    }

@app.post("/moderate", response_model=ModerationResponse)
def moderate_post(request: ModerationRequest):
    """
    Main moderation endpoint using consolidated GuardianAI pipeline
    
    Processes content through three stages:
    1. Rule-based filtering
    2. Detoxify AI toxicity detection
    3. FinBERT financial fraud detection
    """
    try:
        # Run moderation pipeline using consolidated engine
        result = moderation_engine.moderate_content(request.content)
        
        # Save to database with enhanced information
        db = SessionLocal()
        post = Post(
            content=request.content,
            accepted=result.accepted,
            reason=result.reason,
            threat_level=result.threat_level,
            confidence=str(result.confidence),
            stage=result.stage
        )
        db.add(post)
        db.commit()
        db.refresh(post)
        logger.info(f"Saved post ID {post.id} to DB - {result.action.upper()}: {result.reason}")
        db.close()
        
        return ModerationResponse(
            accepted=result.accepted,
            reason=result.reason,
            id=post.id,
            threat_level=result.threat_level,
            confidence=result.confidence,
            stage=result.stage,
            action=result.action,
            explanation=result.explanation
        )
        
    except Exception as e:
        logger.error(f"Moderation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/posts")
def get_posts():
    """Get all moderated posts with enhanced information"""
    try:
        db = SessionLocal()
        posts = db.query(Post).order_by(Post.created_at.desc()).all()
        db.close()
        
        return [
            {
                "id": post.id,
                "content": post.content,
                "accepted": post.accepted,
                "reason": post.reason,
                "threat_level": post.threat_level,
                "confidence": float(post.confidence) if post.confidence else 1.0,
                "stage": post.stage,
                "created_at": post.created_at.isoformat()
            }
            for post in posts
        ]
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    """Get moderation statistics"""
    try:
        db = SessionLocal()
        
        total_posts = db.query(Post).count()
        accepted_posts = db.query(Post).filter(Post.accepted == True).count()
        rejected_posts = total_posts - accepted_posts
        
        # Get rejection reasons breakdown
        from sqlalchemy import func
        reason_stats = db.query(
            Post.stage,
            func.count(Post.id).label('count')
        ).filter(Post.accepted == False).group_by(Post.stage).all()
        
        # Get threat level breakdown
        threat_stats = db.query(
            Post.threat_level,
            func.count(Post.id).label('count')
        ).group_by(Post.threat_level).all()
        
        db.close()
        
        return {
            "total_posts": total_posts,
            "accepted": accepted_posts,
            "rejected": rejected_posts,
            "acceptance_rate": round((accepted_posts / total_posts * 100), 2) if total_posts > 0 else 0,
            "rejection_by_stage": {stat.stage: stat.count for stat in reason_stats},
            "threat_levels": {stat.threat_level: stat.count for stat in threat_stats},
            "models": moderation_engine.get_model_status()
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/reload-keywords")
def reload_keywords():
    """Reload keywords from file - for Day 10 dictionary expansion"""
    try:
        moderation_engine.reload_keywords()
        return {"message": "Keywords reloaded successfully", "count": len(moderation_engine.keywords)}
    except Exception as e:
        logger.error(f"Keyword reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/update-thresholds")
def update_thresholds(
    toxicity_threshold: float = None,
    finbert_threshold: float = None,
    llm_escalation_threshold: int = None
):
    """Update moderation thresholds - for Day 6 LLM escalation logic"""
    try:
        moderation_engine.update_thresholds(
            toxicity_threshold=toxicity_threshold,
            finbert_threshold=finbert_threshold,
            llm_escalation_threshold=llm_escalation_threshold
        )
        return {
            "message": "Thresholds updated successfully",
            "thresholds": moderation_engine.get_model_status()["thresholds"]
        }
    except Exception as e:
        logger.error(f"Threshold update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Server configuration
    host = "0.0.0.0"
    preferred_port = 8000
    
    logger.info("Starting GuardianAI Content Moderation API...")
    
    # Check if preferred port is available
    if is_port_available(host, preferred_port):
        port = preferred_port
        logger.info(f"✅ Port {port} is available, starting server...")
    else:
        # Find an alternative port
        port = find_available_port(host, preferred_port + 1)
        if port:
            logger.warning(f"⚠️  Port {preferred_port} is busy, using alternative port {port}")
        else:
            logger.error(f"❌ No available ports found between {preferred_port+1} and 9000")
            sys.exit(1)
    
    try:
        logger.info(f"🚀 GuardianAI API will be available at: http://localhost:{port}")
        logger.info(f"📊 API Documentation: http://localhost:{port}/docs")
        uvicorn.run(app, host=host, port=port, reload=False)
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1) 