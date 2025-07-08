import os
import datetime
import logging
import socket
import sys
import pathlib
import time

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text, func, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import OperationalError, SQLAlchemyError
import requests
import psycopg2
from psycopg2 import sql
from datetime import datetime

from dotenv import load_dotenv
from logger import get_logger

# Load .env
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
dotenv_path = BASE_DIR / ".env"
print(f"Loading .env from: {dotenv_path}")
load_dotenv(dotenv_path)

# Initialize custom logger
logger = get_logger("main", "moderation.log")

# ---- Import your moderation engine! ----
from app.core.moderation import GuardianModerationEngine

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:123456@localhost:5432/content_moderation")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "content_moderation")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123456")

def test_database_connection():
    """Test if we can connect to PostgreSQL database"""
    try:
        logger.info(f"Testing connection to PostgreSQL database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.close()
        logger.info("✅ Successfully connected to PostgreSQL database")
        return True
    except psycopg2.OperationalError as e:
        logger.error(f"❌ Failed to connect to PostgreSQL database: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error connecting to database: {e}")
        return False

def ensure_database_exists():
    """Ensure the content_moderation database exists"""
    try:
        # Connect to default postgres database to create our database if needed
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database="postgres",  # Connect to default database
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        exists = cursor.fetchone()
        
        if not exists:
            logger.info(f"Creating database: {DB_NAME}")
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
            logger.info(f"✅ Database {DB_NAME} created successfully")
        else:
            logger.info(f"✅ Database {DB_NAME} already exists")
            
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error ensuring database exists: {e}")
        return False

def check_table_exists(table_name="posts"):
    """Check if a specific table exists in the database"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
        """, (table_name,))
        
        exists = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        return exists
        
    except Exception as e:
        logger.error(f"❌ Error checking if table '{table_name}' exists: {e}")
        return False

def validate_table_structure(table_name="posts"):
    """Validate that the table has the expected structure"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        
        # Get table columns
        cursor.execute("""
            SELECT column_name, data_type, is_nullable, column_default 
            FROM information_schema.columns 
            WHERE table_name = %s 
            ORDER BY ordinal_position;
        """, (table_name,))
        
        columns = cursor.fetchall()
        actual_columns = [col[0] for col in columns]
        
        # Expected columns for posts table
        expected_columns = [
            'id', 'content', 'accepted', 'reason', 'threat_level',
            'confidence', 'stage', 'band', 'action', 'created_at',
            'processing_time', 'override', 'llm_explanation', 'llm_troublesome_words', 'llm_suggestion'
        ]
        
        missing_columns = [col for col in expected_columns if col not in actual_columns]
        extra_columns = [col for col in actual_columns if col not in expected_columns]
        
        if missing_columns:
            logger.warning(f"⚠️ Missing columns in '{table_name}': {missing_columns}")
            return False
            
        if extra_columns:
            logger.info(f"ℹ️ Extra columns in '{table_name}': {extra_columns}")
        
        logger.info(f"✅ Table '{table_name}' structure is valid")
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating table '{table_name}' structure: {e}")
        return False

def ensure_tables_exist():
    """Ensure required tables exist with proper structure"""
    try:
        table_exists = check_table_exists("posts")
        
        if table_exists:
            logger.info("✅ Posts table already exists")
            
            # Validate structure
            if validate_table_structure("posts"):
                logger.info("✅ Posts table structure is valid")
                return True
            else:
                logger.warning("⚠️ Posts table structure validation failed")
                # You could add logic here to migrate or recreate table if needed
                return False
        else:
            logger.info("🔧 Posts table does not exist, creating...")
            # Create tables using SQLAlchemy
            Base.metadata.create_all(bind=engine)
            logger.info("✅ Posts table created successfully")
            
            # Validate the newly created table
            if validate_table_structure("posts"):
                logger.info("✅ Newly created table structure validated")
                return True
            else:
                logger.error("❌ Newly created table structure validation failed")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error ensuring tables exist: {e}")
        return False

def create_engine_with_retry(max_retries=3, retry_delay=2):
    """Create SQLAlchemy engine with retry logic"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Creating SQLAlchemy engine (attempt {attempt + 1}/{max_retries})")
            engine = create_engine(
                DATABASE_URL,
                pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
                max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
                pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
                pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
                echo=False  # Set to True for SQL query logging
            )
            
            # Test the connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("✅ SQLAlchemy engine created successfully")
            return engine
            
        except Exception as e:
            logger.error(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("❌ All attempts to create engine failed")
                raise

# Initialize database connection
logger.info("🔗 Initializing database connection...")

# Ensure database exists first
if not ensure_database_exists():
    logger.error("❌ Failed to ensure database exists. Exiting...")
    sys.exit(1)

# Test connection
if not test_database_connection():
    logger.error("❌ Database connection test failed. Exiting...")
    sys.exit(1)

# Create engine with retry
engine = create_engine_with_retry()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    accepted = Column(Boolean, nullable=False)
    reason = Column(Text, nullable=False)
    threat_level = Column(String, default="low")
    confidence = Column(String, default="1.0")
    stage = Column(String, default="unknown")
    band = Column(String, default="SAFE")
    action = Column(String, default="PASS")
    created_at = Column(DateTime, default=func.now())
    processing_time = Column(String, default="0.0")
    override = Column(String, default="No")
    llm_explanation = Column(Text, default="")
    llm_troublesome_words = Column(Text, default="")
    llm_suggestion = Column(Text, default="")
    comments = Column(String(500), nullable=True)

# Ensure tables exist with proper validation
try:
    logger.info("🔧 Ensuring database tables exist with validation...")
    if not ensure_tables_exist():
        logger.error("❌ Failed to ensure tables exist properly")
        sys.exit(1)
    logger.info("✅ Database tables ready and validated")
except Exception as e:
    logger.error(f"❌ Error ensuring tables exist: {e}")
    sys.exit(1)

app = FastAPI(
    title="GuardianAI Content Moderation API",
    version="2.0.0",
    description="Multilayered moderation pipeline with LLM escalation"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModerationRequest(BaseModel):
    content: str
    comments: str = None

class ModerationResponse(BaseModel):
    accepted: bool
    reason: str
    id: int
    threat_level: str = "low"
    confidence: float = 1.0
    stage: str = "unknown"
    band: str = "SAFE"
    action: str = "PASS"
    explanation: str = ""
    troublesome_words: list = []
    suggestion: str = ""
    processing_time: float = 0.0
    override: str = "No"
    comments: str = None

class OverrideRequest(BaseModel):
    post_id: int
    override_value: str  # 'Accepted', 'Rejected', 'Flagged'

class SaveCommentsRequest(BaseModel):
    post_id: int
    comments: str

# ---- Real moderation engine ----
moderation_engine = GuardianModerationEngine()

# ---- LLM Utility ----
def get_llm_explanation_and_suggestion(post_text):
    prompt = f"""
A user's social media post was flagged as inappropriate.

Post:
{post_text}

Instructions:

1. Briefly explain, in plain language (1-2 sentences), why the post may be considered inappropriate. Do NOT mention algorithms, models, scores, or moderation systems.
2. Clearly list any exact word(s) or phrase(s) in the post that could be problematic (as a list).
3. Provide a short, positive alternative way for the user to express the same idea without the problematic words or phrases.

Respond only in JSON as follows:
{{
  "explanation": "Very short, user-facing explanation here.",
  "troublesome_words": ["list", "of", "problem", "words"],
  "suggestion": "Short, friendly rewording here."
}}
"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.3,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        try:
            resp_json = response.json()
        except Exception as e:
            logging.error(f"LLM did not return JSON: {e}. Raw response: {response.text}")
            return "LLM did not return JSON.", [], ""
        
        if "choices" in resp_json and resp_json["choices"]:
            content = resp_json["choices"][0]["message"]["content"]
            try:
                import json as _json
                # Find the start and end of the JSON object to handle preambles
                json_start = content.find('{')
                json_end = content.rfind('}')
                if json_start != -1 and json_end != -1:
                    json_str = content[json_start:json_end+1]
                    parsed = _json.loads(json_str)
                    return (
                        parsed.get("explanation", ""),
                        parsed.get("troublesome_words", []),
                        parsed.get("suggestion", "")
                    )
                else:
                    raise ValueError("No JSON object found in LLM response")
            except Exception as ex:
                return content, [], ""
        else:
            if "error" in resp_json:
                error_message = resp_json["error"].get("message", str(resp_json["error"]))
                logging.error(f"LLM API error: {error_message}")
                return error_message, [], ""
            logging.error(f"LLM response missing 'choices'. Full response: {resp_json}")
            return "LLM response missing 'choices'.", [], ""
    except Exception as e:
        logging.error(f"Exception during LLM request: {e}")
        return "LLM request failed.", [], ""

def is_port_available(host, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError as e:
        logger.error(f"Port {port} is not available: {e}")
        return False

def find_available_port(host, start_port, max_port=9000):
    for port in range(start_port, max_port + 1):
        if is_port_available(host, port):
            return port
    return None

@app.get("/")
def root():
    return {
        "message": "GuardianAI Content Moderation API",
        "status": "running",
        "version": "2.0.0",
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
    logger.info(
        "Incoming moderation request",
        extra={"extra": {
            "content": request.content[:100],  # log a snippet, not full text
            "comments": request.comments,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            # Add user_id, session_id, ip if available
        }},
    )
    start_time = time.time()
    try:
        result = moderation_engine.moderate_content(request.content)
        llm_explanation, llm_troublesome_words, llm_suggestion = "", [], ""

        # Call LLM only if blocked or flagged by *any* strict action (FLAG, BLOCK, etc)
        if result.action.upper() in ("BLOCK", "FLAG", "FLAG_LOW", "FLAG_MEDIUM", "FLAG_HIGH"):
            llm_explanation, llm_troublesome_words, llm_suggestion = get_llm_explanation_and_suggestion(request.content)

        # Calculate processing time
        processing_time = time.time() - start_time
        
        import json as _json
        db = SessionLocal()
        post = Post(
            content=request.content,
            accepted=result.accepted,
            reason=result.reason,
            threat_level=result.threat_level,
            confidence=str(result.confidence),
            stage=result.stage,
            band=result.band,
            action=result.action,
            processing_time=str(processing_time),
            llm_explanation=llm_explanation,
            llm_troublesome_words=_json.dumps(llm_troublesome_words),
            llm_suggestion=llm_suggestion,
            override="No",
            comments=request.comments
        )
        db.add(post)
        db.commit()
        db.refresh(post)
        db.close()
        logger.info(
            "Moderation decision",
            extra={"extra": {
                "content_id": post.id,
                "decision": "ACCEPTED" if result.accepted else "REJECTED",
                "processing_time": processing_time,
            }},
        )
        return ModerationResponse(
            accepted=result.accepted,
            reason=result.reason,
            id=post.id,
            threat_level=result.threat_level,
            confidence=float(result.confidence),
            stage=result.stage,
            band=result.band,
            action=result.action,
            explanation=llm_explanation,
            troublesome_words=llm_troublesome_words,
            suggestion=llm_suggestion,
            processing_time=processing_time,
            override="No",
            comments=request.comments
        )
    except Exception as e:
        logger.error(
            "Error during moderation",
            extra={"extra": {
                "error": str(e),
                "content_id": request.content[:100],  # log a snippet, not full text
            }},
        )
        # Check if this is an external API error that we can handle gracefully
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['404', 'not found', 'api', 'external', 'model']):
            # Return a graceful error response instead of 500
            return ModerationResponse(
                accepted=True,  # Default to accept when external models fail
                reason="External models temporarily unavailable, using fallback logic",
                id=0,
                threat_level="low",
                confidence=0.5,
                stage="fallback",
                band="SAFE",
                action="PASS",
                explanation="Some external moderation models are currently unavailable. Content was processed using available models only.",
                troublesome_words=[],
                suggestion="",
                processing_time=time.time() - start_time,
                override="No",
                comments=request.comments
            )
        else:
            # For other errors, still raise 500
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/posts")
def get_posts():
    try:
        import json as _json
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
                "confidence": float(post.confidence) if post.confidence is not None else 1.0,
                "stage": post.stage,
                "band": post.band if post.band is not None else 'SAFE',
                "action": post.action if post.action is not None else 'PASS',
                "processing_time": float(post.processing_time) if post.processing_time is not None else 0.0,
                "override": post.override if post.override is not None else 'No',
                "llm_explanation": post.llm_explanation,
                "llm_troublesome_words": _json.loads(post.llm_troublesome_words if post.llm_troublesome_words else "[]"),
                "llm_suggestion": post.llm_suggestion,
                "created_at": post.created_at.isoformat() if post.created_at else None
            }
            for post in posts
        ]
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/override")
def override_post_status(request: OverrideRequest):
    try:
        db = SessionLocal()
        post = db.query(Post).filter(Post.id == request.post_id).first()
        if not post:
            db.close()
            raise HTTPException(status_code=404, detail="Post not found")
        # Get original moderation result
        original_result = moderation_engine.moderate_content(post.content)
        # Map logic output to label
        if original_result.accepted:
            original_label = 'Accepted'
        elif original_result.action and 'flag' in original_result.action.lower():
            original_label = 'Flagged'
        else:
            original_label = 'Rejected'
        # If override_value matches original, revert to original and set override='No'
        if request.override_value == original_label:
            post.override = 'No'
            post.accepted = original_result.accepted
            post.action = original_result.action
        else:
            post.override = 'Yes'
            if request.override_value == 'Accepted':
                post.accepted = True
                post.action = 'PASS'
            elif request.override_value == 'Rejected':
                post.accepted = False
                post.action = 'BLOCK'
            elif request.override_value == 'Flagged':
                post.accepted = False
                post.action = 'FLAG_MEDIUM'
        db.commit()
        db.refresh(post)
        db.close()
        logger.info(
            "Manual override",
            extra={"extra": {
                "content_id": post.id,
                "original_decision": original_label,
                "overridden_decision": request.override_value,  # if available
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }},
        )
        return {
            "success": True,
            "message": f"Override set to {request.override_value}",
            "post_id": request.post_id,
            "new_status": post.accepted,
            "new_action": post.action,
            "override": post.override
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Override error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    try:
        db = SessionLocal()
        from sqlalchemy import func
        total_posts = db.query(Post).count()
        accepted_posts = db.query(Post).filter(Post.accepted == True).count()
        rejected_posts = total_posts - accepted_posts
        reason_stats = db.query(
            Post.stage, func.count(Post.id).label('count')
        ).filter(Post.accepted == False).group_by(Post.stage).all()
        threat_stats = db.query(
            Post.threat_level, func.count(Post.id).label('count')
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

@app.post("/save_comments")
def save_comments(request: SaveCommentsRequest):
    try:
        db = SessionLocal()
        post = db.query(Post).filter(Post.id == request.post_id).first()
        if not post:
            db.close()
            raise HTTPException(status_code=404, detail="Post not found")
        post.comments = request.comments
        db.commit()
        db.refresh(post)
        db.close()
        logger.info(
            "Comments updated",
            extra={"extra": {
                "content_id": post.id,
                "comments": request.comments,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }},
        )
        return {"success": True, "message": "Comments updated", "post_id": post.id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Save comments error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    host = "0.0.0.0"
    preferred_port = 8000
    logger.info("Starting GuardianAI Content Moderation API...")
    if is_port_available(host, preferred_port):
        port = preferred_port
    else:
        port = find_available_port(host, preferred_port + 1)
        if not port:
            logger.error(f"No available ports found between {preferred_port+1} and 9000")
            sys.exit(1)
    logger.info(f"🚀 API running at http://localhost:{port}")
    logger.info(f"📊 Docs at http://localhost:{port}/docs")
    uvicorn.run(app, host=host, port=port, reload=False)




