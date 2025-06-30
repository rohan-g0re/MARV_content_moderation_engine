import os
import datetime
import logging
import socket
import sys
import pathlib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
import requests

from dotenv import load_dotenv

# Load .env
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
dotenv_path = BASE_DIR / ".env"
print(f"Loading .env from: {dotenv_path}")
load_dotenv(dotenv_path)
print("DEBUG: GROQ_API_KEY is:", os.getenv("GROQ_API_KEY"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Import your moderation engine! ----
from app.core.moderation import GuardianModerationEngine

DATABASE_URL = "sqlite:///./moderation.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
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
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    llm_explanation = Column(Text, default="")
    llm_troublesome_words = Column(Text, default="")
    llm_suggestion = Column(Text, default="")

Base.metadata.create_all(bind=engine)

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
        logging.warning(f"LLM raw response: {resp_json}")

        if "choices" in resp_json and resp_json["choices"]:
            content = resp_json["choices"][0]["message"]["content"]
            try:
                import json as _json
                parsed = _json.loads(content)
                return (
                    parsed.get("explanation", ""),
                    parsed.get("troublesome_words", []),
                    parsed.get("suggestion", "")
                )
            except Exception as ex:
                logging.warning(f"LLM content is not JSON. Content: {content}. Exception: {ex}")
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
    try:
        result = moderation_engine.moderate_content(request.content)
        llm_explanation, llm_troublesome_words, llm_suggestion = "", [], ""

        # Call LLM only if blocked or flagged by *any* strict action (FLAG, BLOCK, etc)
        if result.action.upper() in ("BLOCK", "FLAG", "FLAG_LOW", "FLAG_MEDIUM", "FLAG_HIGH"):
            llm_explanation, llm_troublesome_words, llm_suggestion = get_llm_explanation_and_suggestion(request.content)

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
            llm_explanation=llm_explanation,
            llm_troublesome_words=_json.dumps(llm_troublesome_words),
            llm_suggestion=llm_suggestion
        )
        db.add(post)
        db.commit()
        db.refresh(post)
        db.close()
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
            suggestion=llm_suggestion
        )
    except Exception as e:
        logger.error(f"Moderation error: {e}")
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
                "confidence": float(post.confidence) if post.confidence else 1.0,
                "stage": post.stage,
                "band": getattr(post, 'band', 'SAFE'),
                "action": getattr(post, 'action', 'PASS'),
                "llm_explanation": post.llm_explanation,
                "llm_troublesome_words": _json.loads(post.llm_troublesome_words or "[]"),
                "llm_suggestion": post.llm_suggestion,
                "created_at": post.created_at.isoformat()
            }
            for post in posts
        ]
    except Exception as e:
        logger.error(f"Database error: {e}")
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
