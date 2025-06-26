from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from transformers import pipeline
import datetime
import re
import json
import os

# Initialize FastAPI
app = FastAPI(title="Content Moderation API", version="1.0.0")

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
print(f"[INFO] Using database at: {os.path.abspath('moderation.db')}")
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

# Load keywords from file
def load_keywords():
    """Load keywords from words.json"""
    keywords = []
    try:
        with open("words.json", "r", encoding="utf-8") as f:
            keywords = json.load(f)
        print(f"✅ Loaded {len(keywords)} keywords from words.json")
    except FileNotFoundError:
        print("⚠️ words.json not found, using default keywords")
        # Default keywords if file doesn't exist
        keywords = ["scammer", "fraud", "hate", "violence", "spam", "badword1", "badword2"]
    except json.JSONDecodeError as e:
        print(f"⚠️ Error parsing words.json: {e}")
        keywords = ["scammer", "fraud", "hate", "violence", "spam"]
    return keywords

# Initialize models
print("Loading models...")
try:
    # Detoxify for toxicity detection
    toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")
    
    # FinBERT for financial sentiment
    finbert_classifier = pipeline("text-classification", model="ProsusAI/finbert")
    
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"⚠️ Model loading failed: {e}")
    print("Using fallback rule-based detection")
    toxicity_classifier = None
    finbert_classifier = None

# Moderation functions
def rule_based_filter(text: str) -> tuple[bool, str]:
    """Step 1: Rule-based filtering (case-insensitive, whole word)"""
    keywords = load_keywords()
    text_lower = text.lower()
    for keyword in keywords:
        # Use regex for whole word match, case-insensitive
        if re.search(rf'\\b{re.escape(keyword.lower())}\\b', text_lower):
            return False, f"Rule-based: {keyword}"
    # Check regex patterns
    patterns = [
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b",
        r"\\b\\d{10,}\\b",  # Phone numbers
        # Violence/threat/crime regex
        r"\\b(kill|bash|hack|steal|threat|attack|rape|murder|shoot|stab|destroy|burn|harass|stalk|blackmail|assault|abuse|bully|rob|terror|terrorist|explosive|bomb|kidnap|extort)\\b"
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False, f"Rule-based: {pattern}"
    return True, ""

def detoxify_check(text: str) -> tuple[bool, str]:
    """Step 2: Detoxify toxicity detection (fix label logic)"""
    if toxicity_classifier is None:
        return True, ""
    try:
        result = toxicity_classifier(text)
        label = result[0]['label'].lower()
        score = result[0]['score']
        if label in ['toxic', 'label_1'] and score > 0.5:
            return False, f"Toxic (Detoxify): {score:.2f}"
        return True, ""
    except Exception as e:
        print(f"Detoxify error: {e}")
        return True, ""

def finbert_check(text: str) -> tuple[bool, str]:
    """Step 3: FinBERT fraud detection"""
    if finbert_classifier is None:
        return True, ""
    
    try:
        result = finbert_classifier(text)
        sentiment = result[0]['label']
        confidence = result[0]['score']
        
        # Consider negative sentiment as potential fraud
        if sentiment == 'negative' and confidence > 0.7:
            return False, f"Potential Fraud (FinBERT): {sentiment} ({confidence:.2f})"
        return True, ""
    except Exception as e:
        print(f"FinBERT error: {e}")
        return True, ""

def moderate_content(content: str) -> tuple[bool, str]:
    """Main moderation pipeline"""
    # Step 1: Rule-based filtering
    passed, reason = rule_based_filter(content)
    if not passed:
        return False, reason
    
    # Step 2: Detoxify
    passed, reason = detoxify_check(content)
    if not passed:
        return False, reason
    
    # Step 3: FinBERT
    passed, reason = finbert_check(content)
    if not passed:
        return False, reason
    
    return True, "All checks passed"

# API endpoints
@app.get("/")
def root():
    return {"message": "Content Moderation API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/moderate", response_model=ModerationResponse)
def moderate_post(request: ModerationRequest):
    """Main moderation endpoint"""
    try:
        # Run moderation pipeline
        accepted, reason = moderate_content(request.content)
        
        # Save to database
        db = SessionLocal()
        post = Post(
            content=request.content,
            accepted=accepted,
            reason=reason
        )
        db.add(post)
        db.commit()
        db.refresh(post)
        print(f"[INFO] Saved post ID {post.id} to DB at {os.path.abspath('moderation.db')}")
        db.close()
        
        return ModerationResponse(
            accepted=accepted,
            reason=reason,
            id=post.id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/posts")
def get_posts():
    """Get all moderated posts"""
    db = SessionLocal()
    posts = db.query(Post).order_by(Post.created_at.desc()).all()
    db.close()
    
    return [
        {
            "id": post.id,
            "content": post.content,
            "accepted": post.accepted,
            "reason": post.reason,
            "created_at": post.created_at.isoformat()
        }
        for post in posts
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 