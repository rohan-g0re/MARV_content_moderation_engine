from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FinBERT API",
    description="Financial sentiment and fraud detection API using FinBERT model",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load FinBERT model and tokenizer"""
    global model, tokenizer
    
    logger.info("Loading FinBERT model...")
    try:
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Set model to evaluation mode
        model.eval()
        
        logger.info("FinBERT model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load FinBERT model: {e}")
        return False

class TextRequest(BaseModel):
    text: str

class FraudResponse(BaseModel):
    sentiment: str
    fraud_score: float
    confidence: float
    message: str

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
def root():
    return {
        "service": "FinBERT API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
def health_check():
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "loaded"}

@app.post("/fraud", response_model=FraudResponse)
def detect_fraud(request: TextRequest):
    """
    Detect fraud and sentiment in financial text.
    
    Returns:
        - sentiment: "positive", "negative", or "neutral"
        - fraud_score: float between 0.0 and 1.0 (higher = more likely fraud)
        - confidence: confidence score for the prediction
        - message: status message
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input text
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get sentiment and scores
        sentiment_labels = ["positive", "negative", "neutral"]
        predicted_class = torch.argmax(probs, dim=-1).item()
        sentiment = sentiment_labels[predicted_class]
        confidence = probs[0][predicted_class].item()
        
        # Calculate fraud score (negative sentiment is often associated with fraud)
        fraud_score = probs[0][1].item()  # negative sentiment probability
        
        logger.info(f"Predicted sentiment: {sentiment}, fraud_score: {fraud_score:.4f} for text: {request.text[:50]}...")
        
        return FraudResponse(
            sentiment=sentiment,
            fraud_score=fraud_score,
            confidence=confidence,
            message="Prediction successful"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/sentiment")
def analyze_sentiment(request: dict):
    """
    Analyze sentiment for multiple texts.
    
    Request: {"texts": ["text1", "text2", ...]}
    Response: {"results": [{"text": "text1", "sentiment": "positive", "score": 0.8}, ...]}
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    texts = request.get("texts", [])
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    try:
        results = []
        sentiment_labels = ["positive", "negative", "neutral"]
        
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            predicted_class = torch.argmax(probs, dim=-1).item()
            sentiment = sentiment_labels[predicted_class]
            score = probs[0][predicted_class].item()
            
            results.append({
                "text": text,
                "sentiment": sentiment,
                "score": score
            })
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Batch sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@app.post("/reload")
def reload_model():
    """Reload the model (useful for updates)"""
    try:
        success = load_model()
        if success:
            return {"message": "Model reloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    ) 