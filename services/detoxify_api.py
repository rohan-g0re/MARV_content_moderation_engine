from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from detoxify import Detoxify
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Detoxify API",
    description="Toxicity detection API using Detoxify model",
    version="1.0.0"
)

# Load model once at startup
logger.info("Loading Detoxify model...")
try:
    model = Detoxify('original')
    logger.info("Detoxify model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load Detoxify model: {e}")
    model = None

class TextRequest(BaseModel):
    text: str

class ToxicityResponse(BaseModel):
    toxicity: float
    message: str

@app.get("/")
def root():
    return {
        "service": "Detoxify API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict", response_model=ToxicityResponse)
def predict_toxicity(request: TextRequest):
    """
    Predict toxicity score for given text.
    
    Returns:
        - toxicity: float between 0.0 and 1.0
        - message: status message
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Predict toxicity
        results = model.predict(request.text)
        toxicity_score = results["toxicity"]
        
        logger.info(f"Predicted toxicity: {toxicity_score:.4f} for text: {request.text[:50]}...")
        
        return ToxicityResponse(
            toxicity=toxicity_score,
            message="Prediction successful"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch")
def batch_predict(request: dict):
    """
    Batch predict toxicity for multiple texts.
    
    Request: {"texts": ["text1", "text2", ...]}
    Response: {"results": [{"text": "text1", "toxicity": 0.1}, ...]}
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    texts = request.get("texts", [])
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    try:
        results = []
        for text in texts:
            prediction = model.predict(text)
            results.append({
                "text": text,
                "toxicity": prediction["toxicity"]
            })
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    ) 