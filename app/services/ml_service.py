"""
ML service for toxicity and fraud detection using Detoxify and FinBERT
"""

import logging
from typing import Dict, Any, List, Optional
from transformers import pipeline
from app.core.config import settings

logger = logging.getLogger(__name__)


class MLService:
    """Service for ML-based content analysis"""
    
    def __init__(self):
        """Initialize ML service with Detoxify and FinBERT models"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("MLService initialized")
        
        # Initialize models as None - will be loaded on first use
        self.detoxify_model = None
        self.finbert_model = None
        self.models_loaded = False
        
        # Load models synchronously
        self._load_models_sync()
    
    def _load_models_sync(self):
        """Load ML models synchronously"""
        try:
            self.logger.info("Loading ML models...")
            
            # Load Detoxify model
            self.detoxify_model = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                return_all_scores=True
            )
            
            # Load FinBERT model
            self.finbert_model = pipeline(
                "text-classification", 
                model="ProsusAI/finbert",
                return_all_scores=True
            )
            
            self.models_loaded = True
            self.logger.info("ML models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load ML models: {e}")
            self.models_loaded = False
    
    async def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analyze content using ML models
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dictionary with ML analysis results
        """
        try:
            if not self.models_loaded:
                return self._get_fallback_result("ML models not loaded")
            
            # Run toxicity analysis
            toxicity_result = await self._analyze_toxicity(content)
            
            # Run financial analysis
            financial_result = await self._analyze_financial(content)
            
            # Combine results
            overall_score = self._calculate_overall_score(toxicity_result, financial_result)
            confidence = self._calculate_confidence(toxicity_result, financial_result)
            
            return {
                "overall_score": overall_score,
                "confidence": confidence,
                "detections": self._get_detections(toxicity_result, financial_result),
                "toxicity_score": toxicity_result["score"],
                "toxicity_label": toxicity_result["label"],
                "fraud_score": financial_result["score"],
                "fraud_label": financial_result["label"],
                "models_loaded": self.models_loaded
            }
            
        except Exception as e:
            logger.error(f"Error in ML analysis: {e}")
            return self._get_fallback_result(str(e))
    
    async def _analyze_toxicity(self, content: str) -> Dict[str, Any]:
        """Analyze content for toxicity using Detoxify"""
        try:
            if not self.detoxify_model:
                return {"score": 0.0, "label": "unknown", "error": "Model not loaded"}
            
            # Run inference
            results = self.detoxify_model(content)
            
            # Find the most toxic category
            max_score = 0.0
            max_label = "non-toxic"
            
            for result in results[0]:
                if result["score"] > max_score:
                    max_score = result["score"]
                    max_label = result["label"]
            
            # Map labels to more readable names
            label_mapping = {
                "toxic": "toxic",
                "severe_toxic": "severely toxic",
                "obscene": "obscene",
                "threat": "threatening",
                "insult": "insulting",
                "identity_hate": "hate speech"
            }
            
            readable_label = label_mapping.get(max_label, max_label)
            
            return {
                "score": max_score,
                "label": readable_label,
                "raw_label": max_label,
                "all_scores": {r["label"]: r["score"] for r in results[0]}
            }
            
        except Exception as e:
            logger.error(f"Error in toxicity analysis: {e}")
            return {"score": 0.0, "label": "unknown", "error": str(e)}
    
    async def _analyze_financial(self, content: str) -> Dict[str, Any]:
        """Analyze content for financial fraud using FinBERT"""
        try:
            if not self.finbert_model:
                return {"score": 0.0, "label": "unknown", "error": "Model not loaded"}
            
            # Run inference
            results = self.finbert_model(content)
            
            # FinBERT returns: positive, negative, neutral
            scores = {r["label"]: r["score"] for r in results[0]}
            
            # Consider negative sentiment as potential fraud
            fraud_score = scores.get("negative", 0.0)
            fraud_label = "suspicious" if fraud_score > settings.FRAUD_THRESHOLD else "legitimate"
            
            return {
                "score": fraud_score,
                "label": fraud_label,
                "raw_label": "negative",
                "all_scores": scores,
                "positive_score": scores.get("positive", 0.0),
                "negative_score": scores.get("negative", 0.0),
                "neutral_score": scores.get("neutral", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error in financial analysis: {e}")
            return {"score": 0.0, "label": "unknown", "error": str(e)}
    
    def _calculate_overall_score(self, toxicity_result: Dict[str, Any], financial_result: Dict[str, Any]) -> float:
        """Calculate overall threat score from ML results"""
        toxicity_score = toxicity_result.get("score", 0.0)
        fraud_score = financial_result.get("score", 0.0)
        
        # Weighted average: 60% toxicity, 40% fraud
        overall_score = (0.6 * toxicity_score) + (0.4 * fraud_score)
        
        return min(overall_score, 1.0)
    
    def _calculate_confidence(self, toxicity_result: Dict[str, Any], financial_result: Dict[str, Any]) -> float:
        """Calculate confidence in ML results"""
        # Base confidence from model scores
        toxicity_conf = toxicity_result.get("score", 0.0)
        fraud_conf = financial_result.get("score", 0.0)
        
        # Average confidence
        confidence = (toxicity_conf + fraud_conf) / 2
        
        # Boost confidence if both models agree on high scores
        if toxicity_conf > 0.7 and fraud_conf > 0.7:
            confidence = min(confidence + 0.1, 1.0)
        
        return confidence
    
    def _get_detections(self, toxicity_result: Dict[str, Any], financial_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get list of detections from ML analysis"""
        detections = []
        
        # Toxicity detections
        if toxicity_result.get("score", 0.0) > settings.TOXICITY_THRESHOLD:
            detections.append({
                "type": "toxicity",
                "score": toxicity_result["score"],
                "label": toxicity_result["label"],
                "severity": "high" if toxicity_result["score"] > 0.8 else "medium"
            })
        
        # Financial fraud detections
        if financial_result.get("score", 0.0) > settings.FRAUD_THRESHOLD:
            detections.append({
                "type": "fraud",
                "score": financial_result["score"],
                "label": financial_result["label"],
                "severity": "high" if financial_result["score"] > 0.8 else "medium"
            })
        
        return detections
    
    def _get_fallback_result(self, error_message: str) -> Dict[str, Any]:
        """Get fallback result when ML models fail"""
        return {
            "overall_score": 0.0,
            "confidence": 0.0,
            "detections": [],
            "toxicity_score": 0.0,
            "toxicity_label": "unknown",
            "fraud_score": 0.0,
            "fraud_label": "unknown",
            "models_loaded": False,
            "error": error_message
        }
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of ML models"""
        return {
            "models_loaded": self.models_loaded,
            "toxicity_model": "loaded" if self.detoxify_model else "not loaded",
            "finbert_model": "loaded" if self.finbert_model else "not loaded",
            "toxicity_threshold": settings.TOXICITY_THRESHOLD,
            "fraud_threshold": settings.FRAUD_THRESHOLD
        }
    
    async def test_models(self) -> Dict[str, Any]:
        """Test ML models with sample content"""
        test_content = "This is a test message to verify ML models are working."
        
        try:
            result = await self.analyze(test_content)
            return {
                "status": "success",
                "test_content": test_content,
                "result": result
            }
        except Exception as e:
            return {
                "status": "error",
                "test_content": test_content,
                "error": str(e)
            } 