"""
GuardianAI Content Moderation Core Pipeline - Multi-Stage Architecture

This module implements a modular, registerable multi-stage moderation pipeline:
1. Stage 1 - Heuristic/Lexical Filter
2. Stage 2 - Toxicity Check  
3. Stage 3a - Sentiment Sentinels (FinBERT for sentiment only)
4. Stage 3b - Fraud-Specific Classifier (CiferAI + heuristics)
5. LLM Escalation 2.0 (Chain-of-Thought with dynamic thresholding)

Designed for A/B testing and hot-swapping components.
"""

import re
import json
import logging
import requests
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from transformers import pipeline
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from backend/.env
backend_dir = Path(__file__).parent.parent.parent
env_path = backend_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logging.getLogger(__name__).info(f"Loaded environment variables from {env_path}")
else:
    logging.getLogger(__name__).warning(f"Environment file not found at {env_path}")

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ModerationResult:
    """Structured moderation result for multi-stage pipeline"""
    decision: str  # ACCEPT, BLOCK, ESCALATE
    stage: str     # stage1, stage2, stage3a, stage3b, llm
    reason: str    # rule_name_or_model
    confidence: float = 1.0
    threat_level: str = "low"
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses"""
        return {
            "decision": self.decision,
            "stage": self.stage,
            "reason": self.reason,
            "confidence": self.confidence,
            "threat_level": self.threat_level,
            "metadata": self.metadata or {}
        }

class ModerationStage(ABC):
    """Abstract base class for modular moderation stages"""
    
    @abstractmethod
    def process(self, text: str, context: Dict[str, Any] = None) -> ModerationResult:
        """Process text and return moderation result"""
        pass
    
    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Return the stage name"""
        pass

class Stage1LexicalFilter(ModerationStage):
    """Stage 1: Heuristic/Lexical Filter with fuzzy matching"""
    
    def __init__(self, keywords_file: str = "data/external/words.json"):
        self.keywords_file = Path(keywords_file)
        self.scam_lexicon = self._load_scam_lexicon()
    
    @property
    def stage_name(self) -> str:
        return "stage1"
    
    def _load_scam_lexicon(self) -> List[str]:
        """Load dynamic scam lexicon"""
        try:
            if self.keywords_file.exists():
                with open(self.keywords_file, 'r', encoding='utf-8') as f:
                    keywords = json.load(f)
                logger.info(f"Loaded {len(keywords)} scam lexicon terms")
                return keywords
            else:
                logger.warning(f"Scam lexicon file not found: {self.keywords_file}")
                return ["scammer", "fraud", "counterfeit", "loophole", "double money"]
        except Exception as e:
            logger.error(f"Error loading scam lexicon: {e}")
            return ["scammer", "fraud", "counterfeit", "loophole", "double money"]
    
    def _fuzzy_match(self, text: str, keyword: str) -> bool:
        """Fuzzy matching allowing up to 2 non-alpha chars between letters"""
        # Convert to lowercase and remove non-alphanumeric for matching
        text_clean = re.sub(r'[^a-zA-Z0-9]', '', text.lower())
        keyword_clean = re.sub(r'[^a-zA-Z0-9]', '', keyword.lower())
        
        # Simple fuzzy matching - check if keyword appears with up to 2 extra chars
        pattern = r'.{0,2}'.join(keyword_clean)
        return bool(re.search(pattern, text_clean))
    
    def _check_crypto_addresses(self, text: str) -> bool:
        """Check for crypto wallet addresses, IBANs, Cash App tags"""
        patterns = [
            r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',  # Bitcoin addresses
            r'\b0x[a-fA-F0-9]{40}\b',  # Ethereum addresses
            r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b',  # IBAN
            r'\$[a-zA-Z0-9]{1,20}\b',  # Cash App tags
            r'\b[a-zA-Z0-9]{26,35}\b',  # Generic crypto addresses
        ]
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _check_urls(self, text: str) -> bool:
        """Check for suspicious URLs (placeholder for domain API integration)"""
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, text)
        
        # TODO: Integrate with urlscan.io or WHOIS API
        # For now, flag any URL as potentially suspicious
        return len(urls) > 0
    
    def process(self, text: str, context: Dict[str, Any] = None) -> ModerationResult:
        """Apply lexical filtering with fuzzy matching"""
        text_lower = text.lower()
        
        # Check scam lexicon with fuzzy matching
        for keyword in self.scam_lexicon:
            if self._fuzzy_match(text, keyword):
                return ModerationResult(
                    decision="BLOCK",
                    stage=self.stage_name,
                    reason=f"lexical_rule: {keyword}",
                    confidence=0.9,
                    threat_level="high"
                )
        
        # Check crypto addresses
        if self._check_crypto_addresses(text):
            return ModerationResult(
                decision="BLOCK",
                stage=self.stage_name,
                reason="lexical_rule: crypto_address",
                confidence=0.8,
                threat_level="medium"
            )
        
        # Check URLs
        if self._check_urls(text):
            return ModerationResult(
                decision="BLOCK",
                stage=self.stage_name,
                reason="lexical_rule: suspicious_url",
                confidence=0.7,
                threat_level="medium"
            )
        
        # Pass to next stage
        return ModerationResult(
            decision="ACCEPT",
            stage=self.stage_name,
            reason="lexical_rule: passed",
            confidence=1.0,
            threat_level="low"
        )

class Stage2ToxicityCheck(ModerationStage):
    """Stage 2: Toxicity Check using Detoxify"""
    
    def __init__(self, toxicity_threshold: float = 0.5):
        self.toxicity_threshold = toxicity_threshold
        self.detoxify_classifier = None
        self._initialize_model()
    
    @property
    def stage_name(self) -> str:
        return "stage2"
    
    def _initialize_model(self):
        """Initialize Detoxify model"""
        try:
            logger.info("Loading Detoxify model...")
            self.detoxify_classifier = pipeline("text-classification", model="unitary/toxic-bert")
            logger.info("✅ Detoxify model loaded successfully")
        except Exception as e:
            logger.warning(f"Detoxify model loading failed: {e}")
            self.detoxify_classifier = None

    def process(self, text: str, context: Dict[str, Any] = None) -> ModerationResult:
        """Check toxicity levels"""
        if self.detoxify_classifier is None:
            logger.warning("Detoxify model not available, skipping stage 2")
            return ModerationResult(
                decision="ACCEPT",
                stage=self.stage_name,
                reason="toxicity: model_unavailable",
                confidence=0.5,
                threat_level="low"
            )
        
        try:
            result = self.detoxify_classifier(text)
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            if label in ['toxic', 'label_1'] and score >= self.toxicity_threshold:
                return ModerationResult(
                    decision="BLOCK",
                    stage=self.stage_name,
                    reason=f"toxicity: {score:.3f}",
                    confidence=score,
                    threat_level="high" if score > 0.8 else "medium"
                )
            
            return ModerationResult(
                decision="ACCEPT",
                stage=self.stage_name,
                reason="toxicity: passed",
                confidence=1.0 - score,
                threat_level="low"
            )
            
        except Exception as e:
            logger.error(f"Detoxify error: {e}")
            return ModerationResult(
                decision="ACCEPT",
                stage=self.stage_name,
                reason="toxicity: error",
                confidence=0.5,
                threat_level="low"
            )

class Stage3aSentimentSentinels(ModerationStage):
    """Stage 3a: Sentiment Sentinels using FinBERT (SENTIMENT ONLY)"""
    
    def __init__(self):
        self.finbert_classifier = None
        self._initialize_model()
    
    @property
    def stage_name(self) -> str:
        return "stage3a"
    
    def _initialize_model(self):
        """Initialize FinBERT model for sentiment analysis"""
        try:
            logger.info("Loading FinBERT model for sentiment analysis...")
            self.finbert_classifier = pipeline("text-classification", model="ProsusAI/finbert")
            logger.info("✅ FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"FinBERT model loading failed: {e}")
            self.finbert_classifier = None
    
    def process(self, text: str, context: Dict[str, Any] = None) -> ModerationResult:
        """Check sentiment ONLY - not fraud detection"""
        if self.finbert_classifier is None:
            logger.warning("FinBERT model not available, skipping stage 3a")
            return ModerationResult(
                decision="ACCEPT",
                stage=self.stage_name,
                reason="sentiment: model_unavailable",
                confidence=0.5,
                threat_level="low"
            )
        
        try:
            result = self.finbert_classifier(text)
            sentiment = result[0]['label']
            confidence = result[0]['score']
            
            # Store sentiment info in metadata for stage3b fraud classifier
            metadata = {
                "sentiment": sentiment,
                "sentiment_confidence": confidence
            }
            
            # Flag highly negative sentiment for review (but don't block)
            if sentiment == 'negative' and confidence > 0.7:
                return ModerationResult(
                    decision="ESCALATE",
                    stage=self.stage_name,
                    reason=f"sentiment: highly_negative_{confidence:.3f}",
                    confidence=confidence,
                    threat_level="medium",
                    metadata=metadata
                )
            
            return ModerationResult(
                decision="ACCEPT",
                stage=self.stage_name,
                reason=f"sentiment: {sentiment}_{confidence:.3f}",
                confidence=confidence,
                threat_level="low",
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"FinBERT error: {e}")
            return ModerationResult(
                decision="ACCEPT",
                stage=self.stage_name,
                reason="sentiment: error",
                confidence=0.5,
                threat_level="low"
            )

class Stage3bFraudClassifier(ModerationStage):
    """Stage 3b: Fraud-Specific Classifier (CiferAI + heuristics)"""
    
    def __init__(self):
        self.cifer_classifier = None
        self.cifer_api_key = None
        self._initialize_cifer_model()
    
    @property
    def stage_name(self) -> str:
        return "stage3b"
    
    def _initialize_cifer_model(self):
        """Initialize CiferAI fraud detection model"""
        try:
            logger.info("Loading CiferAI fraud detection model...")
            
            # Option 1: Use HuggingFace Transformers (if model is available)
            ciferai_token = os.getenv("ModerationAPP")
            if ciferai_token:
                try:
                    self.cifer_classifier = pipeline(
                        "text-classification",
                        model="CiferAI/cifer-fraud-detection-k1-a",
                        use_auth_token=ciferai_token
                    )
                    logger.info("✅ CiferAI model loaded via HuggingFace Inference API")
                    return
                except Exception as hf_error:
                    logger.warning(f"HuggingFace CiferAI model not available: {hf_error}")
            else:
                logger.warning("No ModerationAPP token found for CiferAI")
            
            # Option 2: Use CiferAI API (if you have API access)
            self.cifer_api_key = os.getenv("ModerationAPP")
            if self.cifer_api_key:
                logger.info("✅ CiferAI API configured")
                return
            
            # Option 3: Use local fine-tuned model
            # Uncomment if you have a local fine-tuned fraud detection model
            # local_model_path = "models/fraud-detection-finetuned"
            # if os.path.exists(local_model_path):
            #     self.cifer_classifier = pipeline("text-classification", model=local_model_path)
            #     logger.info("✅ Local fraud detection model loaded")
            #     return
            
            # Option 4: Use alternative fraud detection models
            alternative_models = [
                "microsoft/DialoGPT-medium",
                "distilbert-base-uncased",  # General purpose
                "ProsusAI/finbert",  # Financial sentiment (can be adapted)
            ]
            
            for model_name in alternative_models:
                try:
                    self.cifer_classifier = pipeline("text-classification", model=model_name)
                    logger.info(f"✅ Alternative fraud detection model loaded: {model_name}")
                    return
                except Exception as alt_error:
                    logger.warning(f"Model {model_name} loading failed: {alt_error}")
                    continue
            
            logger.warning("No CiferAI model available, using heuristics only")
            self.cifer_classifier = None
            
        except Exception as e:
            logger.warning(f"CiferAI model loading failed: {e}")
            self.cifer_classifier = None
    
    def _cifer_fraud_detection(self, text: str) -> Tuple[float, str]:
        """CiferAI fraud detection with multiple integration options"""
        
        # Option 1: HuggingFace Transformers Pipeline
        if self.cifer_classifier is not None:
            try:
                result = self.cifer_classifier(text)
                score = result[0]['score']
                label = result[0]['label']
                
                # Normalize score based on label
                if label.lower() in ['fraud', 'scam', 'malicious', 'label_1']:
                    return score, f"hf_{label}"
                else:
                    return 1.0 - score, f"hf_{label}"
                    
            except Exception as e:
                logger.error(f"HuggingFace CiferAI error: {e}")
                return 0.0, "hf_error"
        
        # Option 2: CiferAI API (if configured)
        if hasattr(self, 'cifer_api_key') and self.cifer_api_key:
            return self._call_cifer_api(text)
        
        # Option 3: Enhanced heuristic fallback
        return self._enhanced_heuristic_fraud_detection(text)
    
    def _enhanced_heuristic_fraud_detection(self, text: str) -> Tuple[float, str]:
        """Enhanced heuristic fraud detection when CiferAI is unavailable"""
        text_lower = text.lower()
        
        # Financial fraud indicators
        fraud_indicators = {
            'high_confidence': [
                'counterfeit', 'fake money', 'fake bills', 'undetectable',
                'double your money', 'triple your money', 'get rich quick',
                'loophole', 'legal loophole', 'guaranteed returns',
                'high quality counterfeit', 'ping me for details'
            ],
            'medium_confidence': [
                'investment opportunity', 'quick money', 'easy money',
                'financial freedom', 'passive income', 'crypto investment',
                'forex trading', 'binary options', 'pyramid scheme'
            ],
            'low_confidence': [
                'investment', 'trading', 'profit', 'earn money',
                'financial advice', 'money making', 'business opportunity'
            ]
        }
        
        max_score = 0.0
        matched_indicator = ""
        
        # Check high confidence indicators
        for indicator in fraud_indicators['high_confidence']:
            if indicator in text_lower:
                return 0.9, f"heuristic_high_{indicator}"
        
        # Check medium confidence indicators
        for indicator in fraud_indicators['medium_confidence']:
            if indicator in text_lower:
                max_score = max(max_score, 0.7)
                matched_indicator = indicator
        
        # Check low confidence indicators (require multiple)
        low_indicators = [ind for ind in fraud_indicators['low_confidence'] if ind in text_lower]
        if len(low_indicators) >= 2:
            max_score = max(max_score, 0.5)
            matched_indicator = f"multiple_low_{len(low_indicators)}"
        
        return max_score, f"heuristic_{matched_indicator}" if matched_indicator else "heuristic_clean"
    
    def _call_cifer_api(self, text: str) -> Tuple[float, str]:
        """Call CiferAI API via HuggingFace Inference API"""
        try:
            # Use HuggingFace Inference API for CiferAI
            url = "https://api-inference.huggingface.co/models/CiferAI/cifer-fraud-detection-k1-a"
            headers = {
                "Authorization": f"Bearer {self.cifer_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": text
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                # Handle HuggingFace inference API response format
                if isinstance(result, list) and len(result) > 0:
                    score = result[0].get('score', 0.0)
                    label = result[0].get('label', 'unknown')
                    return score, f"api_{label}"
                else:
                    return 0.0, "api_invalid_response"
            else:
                logger.error(f"CiferAI API error: {response.status_code} - {response.text}")
                return 0.0, "api_error"
                
        except Exception as e:
            logger.error(f"CiferAI API call failed: {e}")
            return 0.0, "api_error"
    
    def _heuristic_fraud_detection(self, text: str) -> Tuple[float, str]:
        """Heuristic fraud detection patterns"""
        fraud_patterns = [
            (r"(?i)\b(?:double|triple|quadruple)\b.*\b(?:bank|balance|money|cash)\b", 0.9),
            (r"(?i)\b(?:counterfeit|fake)\b.*\b(?:cash|money|bills?|currency)\b", 0.95),
            (r"(?i)\b(?:loophole)\b.*\b(?:legal|legally)\b.*\b(?:money|profit|earn)\b", 0.85),
            (r"(?i)\b(?:undetectable)\b.*\b(?:bills?|money|cash)\b", 0.9),
            (r"(?i)\b(?:guaranteed)\b.*\b(?:returns?|profits?)\b", 0.7),
            (r"(?i)\b(?:get rich quick|easy money)\b", 0.8),
            (r"(?i)\b(?:high quality)\b.*\b(?:counterfeit|fake|bills?)\b", 0.9),
            (r"(?i)\b(?:ping me|contact me|DM me)\b.*\b(?:counterfeit|fake|illegal)\b", 0.8),
        ]
        
        max_score = 0.0
        matched_pattern = ""
        
        for pattern, score in fraud_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if score > max_score:
                    max_score = score
                    matched_pattern = pattern
        
        return max_score, matched_pattern
    
    def _ensemble_voting(self, cifer_score: float, heuristic_score: float, 
                        sentiment: str = None, sentiment_conf: float = 0.0) -> Tuple[float, str]:
        """Ensemble voting scheme combining CiferAI, heuristics, and sentiment"""
        # Weighted combination: CiferAI (60%) + Heuristics (40%)
        ensemble_score = (cifer_score * 0.6) + (heuristic_score * 0.4)
        
        # Dynamic thresholding: escalate if models disagree
        if sentiment == 'neutral' and (cifer_score > 0.6 or heuristic_score > 0.6):
            ensemble_score = max(ensemble_score, 0.7)  # Boost score for disagreement
        
        reason = f"ensemble: cifer_{cifer_score:.3f}_heuristic_{heuristic_score:.3f}"
        if sentiment:
            reason += f"_sentiment_{sentiment}_{sentiment_conf:.3f}"
        
        return ensemble_score, reason
    
    def process(self, text: str, context: Dict[str, Any] = None) -> ModerationResult:
        """Run fraud-specific classification with ensemble voting"""
        # Get sentiment info from previous stage
        sentiment = None
        sentiment_conf = 0.0
        if context and "stage3a" in context:
            metadata = context["stage3a"].metadata
            if metadata:
                sentiment = metadata.get("sentiment")
                sentiment_conf = metadata.get("sentiment_confidence", 0.0)
        
        # Run CiferAI fraud detection
        cifer_score, cifer_label = self._cifer_fraud_detection(text)
        
        # Run heuristic fraud detection
        heuristic_score, heuristic_pattern = self._heuristic_fraud_detection(text)
        
        # Ensemble voting
        ensemble_score, reason = self._ensemble_voting(
            cifer_score, heuristic_score, sentiment, sentiment_conf
        )
        
        # Decision logic
        if ensemble_score >= 0.8:
            return ModerationResult(
                decision="BLOCK",
                stage=self.stage_name,
                reason=f"fraud_model: {reason}",
                confidence=ensemble_score,
                threat_level="high"
            )
        elif ensemble_score <= 0.2:
            return ModerationResult(
                decision="ACCEPT",
                stage=self.stage_name,
                reason="fraud_model: clean",
                confidence=1.0 - ensemble_score,
                threat_level="low"
            )
        else:
            return ModerationResult(
                decision="ESCALATE",
                stage=self.stage_name,
                reason=f"fraud_model: uncertain_{reason}",
                confidence=ensemble_score,
                threat_level="medium"
            )

class LLMEscalation(ModerationStage):
    """LLM Escalation 2.0 with Chain-of-Thought reasoning"""
    
    def __init__(self, groq_api_key: str = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
    
    @property
    def stage_name(self) -> str:
        return "llm"
    
    def _call_llm_with_cot(self, text: str, context: Dict[str, Any] = None) -> Tuple[str, float]:
        """Call LLM with Chain-of-Thought reasoning (hidden from user)"""
        # Build context from previous stages
        context_info = ""
        if context:
            for stage_name, result in context.items():
                context_info += f"\n{stage_name}: {result.reason}"
        
        prompt = f"""
You are a sophisticated scammer analyzing a social media post. Think step-by-step:

Post: "{text}"

Previous stage results:{context_info}

Step-by-step analysis:
1. What is the main intent of this post?
2. Does it contain financial promises or guarantees?
3. Is there urgency or pressure tactics?
4. Would this be effective for fraud/scams?
5. Are there any red flags I would recognize as a scammer?

After your analysis, respond with ONLY "FRAUD" or "CLEAN" (no explanation).
"""
        
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,  # Allow for chain-of-thought
                "temperature": 0.1,
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            resp_json = response.json()
            
            if "choices" in resp_json and resp_json["choices"]:
                content = resp_json["choices"][0]["message"]["content"].strip().upper()
                
                # Extract final decision (strip chain-of-thought reasoning)
                if "FRAUD" in content:
                    return "FRAUD", 0.9
                elif "CLEAN" in content:
                    return "CLEAN", 0.9
                else:
                    return "UNCERTAIN", 0.5
            else:
                return "ERROR", 0.5
                
        except Exception as e:
            logger.error(f"LLM escalation error: {e}")
            return "ERROR", 0.5
    
    def process(self, text: str, context: Dict[str, Any] = None) -> ModerationResult:
        """Process escalation with LLM Chain-of-Thought"""
        decision, confidence = self._call_llm_with_cot(text, context)
        
        if decision == "FRAUD":
            return ModerationResult(
                decision="BLOCK",
                stage=self.stage_name,
                reason="llm: fraud_detected_cot",
                confidence=confidence,
                threat_level="high"
            )
        elif decision == "CLEAN":
            return ModerationResult(
                decision="ACCEPT",
                stage=self.stage_name,
                reason="llm: clean_cot",
                confidence=confidence,
                threat_level="low"
            )
        else:
            return ModerationResult(
                decision="BLOCK",  # Default to block on uncertainty
                stage=self.stage_name,
                reason="llm: uncertain_cot",
                confidence=confidence,
                threat_level="medium"
            )

class ModerationPipeline:
    """Modular multi-stage moderation pipeline"""
    
    def __init__(self):
        self.stages: List[ModerationStage] = []
        self.stage_results: Dict[str, ModerationResult] = {}
    
    def register_stage(self, stage: ModerationStage):
        """Register a moderation stage"""
        self.stages.append(stage)
        logger.info(f"Registered stage: {stage.stage_name}")
    
    def process(self, text: str) -> ModerationResult:
        """Process text through all registered stages"""
        context = {}
        
        for stage in self.stages:
            result = stage.process(text, context)
            self.stage_results[stage.stage_name] = result
            
            # Stop processing if decision is BLOCK
            if result.decision == "BLOCK":
                logger.info(f"Content blocked at {stage.stage_name}: {result.reason}")
        return result
            # Continue to next stage for ACCEPT or ESCALATE
            context[stage.stage_name] = result
        
        # If we reach here, all stages passed or escalated
        # Find the last stage that made a decision
        for stage in reversed(self.stages):
            if stage.stage_name in self.stage_results:
                return self.stage_results[stage.stage_name]
        
        # Fallback
        return ModerationResult(
            decision="ACCEPT",
            stage="pipeline",
            reason="all_stages_passed",
            confidence=1.0,
            threat_level="low"
        )
    
    def get_stage_results(self) -> Dict[str, ModerationResult]:
        """Get results from all stages"""
        return self.stage_results.copy()

# Factory function to create default pipeline
def create_default_pipeline(keywords_file: str = "data/external/words.json", 
                          groq_api_key: str = None) -> ModerationPipeline:
    """Create a default moderation pipeline with all stages"""
    pipeline = ModerationPipeline()
    
    # Register stages in order
    pipeline.register_stage(Stage1LexicalFilter(keywords_file))
    pipeline.register_stage(Stage2ToxicityCheck(toxicity_threshold=0.5))
    pipeline.register_stage(Stage3aSentimentSentinels())  # FinBERT for sentiment only
    pipeline.register_stage(Stage3bFraudClassifier())     # CiferAI + heuristics for fraud
    pipeline.register_stage(LLMEscalation(groq_api_key))  # LLM escalation 2.0
    
    return pipeline

# Legacy compatibility
class GuardianModerationEngine:
    """Legacy wrapper for backward compatibility"""
    
    def __init__(self, keywords_file: str = "data/external/words.json"):
        self.pipeline = create_default_pipeline(keywords_file)
    
    def moderate_content(self, content: str) -> ModerationResult:
        """Main moderation entrypoint"""
        return self.pipeline.process(content)

    def get_model_status(self) -> Dict:
        """Get status of loaded models"""
        return {
            "pipeline_stages": len(self.pipeline.stages),
            "stage_names": [stage.stage_name for stage in self.pipeline.stages]
        } 