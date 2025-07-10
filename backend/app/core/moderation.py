"""
GuardianAI Content Moderation Core Pipeline - Multi-Stage Architecture

Stages:
1. Stage 1 - Heuristic/Lexical Filter (Regex/Fuzzy)
2. Stage LR - Logistic Regression ML
3. Stage 2 - Detoxify (Toxicity)
4. Stage 3a - FinBERT (Sentiment)
5. Stage 3b - CiferAI/Heuristic (Fraud)
6. LLM Escalation (Chain-of-Thought, fallback)
7. Quant Mode - Quantitative Analysis & Risk Scoring
"""

import re
import json
import logging
import requests
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from transformers import pipeline
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from dotenv import load_dotenv
import joblib
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from backend/.env
backend_dir = Path(__file__).parent.parent.parent
env_path = backend_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logging.getLogger(__name__).info(f"Loaded environment variables from {env_path}")
else:
    logging.getLogger(__name__).warning(f"Environment file not found at {env_path}")

# Logging config
logger = logging.getLogger(__name__)

@dataclass
class ModerationResult:
    decision: str         # ACCEPT, BLOCK, ESCALATE
    stage: str            # Stage name
    reason: str           # Rule or model
    confidence: float = 1.0
    threat_level: str = "low"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Quantitative analysis fields
    risk_score: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    statistical_significance: float = 0.0
    anomaly_score: float = 0.0
    quant_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def action(self) -> str:
        """Map decision to action for compatibility with main.py"""
        if self.decision == "BLOCK":
            return "BLOCK"
        elif self.decision == "ESCALATE":
            return "FLAG_MEDIUM"
        else:  # ACCEPT
            return "PASS"
    
    @property
    def accepted(self) -> bool:
        """Map decision to accepted boolean for compatibility with main.py"""
        return self.decision == "ACCEPT"
    
    @property
    def band(self) -> str:
        """Map threat_level to band for compatibility with main.py"""
        if self.threat_level == "high":
            return "DANGER"
        elif self.threat_level == "medium":
            return "WARNING"
        else:  # low
            return "SAFE"

    def to_dict(self) -> Dict:
        return {
            "decision": self.decision,
            "stage": self.stage,
            "reason": self.reason,
            "confidence": self.confidence,
            "threat_level": self.threat_level,
            "risk_score": self.risk_score,
            "confidence_interval": self.confidence_interval,
            "statistical_significance": self.statistical_significance,
            "anomaly_score": self.anomaly_score,
            "quant_metrics": self.quant_metrics,
            "metadata": self.metadata or {}
        }

class ModerationStage(ABC):
    @abstractmethod
    def process(self, text: str, context: Dict[str, Any] = None) -> ModerationResult:
        pass

    @property
    @abstractmethod
    def stage_name(self) -> str:
        pass

# Stage 1: Heuristic/Lexical Filter
class Stage1LexicalFilter(ModerationStage):
    def __init__(self, keywords_file: str = "data/external/words.json"):
        self.keywords_file = Path(keywords_file)
        self.scam_lexicon = self._load_scam_lexicon()

    @property
    def stage_name(self):
        return "stage1"

    def _load_scam_lexicon(self) -> List[str]:
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
        text_clean = re.sub(r'[^a-zA-Z0-9]', '', text.lower())
        keyword_clean = re.sub(r'[^a-zA-Z0-9]', '', keyword.lower())
        pattern = r'.{0,2}'.join(keyword_clean)
        return bool(re.search(pattern, text_clean))

    def _check_crypto_addresses(self, text: str) -> bool:
        patterns = [
            r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',  # Bitcoin
            r'\b0x[a-fA-F0-9]{40}\b',               # Ethereum
            r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b',  # IBAN
            r'\$[a-zA-Z0-9]{3,20}\b',               # Cash App
            r'\b[a-zA-Z0-9]{26,35}\b',              # Generic crypto
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def _check_urls(self, text: str) -> bool:
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, text)
        return len(urls) > 0

    def process(self, text: str, context: Dict[str, Any] = None) -> ModerationResult:
        context = context or {}
        for keyword in self.scam_lexicon:
            if self._fuzzy_match(text, keyword):
                logger.info(f"Stage1: Blocked for keyword: {keyword}")
                return ModerationResult(
                    decision="BLOCK",
                    stage=self.stage_name,
                    reason=f"lexical_rule: {keyword}",
                    confidence=0.9,
                    threat_level="high"
                )
        if self._check_crypto_addresses(text):
            return ModerationResult(
                decision="BLOCK",
                stage=self.stage_name,
                reason="lexical_rule: crypto_address",
                confidence=0.8,
                threat_level="medium"
            )
        if self._check_urls(text):
            return ModerationResult(
                decision="ESCALATE",
                stage=self.stage_name,
                reason="lexical_rule: suspicious_url",
                confidence=0.6,
                threat_level="medium"
            )
        return ModerationResult(
            decision="ACCEPT",
            stage=self.stage_name,
            reason="lexical_rule: passed",
            confidence=1.0,
            threat_level="low"
        )

# Stage LR: Logistic Regression ML
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb  # <-- Make sure this import is at top level!
from .features import extract_features  # or adjust if import path is different

class StageLGBMModeration(ModerationStage):
    def __init__(self, model_path=None, support_path=None):
        # Base dir = backend/app/core
        base_dir = Path(__file__).resolve().parent
        # Use given paths or default to files in this folder
        model_path = Path(model_path) if model_path else (base_dir / "lgbm_moderation.txt")
        support_path = Path(support_path) if support_path else (base_dir / "lgbm_support.pkl")
       

        try:
            # Ensure correct str for LightGBM and joblib
            self.model = lgb.Booster(model_file=str(model_path))
            self.keywords, self.feature_names, self.tfidf_vec = joblib.load(str(support_path))
            logger.info(f"Loaded LGBM moderation model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load LGBM model: {e}")
            self.model, self.keywords, self.feature_names, self.tfidf_vec = None, [], [], None

    @property
    def stage_name(self):
        return "stage_lgbm"

    def process(self, text: str, context: Dict[str, Any] = None) -> ModerationResult:
        context = context or {}
        if self.model is None:
            print("[LGBM] Model not loaded.")
            return ModerationResult("ACCEPT", self.stage_name, "lgbm_model_unavailable", 0.5, "low")
        features = extract_features(text, self.keywords, self.tfidf_vec)
        print("[LGBM] Features for input:", features)
        X = pd.DataFrame([[features[f] for f in self.feature_names]], columns=self.feature_names)
        print("[LGBM] Feature DataFrame:", X)
        probs = self.model.predict(X)[0]
        print(f"[LGBM] Model probabilities: {probs}")
        predicted_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
        print(f"[LGBM] Predicted class: {predicted_class} (0=PASS, 1=FLAG, 2=BLOCK), Confidence: {confidence:.4f}")
        if predicted_class == 2:
            print("[LGBM] BLOCKED by LGBM stage!")
            return ModerationResult("BLOCK", self.stage_name, "lgbm_flagged_block", confidence, "high")
        elif predicted_class == 1:
            print("[LGBM] FLAGGED by LGBM stage!")
            return ModerationResult("ESCALATE", self.stage_name, "lgbm_flagged_flag", confidence, "medium")
        else:
            print("[LGBM] PASSED by LGBM stage.")
            return ModerationResult("ACCEPT", self.stage_name, "lgbm_passed", confidence, "low")

# Stage 2: Detoxify Toxicity
class Stage2ToxicityCheck(ModerationStage):
    def __init__(self, toxicity_threshold: float = 0.5):
        self.toxicity_threshold = toxicity_threshold
        self.detoxify_classifier = None
        self._initialize_model()

    @property
    def stage_name(self):
        return "stage2"

    def _initialize_model(self):
        try:
            logger.info("Loading Detoxify model...")
            self.detoxify_classifier = pipeline("text-classification", model="unitary/toxic-bert")
            logger.info("✅ Detoxify model loaded successfully")
        except Exception as e:
            logger.warning(f"Detoxify model loading failed: {e}")
            self.detoxify_classifier = None

    def process(self, text: str, context: Dict[str, Any] = None) -> ModerationResult:
        context = context or {}
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
                logger.info(f"Toxicity blocked: score={score:.2f}")
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

# Stage 3a: FinBERT Sentiment
class Stage3aSentimentSentinels(ModerationStage):
    def __init__(self):
        self.finbert_classifier = None
        self._initialize_model()

    @property
    def stage_name(self):
        return "stage3a"

    def _initialize_model(self):
        try:
            logger.info("Loading FinBERT model for sentiment analysis...")
            self.finbert_classifier = pipeline("text-classification", model="ProsusAI/finbert")
            logger.info("✅ FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"FinBERT model loading failed: {e}")
            self.finbert_classifier = None

    def process(self, text: str, context: Dict[str, Any] = None) -> ModerationResult:
        context = context or {}
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
            metadata = {
                "sentiment": sentiment,
                "sentiment_confidence": confidence
            }
            if sentiment == 'negative' and confidence > 0.8:
                logger.info("FinBERT: highly negative sentiment flagged for review.")
                return ModerationResult(
                    decision="ACCEPT",
                    stage=self.stage_name,
                    reason=f"sentiment_warning: highly_negative_{confidence:.3f}",
                    confidence=confidence,
                    threat_level="low",
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

# Stage3bFraudClassifier with Bilic/Mistral and Heuristics
import os
import requests
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class Stage3bFraudClassifier(ModerationStage):
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        logger.info("Stage3b using Mistral and FinGPT via HuggingFace API")

    @property
    def stage_name(self):
        return "stage3b"

    def _mistral_fraud_detection(self, text: str) -> Tuple[float, str]:
        api_url = "https://api-inference.huggingface.co/models/Bilic/Mistral-7B-LLM-Fraud-Detection"
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        payload = {"inputs": text}
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                label = result[0].get('label', '')
                score = result[0].get('score', 0.0)
                return float(score), f"mistral_{label}"
            else:
                logger.warning(f"Empty response from Mistral API: {result}")
                return 0.0, "mistral_api_empty"
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Mistral model not found (404): {api_url}")
                return 0.0, "mistral_api_404"
            else:
                logger.error(f"Mistral API HTTP error: {e}")
                return 0.0, "mistral_api_http_error"
        except requests.exceptions.Timeout:
            logger.warning("Mistral API timeout")
            return 0.0, "mistral_api_timeout"
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            return 0.0, "mistral_api_error"

    def _fingpt_fraud_detection(self, text: str) -> Tuple[float, str]:
        api_url = "https://api-inference.huggingface.co/models/AI4Finance-Foundation/FinGPT-Llama3-8B"
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        payload = {"inputs": text}
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                label = result[0].get('label', '')
                score = result[0].get('score', 0.0)
                return float(score), f"fingpt_{label}"
            else:
                logger.warning(f"Empty response from FinGPT API: {result}")
                return 0.0, "fingpt_api_empty"
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"FinGPT model not found (404): {api_url}")
                return 0.0, "fingpt_api_404"
            else:
                logger.error(f"FinGPT API HTTP error: {e}")
                return 0.0, "fingpt_api_http_error"
        except requests.exceptions.Timeout:
            logger.warning("FinGPT API timeout")
            return 0.0, "fingpt_api_timeout"
        except Exception as e:
            logger.error(f"FinGPT API error: {e}")
            return 0.0, "fingpt_api_error"

    def _heuristic_fraud_detection(self, text: str) -> Tuple[float, str]:
        text_lower = text.lower()
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
        for indicator in fraud_indicators['high_confidence']:
            if indicator in text_lower:
                return 0.9, f"heuristic_high_{indicator}"
        for indicator in fraud_indicators['medium_confidence']:
            if indicator in text_lower:
                max_score = max(max_score, 0.7)
                matched_indicator = indicator
        low_indicators = [ind for ind in fraud_indicators['low_confidence'] if ind in text_lower]
        if len(low_indicators) >= 2:
            max_score = max(max_score, 0.5)
            matched_indicator = f"multiple_low_{len(low_indicators)}"
        return max_score, f"heuristic_{matched_indicator}" if matched_indicator else "heuristic_clean"

    def _ensemble_voting(self, mistral_score: float, fingpt_score: float, heuristic_score: float, sentiment: str = None, sentiment_conf: float = 0.0) -> Tuple[float, str]:
        ensemble_score = (mistral_score * 0.4) + (fingpt_score * 0.4) + (heuristic_score * 0.2)
        reason = f"ensemble: mistral_{mistral_score:.3f}_fingpt_{fingpt_score:.3f}_heuristic_{heuristic_score:.3f}"
        if sentiment:
            reason += f"_sentiment_{sentiment}_{sentiment_conf:.3f}"
        return ensemble_score, reason

    def process(self, text: str, context: Dict[str, Any] = None) -> ModerationResult:
        context = context or {}
        sentiment = None
        sentiment_conf = 0.0
        if context and "stage3a" in context:
            metadata = context["stage3a"].metadata
            if metadata:
                sentiment = metadata.get("sentiment")
                sentiment_conf = metadata.get("sentiment_confidence", 0.0)
        
        # Get scores from external models with fallback logic
        mistral_score, mistral_label = self._mistral_fraud_detection(text)
        fingpt_score, fingpt_label = self._fingpt_fraud_detection(text)
        heuristic_score, heuristic_pattern = self._heuristic_fraud_detection(text)
        
        # Check if external models are available
        mistral_available = not any(error in mistral_label for error in ["mistral_api_error", "mistral_api_empty", "mistral_api_404", "mistral_api_http_error", "mistral_api_timeout"])
        fingpt_available = not any(error in fingpt_label for error in ["fingpt_api_error", "fingpt_api_empty", "fingpt_api_404", "fingpt_api_http_error", "fingpt_api_timeout"])
        
        # Log availability status
        if not mistral_available:
            logger.warning("Mistral model unavailable, skipping external fraud detection")
        if not fingpt_available:
            logger.warning("FinGPT model unavailable, skipping external fraud detection")
        
        # If both external models are unavailable, use only heuristic
        if not mistral_available and not fingpt_available:
            logger.info("Both external models unavailable, using heuristic-only fraud detection")
            if heuristic_score >= 0.9:
                return ModerationResult(
                    decision="BLOCK",
                    stage=self.stage_name,
                    reason=f"fraud_model: heuristic_only_{heuristic_pattern}",
                    confidence=heuristic_score,
                    threat_level="high"
                )
            elif heuristic_score <= 0.2:
                return ModerationResult(
                    decision="ACCEPT",
                    stage=self.stage_name,
                    reason="fraud_model: heuristic_only_clean",
                    confidence=1.0 - heuristic_score,
                    threat_level="low"
                )
            else:
                return ModerationResult(
                    decision="ESCALATE",
                    stage=self.stage_name,
                    reason=f"fraud_model: heuristic_only_uncertain_{heuristic_pattern}",
                    confidence=heuristic_score,
                    threat_level="medium"
                )
        
        # If only one external model is available, adjust weights
        if mistral_available and not fingpt_available:
            # Use Mistral + Heuristic only
            ensemble_score = (mistral_score * 0.6) + (heuristic_score * 0.4)
            reason = f"ensemble: mistral_{mistral_score:.3f}_heuristic_{heuristic_score:.3f}_fingpt_unavailable"
        elif fingpt_available and not mistral_available:
            # Use FinGPT + Heuristic only
            ensemble_score = (fingpt_score * 0.6) + (heuristic_score * 0.4)
            reason = f"ensemble: fingpt_{fingpt_score:.3f}_heuristic_{heuristic_score:.3f}_mistral_unavailable"
        else:
            # Both available, use normal ensemble
            ensemble_score, reason = self._ensemble_voting(
                mistral_score, fingpt_score, heuristic_score, sentiment, sentiment_conf
            )
        
        if sentiment:
            reason += f"_sentiment_{sentiment}_{sentiment_conf:.3f}"
        
        if ensemble_score >= 0.9:
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



# Stage LLM Escalation
class LLMEscalation(ModerationStage):
    def __init__(self, groq_api_key: str = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "your_groq_api_key_here")

    @property
    def stage_name(self):
        return "llm"

    def _call_llm_with_cot(self, text: str, context: Dict[str, Any] = None, llm_suggestion: str = None) -> Tuple[str, float]:
        context_info = ""
        if context:
            for stage_name, result in context.items():
                context_info += f"\n{stage_name}: {result.reason}"
        suggestion_note = ""
        if llm_suggestion and text.strip().lower() == llm_suggestion.strip().lower():
            suggestion_note = "\nNote: The user is submitting a post that matches the LLM's previous suggestion. Only block if there are additional clear signs of fraud or manipulation."
        prompt = f"""
You are a financial content moderator (financial police) for a social media platform. Your job is to protect users from scams, fraud, and misleading financial posts, but also to allow normal, safe financial discussion.

Post: "{text}"

Previous stage results:{context_info}
{suggestion_note}

Instructions:
1. Only block posts with clear signs of fraud, deception, manipulation, or illegal activity (e.g., guaranteed returns, urgency, impersonation, fake endorsements, phishing, selling illegal goods, etc).
2. Do NOT block posts just for mentioning finance, investment, trading, or similar generic terms.
3. If the post is a simple, harmless financial question or statement, allow it.
4. If the post matches your own previous suggestion, only block if there are additional scammy elements.
5. Think step-by-step, but respond ONLY with "FRAUD" or "CLEAN" (no explanation).
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
                "max_tokens": 200,
                "temperature": 0.1,
            }
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            resp_json = response.json()
            if "choices" in resp_json and resp_json["choices"]:
                content = resp_json["choices"][0]["message"]["content"].strip().upper()
                if "FRAUD" in content:
                    return "FRAUD", 0.9
                elif "CLEAN" in content:
                    return "CLEAN", 0.9
                else:
                    return "UNCERTAIN", 0.5
            else:
                logger.warning("Empty response from Groq LLM API")
                return "ERROR", 0.5
        except requests.exceptions.HTTPError as e:
            logger.error(f"LLM escalation HTTP error: {e}")
            return "ERROR", 0.5
        except requests.exceptions.Timeout:
            logger.warning("LLM escalation timeout")
            return "ERROR", 0.5
        except Exception as e:
            logger.error(f"LLM escalation error: {e}")
            return "ERROR", 0.5

    def process(self, text: str, context: Dict[str, Any] = None) -> ModerationResult:
        context = context or {}
        llm_suggestion = None
        if context and "llm_suggestion" in context:
            llm_suggestion = context["llm_suggestion"]
        
        decision, confidence = self._call_llm_with_cot(text, context, llm_suggestion)
        
        # If LLM API is unavailable, use fallback logic
        if decision == "ERROR":
            logger.warning("LLM escalation unavailable, using fallback logic")
            # Check if any previous stage flagged the content
            if context:
                for stage_name, result in context.items():
                    if result.decision == "ESCALATE":
                        # If content was escalated by previous stage, block it
                        return ModerationResult(
                            decision="BLOCK",
                            stage=self.stage_name,
                            reason="llm: fallback_escalated_content",
                            confidence=0.7,
                            threat_level="medium"
                        )
            # If no previous escalation, accept the content
            return ModerationResult(
                decision="ACCEPT",
                stage=self.stage_name,
                reason="llm: fallback_no_escalation",
                confidence=0.6,
                threat_level="low"
            )
        
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
                decision="BLOCK",
                stage=self.stage_name,
                reason="llm: uncertain_cot",
                confidence=confidence,
                threat_level="medium"
            )

# Pipeline builder

def create_default_pipeline(keywords_file: str = "data/external/words.json", groq_api_key: str = None) -> 'ModerationPipeline':
    pipeline = ModerationPipeline()

    # Always resolve paths from THIS FILE location
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "lgbm_moderation.txt"
    support_path = base_dir / "lgbm_support.pkl"

    pipeline.register_stage(Stage1LexicalFilter(keywords_file))
    pipeline.register_stage(StageLGBMModeration(model_path=model_path, support_path=support_path))
    pipeline.register_stage(Stage2ToxicityCheck(toxicity_threshold=0.5))
    pipeline.register_stage(Stage3aSentimentSentinels())
    pipeline.register_stage(Stage3bFraudClassifier())
    pipeline.register_stage(LLMEscalation(groq_api_key))
    return pipeline

class ModerationPipeline:
    def __init__(self):
        self.stages: List[ModerationStage] = []
        self.stage_results: Dict[str, ModerationResult] = {}

    def register_stage(self, stage: ModerationStage):
        self.stages.append(stage)
        logger.info(f"Registered stage: {stage.stage_name}")

    def process(self, text: str) -> ModerationResult:
        context = {}
        for stage in self.stages:
            result = stage.process(text, context)
            self.stage_results[stage.stage_name] = result
            if result.decision == "BLOCK":
                logger.info(f"Content blocked at {stage.stage_name}: {result.reason}")
                return result
            context[stage.stage_name] = result
        for stage in reversed(self.stages):
            if stage.stage_name in self.stage_results:
                return self.stage_results[stage.stage_name]
        return ModerationResult(
            decision="ACCEPT",
            stage="pipeline",
            reason="all_stages_passed",
            confidence=1.0,
            threat_level="low"
        )

    def get_stage_results(self) -> Dict[str, ModerationResult]:
        return self.stage_results.copy()

# Legacy wrapper
class GuardianModerationEngine:
    def __init__(self, keywords_file: str = "data/external/words.json"):
        self.pipeline = create_default_pipeline(keywords_file)

    def moderate_content(self, content: str) -> ModerationResult:
        return self.pipeline.process(content)

    def get_model_status(self) -> Dict:
        return {
            "pipeline_stages": len(self.pipeline.stages),
            "stage_names": [stage.stage_name for stage in self.pipeline.stages]
        }

# Stage Quant: Quantitative Analysis & Risk Scoring
class QuantModerationStage(ModerationStage):
    def __init__(self, risk_threshold: float = 0.7, confidence_level: float = 0.95):
        self.risk_threshold = risk_threshold
        self.confidence_level = confidence_level
        self.scaler = StandardScaler()
        self._initialize_quant_models()

    @property
    def stage_name(self):
        return "quant"

    def _initialize_quant_models(self):
        """Initialize quantitative analysis models and baseline data"""
        # Baseline statistics for normal content (could be loaded from historical data)
        self.baseline_stats = {
            'avg_length': 50.0,
            'std_length': 30.0,
            'avg_punct_ratio': 0.15,
            'std_punct_ratio': 0.08,
            'avg_word_count': 10.0,
            'std_word_count': 5.0
        }

    def _calculate_risk_score(self, text: str, context: Dict[str, Any] = None) -> float:
        """Calculate comprehensive risk score based on multiple factors"""
        risk_factors = []
        
        # Text-based risk factors
        text_length = len(text)
        punct_count = sum(1 for c in text if c in '!?.,;:')
        word_count = len(text.split())
        
        # Normalize text characteristics
        length_z_score = (text_length - self.baseline_stats['avg_length']) / self.baseline_stats['std_length']
        punct_ratio = punct_count / max(text_length, 1)
        punct_z_score = (punct_ratio - self.baseline_stats['avg_punct_ratio']) / self.baseline_stats['std_punct_ratio']
        
        # Add risk factors
        risk_factors.append(abs(length_z_score) * 0.2)  # Unusual length
        risk_factors.append(abs(punct_z_score) * 0.3)   # Unusual punctuation
        risk_factors.append(1.0 if text.isupper() else 0.0)  # All caps
        risk_factors.append(1.0 if any(word in text.lower() for word in ['urgent', 'now', 'limited', 'act']) else 0.0)  # Urgency words
        
        # Context-based risk factors
        if context:
            for stage_name, result in context.items():
                if result.decision == "ESCALATE":
                    risk_factors.append(0.5)  # Previous escalation
                elif result.decision == "BLOCK":
                    risk_factors.append(0.8)  # Previous block
                elif result.confidence < 0.5:
                    risk_factors.append(0.3)  # Low confidence
        
        # Calculate final risk score (0-1)
        risk_score = min(1.0, sum(risk_factors) / len(risk_factors)) if risk_factors else 0.0
        return risk_score

    def _calculate_confidence_interval(self, confidence: float, sample_size: int = 100) -> Tuple[float, float]:
        """Calculate confidence interval for the confidence score"""
        if sample_size <= 1:
            return (confidence, confidence)
        
        # Standard error approximation
        se = np.sqrt(confidence * (1 - confidence) / sample_size)
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        margin_of_error = z_score * se
        
        lower = max(0.0, confidence - margin_of_error)
        upper = min(1.0, confidence + margin_of_error)
        
        return (lower, upper)

    def _calculate_statistical_significance(self, text: str, context: Dict[str, Any] = None) -> float:
        """Calculate statistical significance of the moderation decision"""
        # This is a simplified version - in practice, you'd compare against historical data
        if not context:
            return 0.5
        
        # Count how many stages made the same decision
        decisions = [result.decision for result in context.values()]
        if not decisions:
            return 0.5
        
        # Calculate agreement ratio
        most_common_decision = max(set(decisions), key=decisions.count)
        agreement_ratio = decisions.count(most_common_decision) / len(decisions)
        
        # Convert to significance score (0-1)
        significance = agreement_ratio * 0.8 + 0.2  # Base significance of 0.2
        return min(1.0, significance)

    def _calculate_anomaly_score(self, text: str) -> float:
        """Calculate anomaly score based on text characteristics"""
        # Extract features
        text_length = len(text)
        punct_count = sum(1 for c in text if c in '!?.,;:')
        word_count = len(text.split())
        unique_words = len(set(text.lower().split()))
        
        # Calculate z-scores
        length_z = abs((text_length - self.baseline_stats['avg_length']) / self.baseline_stats['std_length'])
        punct_ratio = punct_count / max(text_length, 1)
        punct_z = abs((punct_ratio - self.baseline_stats['avg_punct_ratio']) / self.baseline_stats['std_punct_ratio'])
        
        # Anomaly indicators
        all_caps = 1.0 if text.isupper() else 0.0
        excessive_punct = 1.0 if punct_ratio > 0.3 else 0.0
        very_short = 1.0 if text_length < 10 else 0.0
        very_long = 1.0 if text_length > 200 else 0.0
        
        # Calculate anomaly score
        anomaly_factors = [length_z * 0.3, punct_z * 0.3, all_caps * 0.2, 
                          excessive_punct * 0.2, very_short * 0.1, very_long * 0.1]
        anomaly_score = min(1.0, sum(anomaly_factors))
        
        return anomaly_score

    def _calculate_quant_metrics(self, text: str, context: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate comprehensive quantitative metrics"""
        metrics = {}
        
        # Text complexity metrics
        metrics['text_length'] = len(text)
        metrics['word_count'] = len(text.split())
        metrics['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        metrics['punct_ratio'] = sum(1 for c in text if c in '!?.,;:') / max(len(text), 1)
        metrics['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Context-based metrics
        if context:
            metrics['stage_agreement'] = len(set(result.decision for result in context.values())) / max(len(context), 1)
            metrics['avg_confidence'] = np.mean([result.confidence for result in context.values()]) if context else 0.5
            metrics['escalation_count'] = sum(1 for result in context.values() if result.decision == "ESCALATE")
            metrics['block_count'] = sum(1 for result in context.values() if result.decision == "BLOCK")
        
        return metrics

    def process(self, text: str, context: Dict[str, Any] = None) -> ModerationResult:
        """Process text through quantitative analysis"""
        context = context or {}
        
        # Calculate quantitative metrics
        risk_score = self._calculate_risk_score(text, context)
        confidence_interval = self._calculate_confidence_interval(0.7, 100)  # Example confidence
        statistical_significance = self._calculate_statistical_significance(text, context)
        anomaly_score = self._calculate_anomaly_score(text)
        quant_metrics = self._calculate_quant_metrics(text, context)
        
        # Determine decision based on risk score and context
        if risk_score > self.risk_threshold:
            decision = "BLOCK"
            threat_level = "high"
            confidence = risk_score
            reason = f"quant_risk_score_{risk_score:.2f}"
        elif risk_score > self.risk_threshold * 0.7:
            decision = "ESCALATE"
            threat_level = "medium"
            confidence = risk_score
            reason = f"quant_elevated_risk_{risk_score:.2f}"
        else:
            decision = "ACCEPT"
            threat_level = "low"
            confidence = 1.0 - risk_score
            reason = f"quant_low_risk_{risk_score:.2f}"
        
        return ModerationResult(
            decision=decision,
            stage=self.stage_name,
            reason=reason,
            confidence=confidence,
            threat_level=threat_level,
            risk_score=risk_score,
            confidence_interval=confidence_interval,
            statistical_significance=statistical_significance,
            anomaly_score=anomaly_score,
            quant_metrics=quant_metrics,
            metadata={
                "quant_analysis": True,
                "risk_threshold": self.risk_threshold,
                "confidence_level": self.confidence_level
            }
        )

# Updated pipeline builder with quant mode
def create_quant_pipeline(keywords_file: str = "data/external/words.json", groq_api_key: str = None) -> 'ModerationPipeline':
    """Create pipeline with quantitative analysis stage"""
    pipeline = ModerationPipeline()

    # Always resolve paths from THIS FILE location
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "lgbm_moderation.txt"
    support_path = base_dir / "lgbm_support.pkl"

    pipeline.register_stage(Stage1LexicalFilter(keywords_file))
    pipeline.register_stage(StageLGBMModeration(model_path=model_path, support_path=support_path))
    pipeline.register_stage(Stage2ToxicityCheck(toxicity_threshold=0.5))
    pipeline.register_stage(Stage3aSentimentSentinels())
    pipeline.register_stage(Stage3bFraudClassifier())
    pipeline.register_stage(LLMEscalation(groq_api_key))
    pipeline.register_stage(QuantModerationStage())  # Add quant stage
    return pipeline

# Updated legacy wrapper with quant mode
class GuardianModerationEngine:
    def __init__(self, keywords_file: str = "data/external/words.json", quant_mode: bool = False):
        if quant_mode:
            self.pipeline = create_quant_pipeline(keywords_file)
        else:
            self.pipeline = create_default_pipeline(keywords_file)

    def moderate_content(self, content: str) -> ModerationResult:
        return self.pipeline.process(content)

    def get_model_status(self) -> Dict:
        return {
            "pipeline_stages": len(self.pipeline.stages),
            "stage_names": [stage.stage_name for stage in self.pipeline.stages],
            "quant_mode": any(stage.stage_name == "quant" for stage in self.pipeline.stages)
        } 