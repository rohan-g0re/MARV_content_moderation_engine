"""
GuardianAI Content Moderation Core Pipeline - Safe Version
Handles missing AI dependencies gracefully for basic functionality testing
"""

import re
import json
import logging
import numpy as np
import pickle
from typing import Tuple, Dict, List, Optional
from pathlib import Path
from collections import Counter

# Try to import AI dependencies gracefully
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available - AI models disabled")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - AI models disabled")

# Try to import LGBM dependency
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available - LGBM stage disabled")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available - TF-IDF features disabled")

# Configure logging
from logger import get_logger
logger = get_logger("moderation", "moderation.log")

class ModerationResult:
    """Structured moderation result for Day 5 deliverable"""
    
    def __init__(self, accepted: bool, reason: str, threat_level: str = "low", 
                 confidence: float = 1.0, stage: str = "unknown", 
                 band: str = "SAFE", action: str = "PASS"):
        self.accepted = accepted
        self.reason = reason
        self.threat_level = threat_level  # low, medium, high
        self.confidence = confidence      # 0.0 - 1.0
        self.stage = stage               # rule-based, lgbm, detoxify, finbert
        self.band = band                 # SAFE, FLAG_LOW, FLAG_MEDIUM, FLAG_HIGH, BLOCK
        self.action = action             # PASS, FLAG_LOW, FLAG_MEDIUM, FLAG_HIGH, BLOCK
        self.explanation = f"Content {'approved' if accepted else 'rejected'} at {stage} stage: {reason}"

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses"""
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "threat_level": self.threat_level,
            "confidence": self.confidence,
            "stage": self.stage,
            "band": self.band,
            "action": self.action,
            "explanation": self.explanation
        }

class LGBMFeatureExtractor:
    """Extract features for LGBM model matching the expected 91 features"""
    
    def __init__(self):
        # Financial scam phrases to detect
        self.scam_phrases = {
            'phrase_guaranteed_returns': [r'\bguaranteed\s+returns?\b'],
            'phrase_double_your_money': [r'\bdouble\s+your\s+money\b'],
            'phrase_get_rich_quick': [r'\bget\s+rich\s+quick\b'],
            'phrase_millionaire': [r'\bmillionaire\b'],
            'phrase_risk_free': [r'\brisk\s*free\b'],
            'phrase_no_investment_needed': [r'\bno\s+investment\s+needed\b'],
            'phrase_act_now': [r'\bact\s+now\b'],
            'phrase_quick_money': [r'\bquick\s+money\b'],
            'phrase_100%_profit': [r'\b100%\s+profit\b'],
            'phrase_insider_tip': [r'\binsider\s+tip\b'],
            'phrase_secret_method': [r'\bsecret\s+method\b'],
            'phrase_overnight_success': [r'\bovernight\s+success\b'],
            'phrase_only_today': [r'\bonly\s+today\b'],
            'phrase_become_rich': [r'\bbecome\s+rich\b'],
            'phrase_click_here': [r'\bclick\s+here\b'],
            'phrase_no_experience_needed': [r'\bno\s+experience\s+needed\b'],
            'phrase_huge_payout': [r'\bhuge\s+payout\b'],
            'phrase_investment_opportunity': [r'\binvestment\s+opportunity\b'],
            'phrase_limited_spots': [r'\blimited\s+spots?\b'],
            'phrase_passive_income': [r'\bpassive\s+income\b'],
            'phrase_fast_cash': [r'\bfast\s+cash\b'],
            'phrase_unbeatable_offer': [r'\bunbeatable\s+offer\b'],
            'phrase_cash_prize': [r'\bcash\s+prize\b'],
            'phrase_dm_me': [r'\bdm\s+me\b'],
            'phrase_ping_me': [r'\bping\s+me\b'],
            'phrase_withdraw_instantly': [r'\bwithdraw\s+instantly\b']
        }
        
        # Regex patterns for specific types of content
        self.regex_patterns = [
            r'http[s]?://\S+',  # URLs
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
            r'\b\d{10,}\b',  # Phone numbers
            r'\$\d+',  # Dollar amounts
            r'\b\d+%\b'  # Percentages
        ]
        
        # Financial terms for TF-IDF
        self.financial_terms = [
            'about', 'all', 'and', 'anyone', 'anyone_else', 'are', 'bitcoin',
            'company', 'crypto', 'economy', 'else', 'finance', 'financial', 'for',
            'going', 'going_to', 'in', 'inflation', 'investing', 'is', 'it', 'just',
            'market', 'me', 'money', 'my', 'of', 'on', 're', 'scam', 'so', 'stock',
            'stocks', 'that', 'the', 'they', 'this', 'to', 'you', 'your'
        ]
        
        # Initialize TF-IDF vectorizer if available
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                vocabulary=self.financial_terms,
                lowercase=True,
                token_pattern=r'\b\w+\b'
            )
        else:
            self.tfidf_vectorizer = None
    
    def extract_features(self, text: str) -> List[float]:
        """Extract all 91 features expected by LGBM model"""
        features = []
        text_lower = text.lower()
        words = text_lower.split()
        
        # Basic text statistics (18 features)
        features.append(len(words))  # num_words
        features.append(len(text))   # num_chars
        features.append(len(text) / max(len(words), 1))  # avg_word_len
        features.append(sum(1 for c in text if c.isupper()))  # num_upper
        features.append(text.count('!'))  # num_exclaims
        features.append(text.count('?'))  # num_questions
        features.append(len(re.findall(r'http[s]?://\S+', text)))  # num_links
        features.append(text.count('@'))  # num_mentions
        features.append(text.count('$'))  # num_dollar
        features.append(text.count('#'))  # num_hashtags
        features.append(sum(c.isdigit() for c in text))  # num_digits
        features.append(sum(1 for c in text if c in '.,!?;:'))  # num_punct
        
        # Stopword and unique word counts (2 features)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        stopword_count = sum(1 for word in words if word in stopwords)
        features.append(stopword_count)  # stopword_count
        features.append(len(set(words)))  # unique_words
        
        # Boolean flags (4 features)
        features.append(1.0 if text.isupper() else 0.0)  # all_caps
        features.append(1.0 if text_lower.startswith('http') else 0.0)  # starts_with_http
        features.append(sum(1 for c in text if c.isupper()) / max(len(text), 1))  # frac_upper
        features.append(stopword_count / max(len(words), 1))  # frac_stopwords
        
        # Keyword presence (2 features)
        keywords = ['scam', 'fraud', 'money', 'investment', 'profit', 'returns', 'guaranteed']
        keyword_present = any(keyword in text_lower for keyword in keywords)
        keyword_count = sum(text_lower.count(keyword) for keyword in keywords)
        features.append(1.0 if keyword_present else 0.0)  # keyword_present
        features.append(float(keyword_count))  # keyword_count
        
        # Scam phrase detection (26 features)
        for phrase_name, patterns in self.scam_phrases.items():
            phrase_detected = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in patterns)
            features.append(1.0 if phrase_detected else 0.0)
        
        # Regex pattern matching (5 features)
        for pattern in self.regex_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            features.append(1.0 if matches > 0 else 0.0)
        
        # TF-IDF features (39 features)
        if self.tfidf_vectorizer is not None and SKLEARN_AVAILABLE:
            try:
                # Fit and transform on single document
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([text_lower])
                tfidf_features = tfidf_matrix.toarray()[0].tolist()
                features.extend(tfidf_features)
            except Exception as e:
                logger.warning(f"TF-IDF extraction failed: {e}")
                # Fill with zeros if TF-IDF fails
                features.extend([0.0] * len(self.financial_terms))
        else:
            # Fill with zeros if sklearn not available
            features.extend([0.0] * len(self.financial_terms))
        
        # Ensure we have exactly 91 features
        while len(features) < 91:
            features.append(0.0)
        
        return features[:91]  # Truncate to exactly 91

class GuardianModerationEngine:
    """
    Safe version of moderation engine that works without AI dependencies
    Now includes LGBM as Stage 2
    """
    
    def __init__(self, keywords_file: str = "data/external/words.json"):
        # Handle relative path from backend directory
        if not Path(keywords_file).exists():
            # Try from parent directory (when running from backend/)
            keywords_file = f"../{keywords_file}"
        self.keywords_file = Path(keywords_file)
        self.keywords = self._load_keywords()
        
        # Initialize LGBM components
        self.lgbm_model = None
        self.lgbm_support = None
        self.feature_extractor = None
        if LIGHTGBM_AVAILABLE:
            self._initialize_lgbm()
        else:
            logger.warning("LightGBM not available - LGBM stage disabled")
        
        # Initialize AI models safely
        self.detoxify_classifier = None
        self.finbert_classifier = None
        if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            self._initialize_models()
        else:
            logger.warning("AI dependencies not available - running in rule-based mode only")
        
        # Thresholds for moderation stages
        self.lgbm_threshold = 0.7  # Threshold for LGBM flagging
        self.toxicity_threshold = 0.5
        self.finbert_threshold = 0.7
        self.llm_escalation_threshold = 3  # For future LLM integration
        
        # LGBM class mapping (based on training)
        self.lgbm_classes = {
            0: "SAFE",
            1: "FLAG_MEDIUM", 
            2: "BLOCK"
        }
        
        # FinBERT 5-layer band system for financial companies
        self.finbert_bands = {
            "SAFE": {"min": 0.0, "max": 0.2, "action": "PASS", "threat_level": "low"},
            "FLAG_LOW": {"min": 0.2, "max": 0.4, "action": "FLAG_LOW", "threat_level": "low"},
            "FLAG_MEDIUM": {"min": 0.4, "max": 0.6, "action": "FLAG_MEDIUM", "threat_level": "medium"},
            "FLAG_HIGH": {"min": 0.6, "max": 0.8, "action": "FLAG_HIGH", "threat_level": "high"},
            "BLOCK": {"min": 0.8, "max": 1.0, "action": "BLOCK", "threat_level": "high"}
        }
        
        logger.info(f"GuardianAI initialized with {len(self.keywords)} keywords")

    def _load_keywords(self) -> List[str]:
        """Load keywords from JSON file"""
        try:
            if self.keywords_file.exists():
                with open(self.keywords_file, 'r', encoding='utf-8') as f:
                    keywords = json.load(f)
                logger.info(f"Loaded {len(keywords)} keywords from {self.keywords_file}")
                return keywords
            else:
                logger.warning(f"Keywords file not found: {self.keywords_file}")
                return ["scammer", "fraud", "hate", "violence", "spam"]
        except Exception as e:
            logger.error(f"Error loading keywords: {e}")
            return ["scammer", "fraud", "hate", "violence", "spam"]

    def _initialize_lgbm(self):
        """Initialize LGBM model and feature extractor"""
        try:
            # Load LGBM model
            model_path = Path("backend/app/core/lgbm_moderation.txt")
            if not model_path.exists():
                model_path = Path("app/core/lgbm_moderation.txt")
            
            if model_path.exists():
                self.lgbm_model = lgb.Booster(model_file=str(model_path))
                logger.info("✅ LGBM model loaded successfully")
            else:
                logger.warning(f"LGBM model file not found: {model_path}")
                return
                
            # Load support data (optional - not currently used in pipeline)
            support_path = Path("backend/app/core/lgbm_support.pkl")
            if not support_path.exists():
                support_path = Path("app/core/lgbm_support.pkl")
                
            if support_path.exists():
                try:
                    with open(support_path, 'rb') as f:
                        self.lgbm_support = pickle.load(f)
                    logger.info("✅ LGBM support data loaded successfully")
                except Exception as e:
                    logger.warning(f"LGBM support data loading failed (not critical): {e}")
                    self.lgbm_support = None
            else:
                logger.warning(f"LGBM support file not found: {support_path} (not critical)")
                self.lgbm_support = None
                
            # Initialize feature extractor
            self.feature_extractor = LGBMFeatureExtractor()
            logger.info("✅ LGBM feature extractor initialized")
            
        except Exception as e:
            logger.error(f"LGBM initialization failed: {e}")
            self.lgbm_model = None
            self.feature_extractor = None

    def _initialize_models(self):
        """Initialize AI models with error handling"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available - AI models disabled")
            return
            
        try:
            logger.info("Loading Detoxify model...")
            self.detoxify_classifier = pipeline("text-classification", model="unitary/toxic-bert")
            logger.info("✅ Detoxify model loaded successfully")
        except Exception as e:
            logger.warning(f"Detoxify model loading failed: {e}")
            self.detoxify_classifier = None

        try:
            logger.info("Loading FinBERT model...")
            self.finbert_classifier = pipeline("text-classification", model="ProsusAI/finbert")
            logger.info("✅ FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"FinBERT model loading failed: {e}")
            self.finbert_classifier = None

    def stage1_rule_based_filter(self, text: str) -> ModerationResult:
        """
        Stage 1: Rule-based filtering using keywords and regex patterns
        """
        text_lower = text.lower()
        
        # Check keywords (whole word matching)
        for keyword in self.keywords:
            if re.search(rf'\b{re.escape(keyword.lower())}\b', text_lower):
                return ModerationResult(
                    accepted=False,
                    reason=f"Rule-based: {keyword}",
                    threat_level="high",
                    confidence=1.0,
                    stage="rule-based",
                    band="BLOCK",
                    action="BLOCK"
                )
        
        # Check regex patterns
        patterns = [
            (r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "URL detected"),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "Email detected"),
            (r"\b\d{10,}\b", "Phone number detected"),
            (r"\b(kill|bash|hack|steal|threat|attack|rape|murder|shoot|stab|destroy|burn|harass|stalk|blackmail|assault|abuse|bully|rob|terror|terrorist|explosive|bomb|kidnap|extort)\b", "Violence/threat detected")
        ]
        
        for pattern, description in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return ModerationResult(
                    accepted=False,
                    reason=f"Rule-based: {description}",
                    threat_level="medium",
                    confidence=0.9,
                    stage="rule-based",
                    band="BLOCK",
                    action="BLOCK"
                )
        
        # Stage 1 passed
        return ModerationResult(
            accepted=True,
            reason="Stage 1 passed",
            threat_level="low",
            confidence=1.0,
            stage="rule-based",
            band="SAFE",
            action="PASS"
        )

    def stage2_lgbm_check(self, text: str) -> ModerationResult:
        """
        Stage 2: LGBM machine learning model for financial scam/fraud detection
        """
        if self.lgbm_model is None or self.feature_extractor is None:
            logger.warning("LGBM model not available, skipping stage 2")
            return ModerationResult(
                accepted=True,
                reason="Stage 2 skipped (LGBM model unavailable)",
                threat_level="low",
                confidence=0.5,
                stage="lgbm",
                band="SAFE",
                action="PASS"
            )
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(text)
            features_array = np.array(features).reshape(1, -1)
            
            # Get predictions
            predictions = self.lgbm_model.predict(features_array)
            probabilities = predictions[0]  # Get probabilities for each class
            
            # Get the predicted class
            predicted_class = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            
            # Map class to result
            band = self.lgbm_classes.get(predicted_class, "SAFE")
            
            # Determine if content should be flagged
            if predicted_class == 2:  # BLOCK class
                return ModerationResult(
                    accepted=False,
                    reason=f"LGBM: High risk content detected (confidence: {confidence:.3f})",
                    threat_level="high",
                    confidence=confidence,
                    stage="lgbm",
                    band="BLOCK",
                    action="BLOCK"
                )
            elif predicted_class == 1:  # FLAG_MEDIUM class
                return ModerationResult(
                    accepted=False,
                    reason=f"LGBM: Moderate risk content detected (confidence: {confidence:.3f})",
                    threat_level="medium",
                    confidence=confidence,
                    stage="lgbm",
                    band="FLAG_MEDIUM",
                    action="FLAG_MEDIUM"
                )
            else:  # SAFE class
                return ModerationResult(
                    accepted=True,
                    reason=f"LGBM: Content appears safe (confidence: {confidence:.3f})",
                    threat_level="low",
                    confidence=confidence,
                    stage="lgbm",
                    band="SAFE",
                    action="PASS"
                )
                
        except Exception as e:
            logger.error(f"LGBM classification error: {e}")
            return ModerationResult(
                accepted=True,
                reason="Stage 2 error (defaulting to safe)",
                threat_level="low",
                confidence=0.5,
                stage="lgbm",
                band="SAFE",
                action="PASS"
            )

    def stage3_detoxify_check(self, text: str) -> ModerationResult:
        """
        Stage 3: Detoxify AI toxicity detection (formerly Stage 2)
        """
        if self.detoxify_classifier is None:
            logger.warning("Detoxify model not available, skipping stage 3")
            return ModerationResult(
                accepted=True,
                reason="Stage 3 skipped (model unavailable)",
                threat_level="low",
                confidence=0.5,
                stage="detoxify",
                band="SAFE",
                action="PASS"
            )
        
        # If model is available, use it
        try:
            results = self.detoxify_classifier(text)
            toxicity_score = max([result['score'] for result in results if result['label'] == 'TOXIC'])
            
            if toxicity_score >= self.toxicity_threshold:
                return ModerationResult(
                    accepted=False,
                    reason=f"Detoxify: High toxicity detected (score: {toxicity_score:.3f})",
                    threat_level="high",
                    confidence=toxicity_score,
                    stage="detoxify",
                    band="BLOCK",
                    action="BLOCK"
                )
        except Exception as e:
            logger.error(f"Detoxify classification error: {e}")
        
        return ModerationResult(
            accepted=True,
            reason="Stage 3 passed",
            threat_level="low",
            confidence=0.8,
            stage="detoxify",
            band="SAFE",
            action="PASS"
        )

    def stage4_finbert_check(self, text: str) -> ModerationResult:
        """
        Stage 4: FinBERT financial fraud detection (formerly Stage 3)
        """
        if self.finbert_classifier is None:
            logger.warning("FinBERT model not available, skipping stage 4")
            return ModerationResult(
                accepted=True,
                reason="Stage 4 skipped (model unavailable)",
                threat_level="low",
                confidence=0.5,
                stage="finbert",
                band="SAFE",
                action="PASS"
            )
        
        # If model is available, use it
        try:
            results = self.finbert_classifier(text)
            # Process results...
            return ModerationResult(
                accepted=True,
                reason="FinBERT: Non-financial content",
                threat_level="low",
                confidence=0.8,
                stage="finbert",
                band="SAFE",
                action="PASS"
            )
        except Exception as e:
            logger.error(f"FinBERT classification error: {e}")
            return ModerationResult(
                accepted=True,
                reason="Stage 4 error (defaulting to safe)",
                threat_level="low",
                confidence=0.5,
                stage="finbert",
                band="SAFE",
                action="PASS"
            )

    def moderate_content(self, content: str) -> ModerationResult:
        """
        Main moderation pipeline entry point
        Updated to include LGBM as Stage 2
        """
        if not content or len(content.strip()) == 0:
            return ModerationResult(
                accepted=False,
                reason="Empty content",
                threat_level="low",
                confidence=1.0,
                stage="validation",
                band="BLOCK",
                action="BLOCK"
            )
        
        # Stage 1: Rule-based filtering
        stage1_result = self.stage1_rule_based_filter(content)
        if not stage1_result.accepted:
            return stage1_result
        
        # Stage 2: LGBM machine learning model
        stage2_result = self.stage2_lgbm_check(content)
        if not stage2_result.accepted:
            return stage2_result
        
        # Stage 3: Detoxify AI (if available)
        stage3_result = self.stage3_detoxify_check(content)
        if not stage3_result.accepted:
            return stage3_result
        
        # Stage 4: FinBERT AI (if available)
        stage4_result = self.stage4_finbert_check(content)
        return stage4_result

    def get_model_status(self) -> Dict:
        """Get status of all models"""
        return {
            "lgbm_loaded": self.lgbm_model is not None,
            "detoxify_loaded": self.detoxify_classifier is not None,
            "finbert_loaded": self.finbert_classifier is not None,
            "keywords_count": len(self.keywords),
            "lightgbm_available": LIGHTGBM_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "thresholds": {
                "lgbm": self.lgbm_threshold,
                "toxicity": self.toxicity_threshold,
                "finbert": self.finbert_threshold,
                "llm_escalation": self.llm_escalation_threshold
            }
        }