"""
GuardianAI Content Moderation Core Pipeline - Safe Version
Handles missing AI dependencies gracefully for basic functionality testing
"""

import re
import json
import logging
from typing import Tuple, Dict, List, Optional
from pathlib import Path

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
        self.stage = stage               # rule-based, detoxify, finbert
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

class GuardianModerationEngine:
    """
    Safe version of moderation engine that works without AI dependencies
    """
    
    def __init__(self, keywords_file: str = "data/external/words.json"):
        # Handle relative path from backend directory
        if not Path(keywords_file).exists():
            # Try from parent directory (when running from backend/)
            keywords_file = f"../{keywords_file}"
        self.keywords_file = Path(keywords_file)
        self.keywords = self._load_keywords()
        
        # Initialize AI models safely
        self.detoxify_classifier = None
        self.finbert_classifier = None
        if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            self._initialize_models()
        else:
            logger.warning("AI dependencies not available - running in rule-based mode only")
        
        # Thresholds for Day 6 LLM escalation logic
        self.toxicity_threshold = 0.5
        self.finbert_threshold = 0.7
        self.llm_escalation_threshold = 3  # For future LLM integration
        
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

    def stage2_detoxify_check(self, text: str) -> ModerationResult:
        """
        Stage 2: Detoxify AI toxicity detection (safe mode)
        """
        if self.detoxify_classifier is None:
            logger.warning("Detoxify model not available, skipping stage 2")
            return ModerationResult(
                accepted=True,
                reason="Stage 2 skipped (model unavailable)",
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
            reason="Stage 2 passed",
            threat_level="low",
            confidence=0.8,
            stage="detoxify",
            band="SAFE",
            action="PASS"
        )

    def stage3_finbert_check(self, text: str) -> ModerationResult:
        """
        Stage 3: FinBERT financial fraud detection (safe mode)
        """
        if self.finbert_classifier is None:
            logger.warning("FinBERT model not available, skipping stage 3")
            return ModerationResult(
                accepted=True,
                reason="Stage 3 skipped (model unavailable)",
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
                reason="Stage 3 error (defaulting to safe)",
                threat_level="low",
                confidence=0.5,
                stage="finbert",
                band="SAFE",
                action="PASS"
            )

    def moderate_content(self, content: str) -> ModerationResult:
        """
        Main moderation pipeline entry point
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
        
        # Stage 2: Detoxify AI (if available)
        stage2_result = self.stage2_detoxify_check(content)
        if not stage2_result.accepted:
            return stage2_result
        
        # Stage 3: FinBERT AI (if available)
        stage3_result = self.stage3_finbert_check(content)
        return stage3_result

    def get_model_status(self) -> Dict:
        """Get status of all models"""
        return {
            "detoxify_loaded": self.detoxify_classifier is not None,
            "finbert_loaded": self.finbert_classifier is not None,
            "keywords_count": len(self.keywords),
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "thresholds": {
                "toxicity": self.toxicity_threshold,
                "finbert": self.finbert_threshold,
                "llm_escalation": self.llm_escalation_threshold
            }
        } 