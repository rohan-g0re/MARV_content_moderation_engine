"""
GuardianAI Content Moderation Core Pipeline

This module contains the consolidated moderation pipeline with three-layer filtering:
1. Rule-based filtering (keywords + regex patterns)
2. Detoxify AI toxicity detection
3. FinBERT financial fraud detection

Designed for the upcoming deliverables:
- Day 5: GuardianAI Core Pipeline with moderate_content() entrypoint
- Day 6: LLM Escalation Logic integration
- Future: Dictionary expansion and feedback systems
"""

import re
import json
import logging
from typing import Tuple, Dict, List, Optional
from pathlib import Path
from transformers import pipeline
import torch

# Configure logging
logger = logging.getLogger(__name__)

class ModerationResult:
    """Structured moderation result for Day 5 deliverable"""
    
    def __init__(self, accepted: bool, reason: str, threat_level: str = "low", 
                 confidence: float = 1.0, stage: str = "unknown"):
        self.accepted = accepted
        self.reason = reason
        self.threat_level = threat_level  # low, medium, high
        self.confidence = confidence      # 0.0 - 1.0
        self.stage = stage               # rule-based, detoxify, finbert
        self.action = "accept" if accepted else "block"
        self.explanation = f"Content {'approved' if accepted else 'rejected'} at {stage} stage: {reason}"

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses"""
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "threat_level": self.threat_level,
            "confidence": self.confidence,
            "stage": self.stage,
            "action": self.action,
            "explanation": self.explanation
        }

class GuardianModerationEngine:
    """
    Consolidated moderation engine combining DatabaseFilter + moderation router functionality
    
    Architecture aligned with system diagrams:
    Stage 1: Rule-Based Filter (keywords.json + regex patterns)
    Stage 2: Detoxify AI (toxicity detection)
    Stage 3: FinBERT AI (financial fraud detection)
    """
    
    def __init__(self, keywords_file: str = "words.json"):
        self.keywords_file = Path(keywords_file)
        self.keywords = self._load_keywords()
        
        # Initialize AI models
        self.detoxify_classifier = None
        self.finbert_classifier = None
        self._initialize_models()
        
        # Thresholds for Day 6 LLM escalation logic
        self.toxicity_threshold = 0.5
        self.finbert_threshold = 0.7
        self.llm_escalation_threshold = 3  # For future LLM integration
        
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
        
        Returns ModerationResult with acceptance decision and threat level
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
                    stage="rule-based"
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
                    stage="rule-based"
                )
        
        # Stage 1 passed
        return ModerationResult(
            accepted=True,
            reason="Stage 1 passed",
            threat_level="low",
            confidence=1.0,
            stage="rule-based"
        )

    def stage2_detoxify_check(self, text: str) -> ModerationResult:
        """
        Stage 2: Detoxify AI toxicity detection
        
        Returns ModerationResult with toxicity assessment
        """
        if self.detoxify_classifier is None:
            logger.warning("Detoxify model not available, skipping stage 2")
            return ModerationResult(
                accepted=True,
                reason="Stage 2 skipped (model unavailable)",
                threat_level="low",
                confidence=0.5,
                stage="detoxify"
            )
        
        try:
            result = self.detoxify_classifier(text)
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            # Check if toxic content detected
            if label in ['toxic', 'label_1'] and score > self.toxicity_threshold:
                threat_level = "high" if score > 0.8 else "medium"
                return ModerationResult(
                    accepted=False,
                    reason=f"Toxic content detected (score: {score:.3f})",
                    threat_level=threat_level,
                    confidence=score,
                    stage="detoxify"
                )
            
            # Stage 2 passed
            return ModerationResult(
                accepted=True,
                reason="Stage 2 passed",
                threat_level="low",
                confidence=1.0 - score,
                stage="detoxify"
            )
            
        except Exception as e:
            logger.error(f"Detoxify error: {e}")
            return ModerationResult(
                accepted=True,
                reason="Stage 2 error (fallback to accept)",
                threat_level="low",
                confidence=0.5,
                stage="detoxify"
            )

    def stage3_finbert_check(self, text: str) -> ModerationResult:
        """
        Stage 3: FinBERT financial fraud detection
        
        Returns ModerationResult with financial risk assessment
        """
        if self.finbert_classifier is None:
            logger.warning("FinBERT model not available, skipping stage 3")
            return ModerationResult(
                accepted=True,
                reason="Stage 3 skipped (model unavailable)",
                threat_level="low",
                confidence=0.5,
                stage="finbert"
            )
        
        try:
            result = self.finbert_classifier(text)
            sentiment = result[0]['label']
            confidence = result[0]['score']
            
            # Check for potential financial fraud (negative sentiment with high confidence)
            if sentiment == 'negative' and confidence > self.finbert_threshold:
                threat_level = "high" if confidence > 0.9 else "medium"
                return ModerationResult(
                    accepted=False,
                    reason=f"Potential financial fraud detected (confidence: {confidence:.3f})",
                    threat_level=threat_level,
                    confidence=confidence,
                    stage="finbert"
                )
            
            # Stage 3 passed
            return ModerationResult(
                accepted=True,
                reason="All checks passed",
                threat_level="low",
                confidence=1.0 - confidence if sentiment == 'negative' else 1.0,
                stage="finbert"
            )
            
        except Exception as e:
            logger.error(f"FinBERT error: {e}")
            return ModerationResult(
                accepted=True,
                reason="Stage 3 error (fallback to accept)",
                threat_level="low",
                confidence=0.5,
                stage="finbert"
            )

    def moderate_content(self, content: str) -> ModerationResult:
        """
        Main moderation pipeline entrypoint for Day 5 deliverable
        
        Processes content through three sequential stages:
        1. Rule-based filtering
        2. Detoxify toxicity detection  
        3. FinBERT financial fraud detection
        
        Returns structured ModerationResult with threat level, action, and explanation
        """
        logger.info(f"Starting moderation for content: {content[:50]}...")
        
        # Stage 1: Rule-based filtering
        result = self.stage1_rule_based_filter(content)
        if not result.accepted:
            logger.info(f"Content rejected at stage 1: {result.reason}")
            return result
        
        # Stage 2: Detoxify check
        result = self.stage2_detoxify_check(content)
        if not result.accepted:
            logger.info(f"Content rejected at stage 2: {result.reason}")
            return result
        
        # Stage 3: FinBERT check
        result = self.stage3_finbert_check(content)
        logger.info(f"Moderation completed - Stage 3 result: {result.reason}")
        return result

    def reload_keywords(self):
        """Reload keywords from file - useful for Day 10 dictionary expansion"""
        self.keywords = self._load_keywords()
        logger.info(f"Reloaded {len(self.keywords)} keywords")

    def update_thresholds(self, toxicity_threshold: float = None, 
                         finbert_threshold: float = None,
                         llm_escalation_threshold: int = None):
        """Update moderation thresholds - for Day 6 LLM escalation logic"""
        if toxicity_threshold is not None:
            self.toxicity_threshold = toxicity_threshold
            logger.info(f"Updated toxicity threshold to {toxicity_threshold}")
        
        if finbert_threshold is not None:
            self.finbert_threshold = finbert_threshold
            logger.info(f"Updated FinBERT threshold to {finbert_threshold}")
        
        if llm_escalation_threshold is not None:
            self.llm_escalation_threshold = llm_escalation_threshold
            logger.info(f"Updated LLM escalation threshold to {llm_escalation_threshold}")

    def get_model_status(self) -> Dict:
        """Get status of loaded models"""
        return {
            "detoxify_loaded": self.detoxify_classifier is not None,
            "finbert_loaded": self.finbert_classifier is not None,
            "keywords_count": len(self.keywords),
            "thresholds": {
                "toxicity": self.toxicity_threshold,
                "finbert": self.finbert_threshold,
                "llm_escalation": self.llm_escalation_threshold
            }
        } 