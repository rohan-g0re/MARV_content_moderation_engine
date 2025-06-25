"""
GuardianAI - Unified Content Moderation Controller

This module implements the core GuardianAI system that orchestrates all moderation layers
and ensures explainability and tunable thresholds as specified in the project brief.

The system follows a layered approach:
1. Database-Driven Rule Filter (fast keyword lookup)
2. Optional ML Layer (TF-IDF classifier for context) - Future
3. Minimal AI Layer (Llama 3.1 via Ollama for nuance) - Future
4. Feedback & Override System (user/admin moderation) - Future

All logic flows through this unified controller to ensure explainability and tunable thresholds.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from .database_filter import DatabaseFilter
from .ml_filter import MLFilter  # Future implementation
from .llm_filter import LLMFilter  # Future implementation
from .config import ModerationConfig

class ThreatLevel(Enum):
    """Threat levels for content moderation"""
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ModerationAction(Enum):
    """Actions to take based on threat level"""
    ACCEPT = "accept"
    FLAG = "flag"
    REVIEW = "review"
    BLOCK = "block"
    ESCALATE = "escalate"

@dataclass
class ModerationResult:
    """Structured output from moderation pipeline"""
    content_id: str
    threat_level: ThreatLevel
    action: ModerationAction
    confidence: float
    explanation: str
    layer_results: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for API response"""
        return {
            "content_id": self.content_id,
            "threat_level": self.threat_level.name,
            "action": self.action.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "layer_results": self.layer_results,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class LayerResult:
    """Result from individual moderation layer"""
    layer_name: str
    threat_score: float
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class GuardianAI:
    """
    Unified Content Moderation Controller
    
    This is the core system that orchestrates all moderation layers and ensures
    explainability and tunable thresholds as specified in the project brief.
    """
    
    def __init__(self, config: ModerationConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize moderation layers
        self.db_filter = DatabaseFilter(config.database_config, self.logger)
        self.ml_filter = None  # MLFilter(config.ml_config, self.logger)  # Future
        self.llm_filter = None  # LLMFilter(config.llm_config, self.logger)  # Future
        
        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "layer_usage": {
                "database": 0,
                "ml": 0,
                "llm": 0
            },
            "action_distribution": {
                action.value: 0 for action in ModerationAction
            },
            "threat_distribution": {
                level.name: 0 for level in ThreatLevel
            }
        }
        
        self.logger.info("GuardianAI initialized successfully")
    
    def moderate_content(self, content: str, content_id: Optional[str] = None) -> ModerationResult:
        """
        Main entry point for content moderation
        
        Args:
            content: The text content to moderate
            content_id: Optional unique identifier for the content
            
        Returns:
            ModerationResult with threat level, action, and explanation
        """
        import time
        start_time = time.time()
        
        if not content_id:
            content_id = f"content_{int(start_time * 1000)}"
        
        self.logger.info(f"Starting moderation for content_id: {content_id}")
        
        # Initialize result tracking
        layer_results = {}
        final_threat_level = ThreatLevel.SAFE
        final_confidence = 1.0
        final_explanation = "Content appears safe"
        
        # Layer 1: Database-Driven Rule Filter (Always run)
        self.logger.debug("Running database filter layer")
        db_result = self.db_filter.analyze(content)
        layer_results["database"] = db_result.to_dict()
        self.stats["layer_usage"]["database"] += 1
        
        if db_result.threat_score > 0:
            final_threat_level = self._map_score_to_threat_level(db_result.threat_score)
            final_confidence = db_result.confidence
            final_explanation = db_result.explanation
        
        # Layer 2: ML Layer (Optional - for borderline cases)
        if (self.ml_filter and 
            self.config.enable_ml_layer and
            final_threat_level in [ThreatLevel.LOW, ThreatLevel.MEDIUM]):
            
            self.logger.debug("Running ML filter layer")
            ml_result = self.ml_filter.analyze(content)
            layer_results["ml"] = ml_result.to_dict()
            self.stats["layer_usage"]["ml"] += 1
            
            # Combine ML results with database results
            combined_score = (db_result.threat_score + ml_result.threat_score) / 2
            final_threat_level = self._map_score_to_threat_level(combined_score)
            final_confidence = (db_result.confidence + ml_result.confidence) / 2
            final_explanation = f"Database: {db_result.explanation}. ML: {ml_result.explanation}"
        
        # Layer 3: LLM Layer (Minimal use - only for high uncertainty)
        if (self.llm_filter and 
            self.config.enable_llm_layer and
            final_confidence < self.config.llm_threshold and
            final_threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH]):
            
            self.logger.debug("Running LLM filter layer")
            llm_result = self.llm_filter.analyze(content)
            layer_results["llm"] = llm_result.to_dict()
            self.stats["layer_usage"]["llm"] += 1
            
            # Use LLM result as final decision for high uncertainty cases
            final_threat_level = self._map_score_to_threat_level(llm_result.threat_score)
            final_confidence = llm_result.confidence
            final_explanation = llm_result.explanation
        
        # Determine action based on threat level and thresholds
        action = self._determine_action(final_threat_level, final_confidence)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create result
        result = ModerationResult(
            content_id=content_id,
            threat_level=final_threat_level,
            action=action,
            confidence=final_confidence,
            explanation=final_explanation,
            layer_results=layer_results,
            processing_time_ms=processing_time_ms
        )
        
        # Update statistics
        self._update_stats(result)
        
        self.logger.info(f"Moderation completed for {content_id}: {final_threat_level.name} -> {action.value}")
        
        return result
    
    def _map_score_to_threat_level(self, score: float) -> ThreatLevel:
        """Map numerical threat score to ThreatLevel enum"""
        if score >= self.config.critical_threshold:
            return ThreatLevel.CRITICAL
        elif score >= self.config.high_threshold:
            return ThreatLevel.HIGH
        elif score >= self.config.medium_threshold:
            return ThreatLevel.MEDIUM
        elif score >= self.config.low_threshold:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.SAFE
    
    def _determine_action(self, threat_level: ThreatLevel, confidence: float) -> ModerationAction:
        """Determine action based on threat level and confidence"""
        if threat_level == ThreatLevel.SAFE:
            return ModerationAction.ACCEPT
        elif threat_level == ThreatLevel.LOW:
            return ModerationAction.FLAG
        elif threat_level == ThreatLevel.MEDIUM:
            return ModerationAction.REVIEW
        elif threat_level == ThreatLevel.HIGH:
            return ModerationAction.BLOCK
        elif threat_level == ThreatLevel.CRITICAL:
            return ModerationAction.ESCALATE
        else:
            return ModerationAction.REVIEW
    
    def _update_stats(self, result: ModerationResult) -> None:
        """Update internal statistics"""
        self.stats["total_processed"] += 1
        self.stats["action_distribution"][result.action.value] += 1
        self.stats["threat_distribution"][result.threat_level.name] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current moderation statistics"""
        if self.stats["total_processed"] == 0:
            return self.stats
        
        # Calculate percentages
        total = self.stats["total_processed"]
        
        action_percentages = {
            action: (count / total) * 100 
            for action, count in self.stats["action_distribution"].items()
        }
        
        threat_percentages = {
            level: (count / total) * 100 
            for level, count in self.stats["threat_distribution"].items()
        }
        
        layer_percentages = {
            layer: (count / total) * 100 
            for layer, count in self.stats["layer_usage"].items()
        }
        
        return {
            **self.stats,
            "action_percentages": action_percentages,
            "threat_percentages": threat_percentages,
            "layer_percentages": layer_percentages
        }
    
    def update_thresholds(self, new_thresholds: Dict[str, float]) -> bool:
        """Update moderation thresholds dynamically"""
        try:
            if "low_threshold" in new_thresholds:
                self.config.low_threshold = new_thresholds["low_threshold"]
            if "medium_threshold" in new_thresholds:
                self.config.medium_threshold = new_thresholds["medium_threshold"]
            if "high_threshold" in new_thresholds:
                self.config.high_threshold = new_thresholds["high_threshold"]
            if "critical_threshold" in new_thresholds:
                self.config.critical_threshold = new_thresholds["critical_threshold"]
            if "llm_threshold" in new_thresholds:
                self.config.llm_threshold = new_thresholds["llm_threshold"]
            
            self.logger.info(f"Updated thresholds: {new_thresholds}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update thresholds: {e}")
            return False
    
    def enable_layer(self, layer_name: str, enabled: bool) -> bool:
        """Enable or disable specific moderation layers"""
        try:
            if layer_name == "ml":
                self.config.enable_ml_layer = enabled
            elif layer_name == "llm":
                self.config.enable_llm_layer = enabled
            else:
                self.logger.error(f"Unknown layer: {layer_name}")
                return False
            
            self.logger.info(f"{layer_name} layer {'enabled' if enabled else 'disabled'}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to {('enable' if enabled else 'disable')} {layer_name} layer: {e}")
            return False 