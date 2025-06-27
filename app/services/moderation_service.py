"""
Main moderation service that orchestrates the multi-layer moderation pipeline
"""

import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

from app.models.schemas import ModerationRequest, ModerationResult, ThreatLevel, ModerationAction
from app.services.rule_service import RuleService
from app.services.ml_service import MLService
from app.services.llm_service import LLMService
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LayerResult:
    """Result from a single moderation layer"""
    layer_name: str
    score: float
    confidence: float
    matches: List[Dict[str, Any]]
    processing_time_ms: int
    metadata: Dict[str, Any]


class ModerationService:
    """Main moderation service orchestrating all layers"""
    
    def __init__(self):
        """Initialize moderation service with all layers"""
        self.rule_service = RuleService()
        self.ml_service = MLService()
        self.llm_service = LLMService()
        
        logger.info("ModerationService initialized with all layers")
    
    async def moderate_content(self, request: ModerationRequest) -> ModerationResult:
        """
        Main entrypoint for content moderation
        
        Args:
            request: ModerationRequest containing content and metadata
            
        Returns:
            ModerationResult with structured output
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting moderation for content: {request.content[:100]}...")
            
            # Layer 1: Rule-based filtering
            rule_result = await self._run_rule_layer(request.content)
            
            # Layer 2: ML/AI detection
            ml_result = await self._run_ml_layer(request.content)
            
            # Layer 3: LLM escalation (if needed)
            llm_result = await self._run_llm_layer(request.content, rule_result, ml_result)
            
            # Calculate final decision
            final_result = self._calculate_final_decision(rule_result, ml_result, llm_result)
            
            # Calculate total processing time
            total_time_ms = int((time.time() - start_time) * 1000)
            
            # Create moderation result
            result = ModerationResult(
                post_id=0,  # Will be set by the API layer
                action=final_result["action"],
                threat_level=final_result["threat_level"],
                confidence=final_result["confidence"],
                explanation=final_result["explanation"],
                processing_time_ms=total_time_ms,
                metadata={
                    "layer_results": {
                        "rule": rule_result.metadata,
                        "ml": ml_result.metadata,
                        "llm": llm_result.metadata if llm_result else None
                    },
                    "layer_times": {
                        "rule": rule_result.processing_time_ms,
                        "ml": ml_result.processing_time_ms,
                        "llm": llm_result.processing_time_ms if llm_result else 0
                    }
                },
                created_at=datetime.utcnow()
            )
            
            logger.info(f"Moderation completed: {result.action} ({result.threat_level})")
            return result
            
        except Exception as e:
            logger.error(f"Error in moderation pipeline: {e}")
            # Return safe default result
            return ModerationResult(
                post_id=0,
                action=ModerationAction.FLAG,
                threat_level=ThreatLevel.MEDIUM,
                confidence=0.0,
                explanation=f"Error during moderation: {str(e)}",
                processing_time_ms=int((time.time() - start_time) * 1000),
                metadata={"error": str(e)},
                created_at=datetime.utcnow()
            )
    
    async def _run_rule_layer(self, content: str) -> LayerResult:
        """Run rule-based filtering layer"""
        start_time = time.time()
        
        try:
            result = await self.rule_service.analyze(content)
            
            return LayerResult(
                layer_name="rule",
                score=result["total_severity"],
                confidence=result["confidence"],
                matches=result["matches"],
                processing_time_ms=int((time.time() - start_time) * 1000),
                metadata=result
            )
        except Exception as e:
            logger.error(f"Error in rule layer: {e}")
            return LayerResult(
                layer_name="rule",
                score=0.0,
                confidence=0.0,
                matches=[],
                processing_time_ms=int((time.time() - start_time) * 1000),
                metadata={"error": str(e)}
            )
    
    async def _run_ml_layer(self, content: str) -> LayerResult:
        """Run ML/AI detection layer"""
        start_time = time.time()
        
        try:
            result = await self.ml_service.analyze(content)
            
            return LayerResult(
                layer_name="ml",
                score=result["overall_score"],
                confidence=result["confidence"],
                matches=result["detections"],
                processing_time_ms=int((time.time() - start_time) * 1000),
                metadata=result
            )
        except Exception as e:
            logger.error(f"Error in ML layer: {e}")
            return LayerResult(
                layer_name="ml",
                score=0.0,
                confidence=0.0,
                matches=[],
                processing_time_ms=int((time.time() - start_time) * 1000),
                metadata={"error": str(e)}
            )
    
    async def _run_llm_layer(self, content: str, rule_result: LayerResult, ml_result: LayerResult) -> Optional[LayerResult]:
        """Run LLM escalation layer (if needed)"""
        if not settings.ENABLE_LLM:
            return None
        
        # Check if LLM escalation is needed
        if not self._should_escalate_to_llm(rule_result, ml_result):
            return None
        
        start_time = time.time()
        
        try:
            result = await self.llm_service.analyze(content, rule_result, ml_result)
            
            return LayerResult(
                layer_name="llm",
                score=result["threat_score"],
                confidence=result["confidence"],
                matches=result["reasoning"],
                processing_time_ms=int((time.time() - start_time) * 1000),
                metadata=result
            )
        except Exception as e:
            logger.error(f"Error in LLM layer: {e}")
            return None
    
    def _should_escalate_to_llm(self, rule_result: LayerResult, ml_result: LayerResult) -> bool:
        """Determine if content should be escalated to LLM"""
        # Escalate if rule severity is medium or ML confidence is uncertain
        rule_escalation = rule_result.score >= settings.RULE_SEVERITY_THRESHOLD
        ml_escalation = ml_result.confidence < settings.LLM_THRESHOLD
        
        return rule_escalation or ml_escalation
    
    def _calculate_final_decision(self, rule_result: LayerResult, ml_result: LayerResult, llm_result: Optional[LayerResult]) -> Dict[str, Any]:
        """Calculate final moderation decision based on all layers"""
        
        # Calculate threat level
        threat_level = self._calculate_threat_level(rule_result, ml_result, llm_result)
        
        # Determine action
        action = self._determine_action(threat_level, rule_result, ml_result, llm_result)
        
        # Calculate confidence
        confidence = self._calculate_confidence(rule_result, ml_result, llm_result)
        
        # Generate explanation
        explanation = self._generate_explanation(rule_result, ml_result, llm_result, threat_level, action)
        
        return {
            "threat_level": threat_level,
            "action": action,
            "confidence": confidence,
            "explanation": explanation
        }
    
    def _calculate_threat_level(self, rule_result: LayerResult, ml_result: LayerResult, llm_result: Optional[LayerResult]) -> ThreatLevel:
        """Calculate overall threat level with enhanced financial fraud detection"""
        
        # Base threat from rule severity
        rule_threat = min(rule_result.score / 10.0, 1.0)
        
        # ML threat (average of toxicity and fraud)
        ml_threat = ml_result.score
        
        # LLM threat (if available)
        llm_threat = llm_result.score if llm_result else 0.0
        
        # Enhanced financial fraud detection
        financial_fraud_boost = self._detect_financial_fraud(rule_result, ml_result)
        
        # Combine threats with financial fraud boost
        if llm_result:
            # With LLM: 25% rule, 25% ML, 35% LLM, 15% financial fraud boost
            overall_threat = (0.25 * rule_threat + 0.25 * ml_threat + 0.35 * llm_threat + 0.15 * financial_fraud_boost)
        else:
            # Without LLM: 40% rule, 40% ML, 20% financial fraud boost
            overall_threat = (0.4 * rule_threat + 0.4 * ml_threat + 0.2 * financial_fraud_boost)
        
        # Apply financial fraud multiplier if significant fraud detected
        if financial_fraud_boost > 0.5:
            overall_threat = min(overall_threat * settings.FINANCIAL_FRAUD_MULTIPLIER, 1.0)
        
        # Determine threat level with adjusted thresholds
        if overall_threat >= 0.7:  # Lowered from 0.8
            return ThreatLevel.CRITICAL
        elif overall_threat >= 0.5:  # Lowered from 0.6
            return ThreatLevel.HIGH
        elif overall_threat >= 0.3:  # Lowered from 0.4
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _detect_financial_fraud(self, rule_result: LayerResult, ml_result: LayerResult) -> float:
        """Detect financial fraud patterns and return boost score"""
        fraud_score = 0.0
        
        # Check rule matches for financial fraud patterns
        if rule_result.matches:
            for match in rule_result.matches:
                category = match.get("category", "").lower()
                severity = match.get("severity", 0)
                
                # Financial fraud patterns
                if "financial_fraud" in category:
                    fraud_score += severity / 10.0
                elif "fraud" in category:
                    fraud_score += severity / 10.0 * 0.8
                elif "scam" in category:
                    fraud_score += severity / 10.0 * 0.7
                
                # Check for specific high-risk patterns
                pattern = match.get("pattern", "").lower()
                if any(keyword in pattern for keyword in ["guarantee", "insider", "parabolic", "10x", "20x", "moon", "explode"]):
                    fraud_score += 0.3
        
        # Check ML fraud score
        ml_fraud = ml_result.metadata.get("fraud_score", 0)
        if ml_fraud > settings.FRAUD_THRESHOLD:
            fraud_score += ml_fraud * 0.5
        
        # Check for pump-and-dump language patterns
        pump_dump_keywords = ["guarantee", "insider", "parabolic", "pump", "dump", "moon", "rocket", "explode", "breakthrough"]
        content_lower = rule_result.metadata.get("content", "").lower()
        pump_dump_count = sum(1 for keyword in pump_dump_keywords if keyword in content_lower)
        
        if pump_dump_count >= 3:
            fraud_score += 0.4
        elif pump_dump_count >= 2:
            fraud_score += 0.2
        elif pump_dump_count >= 1:
            fraud_score += 0.1
        
        return min(fraud_score, 1.0)
    
    def _determine_action(self, threat_level: ThreatLevel, rule_result: LayerResult, ml_result: LayerResult, llm_result: Optional[LayerResult]) -> ModerationAction:
        """Determine moderation action based on threat level and layer results"""
        
        if threat_level == ThreatLevel.CRITICAL:
            return ModerationAction.BLOCK
        elif threat_level == ThreatLevel.HIGH:
            return ModerationAction.QUARANTINE
        elif threat_level == ThreatLevel.MEDIUM:
            return ModerationAction.FLAG
        else:
            return ModerationAction.ACCEPT
    
    def _calculate_confidence(self, rule_result: LayerResult, ml_result: LayerResult, llm_result: Optional[LayerResult]) -> float:
        """Calculate overall confidence score"""
        
        # Base confidence from ML layer
        confidence = ml_result.confidence
        
        # Boost confidence if rule layer agrees
        if rule_result.score > 0:
            confidence = min(confidence + 0.1, 1.0)
        
        # Boost confidence if LLM was used and agrees
        if llm_result and llm_result.confidence > 0.7:
            confidence = min(confidence + 0.1, 1.0)
        
        return confidence
    
    def _generate_explanation(self, rule_result: LayerResult, ml_result: LayerResult, llm_result: Optional[LayerResult], threat_level: ThreatLevel, action: ModerationAction) -> str:
        """Generate human-readable explanation with enhanced financial fraud details"""
        
        explanations = []
        
        # Rule-based explanations with specific details
        if rule_result.matches:
            # Group matches by category
            categories = {}
            for match in rule_result.matches:
                category = match.get("category", "unknown")
                if category not in categories:
                    categories[category] = []
                categories[category].append(match)
            
            # Generate specific explanations for each category
            for category, matches in categories.items():
                if category == "financial_fraud":
                    fraud_patterns = [m.get("pattern", "") for m in matches]
                    explanations.append(f"Financial fraud detected: {', '.join(fraud_patterns[:3])}")
                elif category == "profanity":
                    explanations.append(f"Profanity detected: {len(matches)} violations")
                elif category == "threat":
                    explanations.append(f"Threats detected: {len(matches)} violations")
                elif category == "fraud":
                    explanations.append(f"Fraud patterns detected: {len(matches)} violations")
                else:
                    explanations.append(f"{category.title()} violations: {len(matches)} detected")
        
        # ML explanations with specific scores
        toxicity_score = ml_result.metadata.get("toxicity_score", 0)
        fraud_score = ml_result.metadata.get("fraud_score", 0)
        
        if toxicity_score > settings.TOXICITY_THRESHOLD:
            explanations.append(f"Toxicity score: {toxicity_score:.2f} (threshold: {settings.TOXICITY_THRESHOLD})")
        
        if fraud_score > settings.FRAUD_THRESHOLD:
            explanations.append(f"Fraud risk score: {fraud_score:.2f} (threshold: {settings.FRAUD_THRESHOLD})")
        
        # Financial fraud specific detection
        financial_fraud_boost = self._detect_financial_fraud(rule_result, ml_result)
        if financial_fraud_boost > 0.3:
            explanations.append(f"Financial fraud indicators detected (boost: {financial_fraud_boost:.2f})")
        
        # LLM explanations
        if llm_result and llm_result.matches:
            llm_reasoning = llm_result.matches[0] if llm_result.matches else "Complex content analysis"
            explanations.append(f"AI analysis: {llm_reasoning}")
        
        # Threat level explanation
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            explanations.append(f"High threat level: {threat_level.value}")
        
        # Default explanation if no issues found
        if not explanations:
            explanations.append("Content passed all moderation checks")
        
        return "; ".join(explanations)
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_rules": await self.rule_service.get_rule_count(),
            "ml_models_loaded": self.ml_service.models_loaded,
            "llm_enabled": settings.ENABLE_LLM,
            "llm_threshold": settings.LLM_THRESHOLD,
        } 