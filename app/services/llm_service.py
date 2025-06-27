"""
LLM service for complex content analysis using Ollama with LLaMA 3.1
"""

import logging
import aiohttp
import json
from typing import Dict, Any, List, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM-based content analysis using Ollama"""
    
    def __init__(self):
        """Initialize LLM service"""
        self.ollama_url = settings.OLLAMA_URL
        self.model_name = settings.OLLAMA_MODEL
        self.enabled = settings.ENABLE_LLM
        
        logger.info(f"LLMService initialized with Ollama at {self.ollama_url}")
    
    async def analyze(self, content: str, rule_result: Any, ml_result: Any) -> Dict[str, Any]:
        """
        Analyze content using LLM for complex reasoning
        
        Args:
            content: Text content to analyze
            rule_result: Results from rule-based filtering
            ml_result: Results from ML analysis
            
        Returns:
            Dictionary with LLM analysis results
        """
        try:
            if not self.enabled:
                return self._get_disabled_result()
            
            # Create prompt for LLM analysis
            prompt = self._create_analysis_prompt(content, rule_result, ml_result)
            
            # Send request to Ollama
            response = await self._call_ollama(prompt)
            
            # Parse LLM response
            analysis = self._parse_llm_response(response)
            
            return {
                "threat_score": analysis["threat_score"],
                "confidence": analysis["confidence"],
                "reasoning": analysis["reasoning"],
                "suggestions": analysis.get("suggestions", []),
                "model_used": self.model_name,
                "prompt_tokens": len(prompt.split()),
                "response_tokens": len(response.split())
            }
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return self._get_error_result(str(e))
    
    def _create_analysis_prompt(self, content: str, rule_result: Any, ml_result: Any) -> str:
        """Create analysis prompt for LLM"""
        
        # Extract relevant information from previous layers
        rule_severity = getattr(rule_result, 'score', 0)
        rule_matches = getattr(rule_result, 'matches', [])
        ml_score = getattr(ml_result, 'score', 0)
        ml_confidence = getattr(ml_result, 'confidence', 0)
        
        # Create rule summary
        rule_summary = "No rule violations detected."
        if rule_matches:
            rule_types = set(match.get("type", "unknown") for match in rule_matches)
            rule_summary = f"Rule violations: {', '.join(rule_types)} (severity: {rule_severity})"
        
        prompt = f"""
You are an AI content moderator analyzing a piece of text for potential violations. 

CONTENT TO ANALYZE:
"{content}"

PREVIOUS ANALYSIS RESULTS:
- Rule-based filtering: {rule_summary}
- ML analysis: Overall score {ml_score:.2f} (confidence: {ml_confidence:.2f})

TASK:
Analyze this content considering the previous results and provide:
1. A threat assessment (0.0 to 1.0 scale)
2. Your confidence in this assessment (0.0 to 1.0 scale)
3. Detailed reasoning for your decision
4. Suggestions for safer alternatives (if content is problematic)

Consider:
- Context and intent
- Potential for harm
- Cultural sensitivity
- Nuanced language use
- False positives from rule-based systems

RESPONSE FORMAT (JSON):
{{
    "threat_score": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of your assessment",
    "suggestions": ["suggestion1", "suggestion2"]
}}

Analyze the content and respond with the JSON format above:
"""
        
        return prompt.strip()
    
    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API with the given prompt"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more consistent results
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                }
                
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status != 200:
                        raise Exception(f"Ollama API error: {response.status}")
                    
                    data = await response.json()
                    return data.get("response", "")
                    
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract structured data"""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                return {
                    "threat_score": float(data.get("threat_score", 0.0)),
                    "confidence": float(data.get("confidence", 0.0)),
                    "reasoning": data.get("reasoning", "No reasoning provided"),
                    "suggestions": data.get("suggestions", [])
                }
            else:
                # Fallback parsing if JSON not found
                return self._parse_fallback_response(response)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            return self._parse_fallback_response(response)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._get_error_result("Failed to parse LLM response")
    
    def _parse_fallback_response(self, response: str) -> Dict[str, Any]:
        """Parse response when JSON format is not available"""
        # Simple keyword-based parsing
        response_lower = response.lower()
        
        # Extract threat score from text
        threat_score = 0.0
        if "high threat" in response_lower or "dangerous" in response_lower:
            threat_score = 0.8
        elif "medium threat" in response_lower or "concerning" in response_lower:
            threat_score = 0.5
        elif "low threat" in response_lower or "safe" in response_lower:
            threat_score = 0.2
        
        # Extract confidence
        confidence = 0.7  # Default confidence
        
        return {
            "threat_score": threat_score,
            "confidence": confidence,
            "reasoning": response[:200] + "..." if len(response) > 200 else response,
            "suggestions": []
        }
    
    def _get_disabled_result(self) -> Dict[str, Any]:
        """Get result when LLM is disabled"""
        return {
            "threat_score": 0.0,
            "confidence": 0.0,
            "reasoning": "LLM analysis disabled",
            "suggestions": [],
            "model_used": "none",
            "prompt_tokens": 0,
            "response_tokens": 0
        }
    
    def _get_error_result(self, error_message: str) -> Dict[str, Any]:
        """Get result when LLM analysis fails"""
        return {
            "threat_score": 0.0,
            "confidence": 0.0,
            "reasoning": f"LLM analysis failed: {error_message}",
            "suggestions": [],
            "model_used": self.model_name,
            "prompt_tokens": 0,
            "response_tokens": 0
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ollama_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        models = [model["name"] for model in data.get("models", [])]
                        
                        return {
                            "status": "connected",
                            "available_models": models,
                            "target_model": self.model_name,
                            "model_available": self.model_name in models
                        }
                    else:
                        return {
                            "status": "error",
                            "error": f"HTTP {response.status}"
                        }
                        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the LLM model"""
        return {
            "enabled": self.enabled,
            "ollama_url": self.ollama_url,
            "model_name": self.model_name,
            "threshold": settings.LLM_THRESHOLD
        }
    
    async def generate_suggestions(self, content: str, issue_type: str) -> List[str]:
        """Generate suggestions for improving content"""
        try:
            if not self.enabled:
                return []
            
            prompt = f"""
The following content was flagged for {issue_type}:

"{content}"

Please provide 3-5 suggestions for how to rephrase this content to make it more appropriate while preserving the original intent. 

Respond with a JSON array of suggestions:
["suggestion1", "suggestion2", "suggestion3"]
"""
            
            response = await self._call_ollama(prompt)
            
            # Parse suggestions
            try:
                json_start = response.find('[')
                json_end = response.rfind(']') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    suggestions = json.loads(json_str)
                    return suggestions if isinstance(suggestions, list) else []
                else:
                    # Fallback: extract suggestions from text
                    lines = response.split('\n')
                    suggestions = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('[') and not line.startswith(']'):
                            suggestions.append(line)
                    return suggestions[:5]  # Limit to 5 suggestions
                    
            except json.JSONDecodeError:
                return []
                
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return [] 