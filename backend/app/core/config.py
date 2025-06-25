"""
Configuration Management for GuardianAI Content Moderation Engine

This module provides configuration classes for all components of the moderation system,
including tunable thresholds, database settings, and layer configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import os

@dataclass
class DatabaseConfig:
    """Configuration for database-driven rule filtering"""
    database_url: str = "sqlite:///data/moderation_rules.db"
    rules_file: str = "data/rules/profanity_patterns.json"
    keywords_file: str = "data/rules/severity_keywords.json"
    cache_size: int = 1000
    enable_regex: bool = True
    enable_keyword_matching: bool = True
    enable_pattern_matching: bool = True

@dataclass
class MLConfig:
    """Configuration for ML-based filtering (TF-IDF classifier)"""
    model_path: str = "models/tfidf_classifier.pkl"
    vectorizer_path: str = "models/tfidf_vectorizer.pkl"
    training_data_path: str = "data/processed/training_data.csv"
    confidence_threshold: float = 0.7
    enable_context_analysis: bool = True
    max_features: int = 5000
    ngram_range: tuple = (1, 3)

@dataclass
class LLMConfig:
    """Configuration for LLM-based filtering (Llama 3.1 via Ollama)"""
    ollama_url: str = "http://localhost:11434"
    model_name: str = "llama3.1"
    max_tokens: int = 100
    temperature: float = 0.1
    timeout_seconds: int = 30
    retry_attempts: int = 3
    enable_fallback: bool = True
    fallback_model: str = "llama3.1:3b"

@dataclass
class ModerationConfig:
    """Main configuration for GuardianAI moderation system"""
    
    # Threat level thresholds (0.0 to 1.0)
    low_threshold: float = 0.2
    medium_threshold: float = 0.4
    high_threshold: float = 0.7
    critical_threshold: float = 0.9
    
    # Layer enablement
    enable_ml_layer: bool = False  # Disabled by default for cost efficiency
    enable_llm_layer: bool = False  # Disabled by default for cost efficiency
    
    # LLM usage threshold (only use LLM if confidence < this value)
    llm_threshold: float = 0.6
    
    # Performance settings
    max_processing_time_ms: float = 5000.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    # Database configuration
    database_config: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # ML configuration
    ml_config: MLConfig = field(default_factory=MLConfig)
    
    # LLM configuration
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: str = "logs/guardian_ai.log"
    enable_structured_logging: bool = True
    
    # Output configuration
    enable_detailed_explanations: bool = True
    include_layer_results: bool = True
    include_processing_time: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_thresholds()
        self._validate_paths()
    
    def _validate_thresholds(self):
        """Validate that thresholds are in correct order"""
        thresholds = [
            self.low_threshold,
            self.medium_threshold,
            self.high_threshold,
            self.critical_threshold
        ]
        
        for i in range(len(thresholds) - 1):
            if thresholds[i] >= thresholds[i + 1]:
                raise ValueError(
                    f"Thresholds must be in ascending order. "
                    f"Current: {thresholds[i]} >= {thresholds[i + 1]}"
                )
        
        # Validate threshold ranges
        for threshold in thresholds:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Threshold must be between 0.0 and 1.0: {threshold}")
    
    def _validate_paths(self):
        """Validate that required paths exist or can be created"""
        paths_to_check = [
            Path(self.database_config.rules_file).parent,
            Path(self.database_config.keywords_file).parent,
            Path(self.log_file).parent,
            Path("data/processed"),
            Path("models"),
            Path("logs")
        ]
        
        for path in paths_to_check:
            path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for API responses"""
        return {
            "thresholds": {
                "low": self.low_threshold,
                "medium": self.medium_threshold,
                "high": self.high_threshold,
                "critical": self.critical_threshold
            },
            "layers": {
                "ml_enabled": self.enable_ml_layer,
                "llm_enabled": self.enable_llm_layer,
                "llm_threshold": self.llm_threshold
            },
            "performance": {
                "max_processing_time_ms": self.max_processing_time_ms,
                "enable_caching": self.enable_caching,
                "cache_ttl_seconds": self.cache_ttl_seconds
            },
            "output": {
                "detailed_explanations": self.enable_detailed_explanations,
                "include_layer_results": self.include_layer_results,
                "include_processing_time": self.include_processing_time
            }
        }
    
    def update_from_dict(self, config_dict: Dict) -> bool:
        """Update configuration from dictionary"""
        try:
            # Update thresholds
            if "thresholds" in config_dict:
                thresholds = config_dict["thresholds"]
                if "low" in thresholds:
                    self.low_threshold = float(thresholds["low"])
                if "medium" in thresholds:
                    self.medium_threshold = float(thresholds["medium"])
                if "high" in thresholds:
                    self.high_threshold = float(thresholds["high"])
                if "critical" in thresholds:
                    self.critical_threshold = float(thresholds["critical"])
            
            # Update layer settings
            if "layers" in config_dict:
                layers = config_dict["layers"]
                if "ml_enabled" in layers:
                    self.enable_ml_layer = bool(layers["ml_enabled"])
                if "llm_enabled" in layers:
                    self.enable_llm_layer = bool(layers["llm_enabled"])
                if "llm_threshold" in layers:
                    self.llm_threshold = float(layers["llm_threshold"])
            
            # Validate updated configuration
            self._validate_thresholds()
            return True
            
        except Exception as e:
            print(f"Failed to update configuration: {e}")
            return False

def load_config_from_env() -> ModerationConfig:
    """Load configuration from environment variables"""
    config = ModerationConfig()
    
    # Load thresholds from environment
    if os.getenv("LOW_THRESHOLD"):
        config.low_threshold = float(os.getenv("LOW_THRESHOLD"))
    if os.getenv("MEDIUM_THRESHOLD"):
        config.medium_threshold = float(os.getenv("MEDIUM_THRESHOLD"))
    if os.getenv("HIGH_THRESHOLD"):
        config.high_threshold = float(os.getenv("HIGH_THRESHOLD"))
    if os.getenv("CRITICAL_THRESHOLD"):
        config.critical_threshold = float(os.getenv("CRITICAL_THRESHOLD"))
    
    # Load layer settings
    if os.getenv("ENABLE_ML_LAYER"):
        config.enable_ml_layer = os.getenv("ENABLE_ML_LAYER").lower() == "true"
    if os.getenv("ENABLE_LLM_LAYER"):
        config.enable_llm_layer = os.getenv("ENABLE_LLM_LAYER").lower() == "true"
    if os.getenv("LLM_THRESHOLD"):
        config.llm_threshold = float(os.getenv("LLM_THRESHOLD"))
    
    # Load database settings
    if os.getenv("DATABASE_URL"):
        config.database_config.database_url = os.getenv("DATABASE_URL")
    
    # Load LLM settings
    if os.getenv("OLLAMA_URL"):
        config.llm_config.ollama_url = os.getenv("OLLAMA_URL")
    if os.getenv("OLLAMA_MODEL"):
        config.llm_config.model_name = os.getenv("OLLAMA_MODEL")
    
    return config

def get_default_config() -> ModerationConfig:
    """Get default configuration optimized for cost efficiency"""
    return ModerationConfig(
        # Conservative thresholds for high precision
        low_threshold=0.3,
        medium_threshold=0.5,
        high_threshold=0.8,
        critical_threshold=0.95,
        
        # Disable expensive layers by default
        enable_ml_layer=False,
        enable_llm_layer=False,
        
        # High LLM threshold to minimize usage
        llm_threshold=0.7,
        
        # Performance optimizations
        enable_caching=True,
        cache_ttl_seconds=7200,  # 2 hours
        max_processing_time_ms=2000.0
    )

def get_aggressive_config() -> ModerationConfig:
    """Get aggressive configuration for high recall"""
    return ModerationConfig(
        # Lower thresholds for higher recall
        low_threshold=0.1,
        medium_threshold=0.3,
        high_threshold=0.6,
        critical_threshold=0.8,
        
        # Enable all layers
        enable_ml_layer=True,
        enable_llm_layer=True,
        
        # Lower LLM threshold for more usage
        llm_threshold=0.5,
        
        # Allow longer processing
        max_processing_time_ms=10000.0
    ) 