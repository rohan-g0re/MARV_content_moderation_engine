"""
Core configuration for MARV Content Moderation Engine
"""

import os
from typing import Optional, List


class Settings:
    """Application settings with environment variable support"""
    
    def __init__(self):
        # Database Configuration
        self.DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./marv.db")
        
        # API Configuration
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        self.DEBUG = os.getenv("DEBUG", "True").lower() == "true"
        
        # Security
        self.SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
        self.ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        
        # ML Model Configuration
        self.DETOXIFY_MODEL = os.getenv("DETOXIFY_MODEL", "original")
        self.FINBERT_MODEL = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
        
        # LLM Configuration
        self.OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
        self.ENABLE_LLM = os.getenv("ENABLE_LLM", "True").lower() == "true"
        self.LLM_THRESHOLD = float(os.getenv("LLM_THRESHOLD", "0.6"))
        
        # Moderation Configuration
        self.TOXICITY_THRESHOLD = float(os.getenv("TOXICITY_THRESHOLD", "0.4"))
        self.FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.3"))
        self.RULE_SEVERITY_THRESHOLD = int(os.getenv("RULE_SEVERITY_THRESHOLD", "3"))
        self.FINANCIAL_FRAUD_MULTIPLIER = float(os.getenv("FINANCIAL_FRAUD_MULTIPLIER", "2.0"))
        
        # Rate Limiting
        self.RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
        
        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = os.getenv(
            "LOG_FORMAT", 
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # CORS
        allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
        self.ALLOWED_ORIGINS = allowed_origins.split(",") if "," in allowed_origins else [allowed_origins]
        
        # File Upload
        self.MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10MB


# Global settings instance
settings = Settings()


def get_database_url() -> str:
    """Get database URL with fallback to SQLite"""
    return settings.DATABASE_URL


def is_production() -> bool:
    """Check if running in production mode"""
    return not settings.DEBUG


def get_allowed_origins() -> List[str]:
    """Get allowed CORS origins"""
    return settings.ALLOWED_ORIGINS


def get_ml_config() -> dict:
    """Get ML model configuration"""
    return {
        "detoxify_model": settings.DETOXIFY_MODEL,
        "finbert_model": settings.FINBERT_MODEL,
        "toxicity_threshold": settings.TOXICITY_THRESHOLD,
        "fraud_threshold": settings.FRAUD_THRESHOLD,
    }


def get_llm_config() -> dict:
    """Get LLM configuration"""
    return {
        "ollama_url": settings.OLLAMA_URL,
        "ollama_model": settings.OLLAMA_MODEL,
        "enabled": settings.ENABLE_LLM,
        "threshold": settings.LLM_THRESHOLD,
    } 