"""
Database configuration and connection management.
Handles PostgreSQL connection using SQLAlchemy ORM.
"""

import logging
import os
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from typing import Generator
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
class DatabaseConfig:
    """Database configuration class."""
    
    def __init__(self):
        # PostgreSQL connection parameters
        # Default values for development - should be overridden by environment variables
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = os.getenv("POSTGRES_DB", "content_moderation")
        self.username = os.getenv("POSTGRES_USER", "postgres")
        self.password = os.getenv("POSTGRES_PASSWORD", "postgres")
        
        # Connection pool settings
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        
        # Construct database URL
        self.database_url = (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )
        
        logger.info(f"Database config: {self.host}:{self.port}/{self.database}")

# Global database configuration
db_config = DatabaseConfig()

# Create SQLAlchemy engine
engine = create_engine(
    db_config.database_url,
    poolclass=QueuePool,
    pool_size=db_config.pool_size,
    max_overflow=db_config.max_overflow,
    pool_timeout=db_config.pool_timeout,
    pool_pre_ping=True,  # Validate connections before use
    echo=False  # Set to True for SQL query debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base for models
Base = declarative_base()

def get_database_session() -> Generator[Session, None, None]:
    """
    Create and manage database session.
    
    Yields:
        Session: SQLAlchemy database session
    """
    session = SessionLocal()
    try:
        yield session
    except Exception as e:
        logger.error(f"Database session error: {e}")
        session.rollback()
        raise
    finally:
        session.close()

def test_database_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        logger.info("Testing database connection...")
        
        # Create a test session
        with SessionLocal() as session:
            # Simple query to test connection
            result = session.execute(text("SELECT 1 as test"))
            test_value = result.scalar()
            
            if test_value == 1:
                logger.info("✅ Database connection successful!")
                return True
            else:
                logger.error("❌ Database connection test failed")
                return False
                
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def create_database_tables():
    """
    Create all database tables defined in models.
    """
    try:
        logger.info("Creating database tables...")
        
        # Import models to register them with Base
        from ..models.keyword import Keyword
        from ..models.pattern import Pattern
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("✅ Database tables created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {e}")
        return False

def drop_database_tables():
    """
    Drop all database tables (for testing/reset purposes).
    WARNING: This will delete all data!
    """
    try:
        logger.warning("⚠️  Dropping all database tables...")
        
        # Import models to register them with Base
        from ..models.keyword import Keyword
        from ..models.pattern import Pattern
        
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        
        logger.info("✅ Database tables dropped successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error dropping database tables: {e}")
        return False

def get_database_info() -> dict:
    """
    Get database connection information.
    
    Returns:
        dict: Database connection information
    """
    return {
        "host": db_config.host,
        "port": db_config.port,
        "database": db_config.database,
        "username": db_config.username,
        "pool_size": db_config.pool_size,
        "max_overflow": db_config.max_overflow,
        "connection_url": db_config.database_url.replace(db_config.password, "***")
    }

# Database initialization function
def initialize_database():
    """
    Initialize the database by testing connection and creating tables.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    logger.info("Initializing database...")
    
    # Test connection first
    if not test_database_connection():
        logger.error("Cannot proceed with database initialization - connection failed")
        return False
    
    # Create tables
    if not create_database_tables():
        logger.error("Cannot proceed with database initialization - table creation failed")
        return False
    
    logger.info("✅ Database initialization completed successfully!")
    return True 