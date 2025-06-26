"""
Database population service.
Combines JSON loading, ML classification, and database population.
"""

import logging
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
import time

from ..core.database import get_database_session, initialize_database, test_database_connection
from ..models.keyword import Keyword
from ..models.pattern import Pattern
from ..services.ml_classifier import ToxicityClassifier
from ..utils.file_operations import load_words_from_json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabasePopulator:
    """
    Service for populating database with keywords and their ML classifications.
    """
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize database populator.
        
        Args:
            batch_size (int): Batch size for ML processing
        """
        self.batch_size = batch_size
        self.classifier = None
        
    def setup_ml_classifier(self) -> bool:
        """
        Setup and load the ML classifier.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Setting up ML classifier...")
            self.classifier = ToxicityClassifier(batch_size=self.batch_size)
            
            if not self.classifier.load_model():
                logger.error("Failed to load ML model")
                return False
            
            logger.info("✅ ML classifier setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up ML classifier: {e}")
            return False
    
    def populate_keywords_from_json(self, json_file_path: str, clear_existing: bool = False) -> bool:
        """
        Populate keywords table from JSON file with ML classification.
        
        Args:
            json_file_path (str): Path to JSON file containing words
            clear_existing (bool): Whether to clear existing keywords first
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Starting keyword population from JSON...")
            
            # Step 1: Load words from JSON
            words = load_words_from_json(json_file_path)
            if not words:
                logger.error("No words loaded from JSON file")
                return False
            
            logger.info(f"Loaded {len(words)} words from JSON")
            
            # Step 2: Setup ML classifier if not already done
            if not self.classifier:
                if not self.setup_ml_classifier():
                    return False
            
            # Step 3: Classify words using ML
            logger.info("Classifying words using ML model...")
            ml_results = self.classifier.classify_words(words)
            
            if not ml_results:
                logger.error("ML classification failed")
                return False
            
            logger.info(f"Successfully classified {len(ml_results)} words")
            
            # Step 4: Populate database
            return self._populate_keywords_from_ml_results(ml_results, clear_existing)
            
        except Exception as e:
            logger.error(f"Error in populate_keywords_from_json: {e}")
            return False
    
    def _populate_keywords_from_ml_results(self, ml_results: List[Dict], clear_existing: bool = False) -> bool:
        """
        Populate keywords table from ML classification results.
        
        Args:
            ml_results (List[Dict]): ML classification results
            clear_existing (bool): Whether to clear existing keywords first
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Populating database with ML results...")
            
            with next(get_database_session()) as session:
                # Clear existing data if requested
                if clear_existing:
                    logger.warning("Clearing existing keywords and patterns...")
                    session.execute(text("DELETE FROM patterns"))
                    session.execute(text("DELETE FROM keywords"))
                    session.commit()
                    logger.info("Existing data cleared")
                
                # Insert keywords in batches
                batch_size = 100  # Database batch size
                keywords_added = 0
                
                for i in range(0, len(ml_results), batch_size):
                    batch = ml_results[i:i + batch_size]
                    
                    # Create keyword objects
                    keyword_objects = []
                    for result in batch:
                        try:
                            keyword = Keyword.create_from_ml_result(result)
                            keyword_objects.append(keyword)
                        except Exception as e:
                            logger.warning(f"Error creating keyword for '{result.get('word', 'unknown')}': {e}")
                            continue
                    
                    # Insert batch
                    if keyword_objects:
                        session.add_all(keyword_objects)
                        session.commit()
                        keywords_added += len(keyword_objects)
                        
                        logger.info(f"Inserted batch {i//batch_size + 1}: {len(keyword_objects)} keywords "
                                  f"(Total: {keywords_added}/{len(ml_results)})")
                
                logger.info(f"✅ Successfully populated {keywords_added} keywords")
                
                # Log tier statistics
                self._log_tier_statistics(session)
                
                return True
                
        except Exception as e:
            logger.error(f"Error populating database: {e}")
            return False
    
    def _log_tier_statistics(self, session: Session):
        """Log statistics about tier distribution."""
        try:
            tier_stats = session.execute(
                text("SELECT tier, COUNT(*) as count FROM keywords GROUP BY tier ORDER BY tier")
            ).fetchall()
            
            logger.info("Tier distribution in database:")
            for tier, count in tier_stats:
                tier_name = {1: "High", 2: "Medium", 3: "Low"}.get(tier, "Unknown")
                logger.info(f"  Tier {tier} ({tier_name}): {count} keywords")
                
        except Exception as e:
            logger.warning(f"Could not log tier statistics: {e}")
    
    def verify_database_population(self) -> Dict:
        """
        Verify database population and return statistics.
        
        Returns:
            Dict: Database population statistics
        """
        try:
            with next(get_database_session()) as session:
                # Count keywords by tier
                tier_counts = {}
                for tier in [1, 2, 3]:
                    count = session.query(Keyword).filter(Keyword.tier == tier).count()
                    tier_counts[tier] = count
                
                # Total keywords
                total_keywords = session.query(Keyword).count()
                
                # Active keywords
                active_keywords = session.query(Keyword).filter(Keyword.is_active == True).count()
                
                # Sample keywords from each tier
                samples = {}
                for tier in [1, 2, 3]:
                    sample = session.query(Keyword).filter(Keyword.tier == tier).limit(3).all()
                    samples[tier] = [k.word for k in sample]
                
                return {
                    'total_keywords': total_keywords,
                    'active_keywords': active_keywords,
                    'tier_counts': tier_counts,
                    'samples': samples,
                    'verification_time': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
        except Exception as e:
            logger.error(f"Error verifying database population: {e}")
            return {'error': str(e)}

def populate_database_from_json(json_file_path: str, clear_existing: bool = False, 
                               batch_size: int = 32) -> bool:
    """
    Convenience function to populate database from JSON file.
    
    Args:
        json_file_path (str): Path to JSON file
        clear_existing (bool): Whether to clear existing data
        batch_size (int): ML processing batch size
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("=" * 60)
        logger.info("STARTING DATABASE POPULATION")
        logger.info("=" * 60)
        
        # Step 1: Test database connection
        if not test_database_connection():
            logger.error("Cannot proceed - database connection failed")
            return False
        
        # Step 2: Initialize database (create tables if needed)
        if not initialize_database():
            logger.error("Cannot proceed - database initialization failed")
            return False
        
        # Step 3: Populate with data
        populator = DatabasePopulator(batch_size=batch_size)
        if not populator.populate_keywords_from_json(json_file_path, clear_existing):
            logger.error("Database population failed")
            return False
        
        # Step 4: Verify population
        stats = populator.verify_database_population()
        if 'error' in stats:
            logger.warning(f"Verification had issues: {stats['error']}")
        else:
            logger.info("Database population verification:")
            logger.info(f"  Total keywords: {stats['total_keywords']}")
            logger.info(f"  Active keywords: {stats['active_keywords']}")
            logger.info(f"  Tier distribution: {stats['tier_counts']}")
            for tier, samples in stats['samples'].items():
                tier_name = {1: "High", 2: "Medium", 3: "Low"}[tier]
                logger.info(f"  Tier {tier} ({tier_name}) samples: {samples[:3]}")
        
        logger.info("=" * 60)
        logger.info("✅ DATABASE POPULATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in populate_database_from_json: {e}")
        return False 