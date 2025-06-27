"""
GuardianAI Utility Functions

Consolidated helper functions for the moderation system, including:
- Database operations
- Keyword management  
- Data loading utilities
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_keywords_from_json(file_path: str = "words.json") -> List[str]:
    """
    Load keywords from JSON file
    
    Args:
        file_path: Path to the keywords JSON file
        
    Returns:
        List of keywords
    """
    try:
        path = Path(file_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                keywords = json.load(f)
            logger.info(f"Loaded {len(keywords)} keywords from {file_path}")
            return keywords
        else:
            logger.warning(f"Keywords file not found: {file_path}")
            return ["scammer", "fraud", "hate", "violence", "spam"]
    except Exception as e:
        logger.error(f"Error loading keywords: {e}")
        return ["scammer", "fraud", "hate", "violence", "spam"]

def update_keywords_json(keywords: List[str], file_path: str = "words.json") -> bool:
    """
    Update keywords JSON file - for Day 10 dictionary expansion
    
    Args:
        keywords: List of keywords to save
        file_path: Path to save the keywords
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(keywords, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated keywords file with {len(keywords)} keywords")
        return True
    except Exception as e:
        logger.error(f"Error updating keywords: {e}")
        return False

def add_keywords(new_keywords: List[str], file_path: str = "words.json") -> bool:
    """
    Add new keywords to existing list - for Day 10 dictionary expansion
    
    Args:
        new_keywords: List of new keywords to add
        file_path: Path to the keywords file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        existing_keywords = load_keywords_from_json(file_path)
        combined_keywords = list(set(existing_keywords + new_keywords))  # Remove duplicates
        return update_keywords_json(combined_keywords, file_path)
    except Exception as e:
        logger.error(f"Error adding keywords: {e}")
        return False

def initialize_keywords_database(db_path: str = "database/keywords.db") -> bool:
    """
    Initialize keywords database with SQLite schema
    For future use with Day 10 word embeddings expansion
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create keywords table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL UNIQUE,
                category TEXT DEFAULT 'general',
                severity INTEGER DEFAULT 5,
                source TEXT DEFAULT 'manual',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Create synonyms table for word embeddings expansion
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS synonyms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword_id INTEGER,
                synonym TEXT NOT NULL,
                similarity_score REAL DEFAULT 0.0,
                reviewed BOOLEAN DEFAULT 0,
                approved BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(keyword_id) REFERENCES keywords(id)
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Initialized keywords database at {db_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing keywords database: {e}")
        return False

def load_keywords_from_database(db_path: str = "database/keywords.db") -> List[str]:
    """
    Load active keywords from database
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        List of active keywords
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT word FROM keywords WHERE is_active = 1")
        keywords = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        logger.info(f"Loaded {len(keywords)} keywords from database")
        return keywords
        
    except Exception as e:
        logger.error(f"Error loading keywords from database: {e}")
        return []

def insert_keywords_to_database(keywords: List[Dict[str, Any]], 
                               db_path: str = "database/keywords.db") -> bool:
    """
    Insert keywords into database with metadata
    
    Args:
        keywords: List of keyword dictionaries with metadata
        db_path: Path to the SQLite database
        
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        for keyword_data in keywords:
            cursor.execute("""
                INSERT OR IGNORE INTO keywords (word, category, severity, source)
                VALUES (?, ?, ?, ?)
            """, (
                keyword_data.get('word'),
                keyword_data.get('category', 'general'),
                keyword_data.get('severity', 5),
                keyword_data.get('source', 'manual')
            ))
        
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        
        logger.info(f"Inserted {rows_affected} keywords into database")
        return True
        
    except Exception as e:
        logger.error(f"Error inserting keywords to database: {e}")
        return False

def get_moderation_statistics(db_path: str = "moderation.db") -> Dict[str, Any]:
    """
    Get comprehensive moderation statistics
    
    Args:
        db_path: Path to the moderation database
        
    Returns:
        Dictionary with statistics
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Basic stats
        cursor.execute("SELECT COUNT(*) FROM posts")
        total_posts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM posts WHERE accepted = 1")
        accepted_posts = cursor.fetchone()[0]
        
        # Stage breakdown
        cursor.execute("""
            SELECT stage, COUNT(*) 
            FROM posts 
            WHERE accepted = 0 
            GROUP BY stage
        """)
        stage_stats = dict(cursor.fetchall())
        
        # Threat level breakdown
        cursor.execute("""
            SELECT threat_level, COUNT(*) 
            FROM posts 
            GROUP BY threat_level
        """)
        threat_stats = dict(cursor.fetchall())
        
        # Recent activity (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) 
            FROM posts 
            WHERE created_at > datetime('now', '-24 hours')
        """)
        recent_posts = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_posts": total_posts,
            "accepted_posts": accepted_posts,
            "rejected_posts": total_posts - accepted_posts,
            "acceptance_rate": (accepted_posts / total_posts * 100) if total_posts > 0 else 0,
            "rejection_by_stage": stage_stats,
            "threat_level_distribution": threat_stats,
            "posts_last_24h": recent_posts
        }
        
    except Exception as e:
        logger.error(f"Error getting moderation statistics: {e}")
        return {}

def export_moderation_data(db_path: str = "moderation.db", 
                          output_file: str = "moderation_export.json") -> bool:
    """
    Export moderation data for analysis
    
    Args:
        db_path: Path to the moderation database
        output_file: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, accepted, reason, threat_level, 
                   confidence, stage, created_at
            FROM posts 
            ORDER BY created_at DESC
        """)
        
        posts = []
        for row in cursor.fetchall():
            posts.append({
                "id": row[0],
                "content": row[1],
                "accepted": bool(row[2]),
                "reason": row[3],
                "threat_level": row[4],
                "confidence": float(row[5]) if row[5] else 1.0,
                "stage": row[6],
                "created_at": row[7]
            })
        
        conn.close()
        
        # Export to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "export_timestamp": str(Path().cwd()),
                "total_posts": len(posts),
                "posts": posts
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(posts)} posts to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting moderation data: {e}")
        return False 