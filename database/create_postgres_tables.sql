-- PostgreSQL table creation script for content_moderation database
-- Run this script to create the required database schema

-- Connect to content_moderation database before running this script
-- psql -U postgres -d content_moderation

-- Drop table if exists (for clean recreation)
DROP TABLE IF EXISTS posts;

-- Create posts table with PostgreSQL schema
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    accepted BOOLEAN NOT NULL,
    reason TEXT NOT NULL,
    threat_level VARCHAR(50) DEFAULT 'low',
    confidence VARCHAR(20) DEFAULT '1.0',
    stage VARCHAR(50) DEFAULT 'unknown',
    band VARCHAR(50) DEFAULT 'SAFE',
    action VARCHAR(50) DEFAULT 'PASS',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_time DECIMAL(10,3) DEFAULT 0.0,
    rulebased_time DECIMAL(10,3) DEFAULT 0.0,
    lgbm_time DECIMAL(10,3) DEFAULT 0.0,
    detoxify_time DECIMAL(10,3) DEFAULT 0.0,
    finbert_time DECIMAL(10,3) DEFAULT 0.0,
    llm_time DECIMAL(10,3) DEFAULT 0.0,
    override VARCHAR(3) DEFAULT 'No',
    llm_explanation TEXT DEFAULT '',
    llm_troublesome_words TEXT DEFAULT '',
    llm_suggestion TEXT DEFAULT '',
    comments VARCHAR(500)
);

-- Create index on id column
CREATE INDEX idx_posts_id ON posts(id);

-- Verify table creation
SELECT table_name, column_name, data_type, is_nullable, column_default 
FROM information_schema.columns 
WHERE table_name = 'posts' 
ORDER BY ordinal_position; 