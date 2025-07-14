-- Fix confidence column length issue
-- Run this script in pgAdmin or psql

-- Connect to content_moderation database first

-- Option 1: Alter existing table (if it has data you want to keep)
ALTER TABLE posts ALTER COLUMN confidence TYPE VARCHAR(20);

-- Option 2: If table is empty, you can recreate it
-- DROP TABLE IF EXISTS posts;
-- Then run the full create_postgres_tables.sql script

-- Verify the change
SELECT column_name, data_type, character_maximum_length 
FROM information_schema.columns 
WHERE table_name = 'posts' AND column_name = 'confidence'; 