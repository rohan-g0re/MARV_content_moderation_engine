#!/usr/bin/env python3
"""
Content Moderation Engine - Gemini-Based Data Generation Script

This script generates diverse social media posts using Google's Gemini API
with a stringent system prompt to create a balanced dataset across all
severity levels for training content moderation models.

Usage:
    python scripts/generate_data_gemini.py

Output:
    data/raw/gemini_dataset.csv with columns: post, label
"""

import os
import sys
import time
import csv
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    import pandas as pd
except ImportError as e:
    print(f"Missing required packages. Please install: {e}")
    print("Run: pip install google-generativeai python-dotenv pandas")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configuration
@dataclass
class Config:
    api_key: str
    model_name: str = "gemini-1.5-flash"
    target_samples: int = 5000
    rate_limit_delay: float = 1.0
    output_file: str = "data/raw/gemini_dataset.csv"
    log_file: str = "logs/gemini_generation.log"
    batch_size: int = 50
    max_retries: int = 3

# Setup logging
def setup_logging(log_file: str) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Content generation prompt templates
class PromptTemplates:
    SYSTEM_PROMPT = """
    You are a content generation system for training a content moderation AI. 
    Generate realistic social media posts that span the full spectrum of content appropriateness.
    
    CLASSIFICATION BANDS (use exactly these labels):
    - SAFE: Completely appropriate content (news, hobbies, education, positive interactions)
    - QUESTIONABLE: Mildly inappropriate or concerning (mild profanity, heated debates, controversial opinions)
    - INAPPROPRIATE: Clearly inappropriate but not harmful (strong profanity, adult themes, off-topic spam)
    - HARMFUL: Potentially dangerous content (harassment, threats, misinformation, hate speech)
    - ILLEGAL: Clearly illegal or extremely harmful content (explicit threats, doxxing, illegal activities)
    
    REQUIREMENTS:
    1. Generate exactly the requested number of posts
    2. Ensure realistic variety in length (10-500 characters)
    3. Include diverse topics: politics, sports, gaming, social issues, technology, entertainment
    4. Make content realistic to actual social media posts
    5. Ensure balanced distribution across all 5 categories
    6. Return ONLY valid JSON format
    """
    
    USER_PROMPT_TEMPLATE = """
    Generate {count} diverse social media posts with their appropriate classification labels.
    Target distribution:
    - SAFE: {safe_count} posts
    - QUESTIONABLE: {questionable_count} posts  
    - INAPPROPRIATE: {inappropriate_count} posts
    - HARMFUL: {harmful_count} posts
    - ILLEGAL: {illegal_count} posts
    
    Return as JSON array with this exact format:
    [
        {{"post": "example post text here", "label": "SAFE"}},
        {{"post": "another post text", "label": "QUESTIONABLE"}}
    ]
    
    Ensure posts are realistic, varied in length, and cover diverse topics.
    """

class GeminiDataGenerator:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.generated_posts = []
        
    def initialize_gemini(self) -> bool:
        """Initialize Gemini API connection"""
        try:
            genai.configure(api_key=self.config.api_key)
            self.model = genai.GenerativeModel(self.config.model_name)
            self.logger.info(f"Successfully initialized Gemini model: {self.config.model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini: {e}")
            return False
    
    def calculate_distribution(self, total_posts: int) -> Dict[str, int]:
        """Calculate balanced distribution across labels"""
        # Aim for realistic distribution with more safe content
        distribution = {
            'SAFE': int(total_posts * 0.35),          # 35% - Most content should be safe
            'QUESTIONABLE': int(total_posts * 0.25),   # 25% - Borderline content
            'INAPPROPRIATE': int(total_posts * 0.20),  # 20% - Clearly inappropriate
            'HARMFUL': int(total_posts * 0.15),        # 15% - Harmful content
            'ILLEGAL': int(total_posts * 0.05)         # 5% - Extreme content
        }
        
        # Adjust for rounding errors
        current_total = sum(distribution.values())
        if current_total < total_posts:
            distribution['SAFE'] += total_posts - current_total
            
        return distribution
    
    def generate_batch(self, batch_count: int, distribution: Dict[str, int]) -> List[Dict]:
        """Generate a batch of posts using Gemini"""
        prompt = PromptTemplates.USER_PROMPT_TEMPLATE.format(
            count=batch_count,
            safe_count=distribution['SAFE'],
            questionable_count=distribution['QUESTIONABLE'],
            inappropriate_count=distribution['INAPPROPRIATE'],
            harmful_count=distribution['HARMFUL'],
            illegal_count=distribution['ILLEGAL']
        )
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.info(f"Generating batch of {batch_count} posts (attempt {attempt + 1})")
                
                # Generate content with system prompt
                chat = self.model.start_chat(history=[])
                response = chat.send_message([
                    PromptTemplates.SYSTEM_PROMPT,
                    prompt
                ])
                
                # Parse JSON response
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3]
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3]
                
                posts_data = json.loads(response_text)
                
                # Validate structure
                if not isinstance(posts_data, list):
                    raise ValueError("Response is not a list")
                
                valid_posts = []
                for post_data in posts_data:
                    if isinstance(post_data, dict) and 'post' in post_data and 'label' in post_data:
                        # Validate label
                        if post_data['label'] in ['SAFE', 'QUESTIONABLE', 'INAPPROPRIATE', 'HARMFUL', 'ILLEGAL']:
                            valid_posts.append(post_data)
                
                self.logger.info(f"Successfully generated {len(valid_posts)} valid posts")
                return valid_posts
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.rate_limit_delay * 2)  # Longer delay on error
                    
        self.logger.error(f"Failed to generate batch after {self.config.max_retries} attempts")
        return []
    
    def generate_dataset(self) -> bool:
        """Generate the complete dataset"""
        self.logger.info(f"Starting dataset generation with target: {self.config.target_samples} posts")
        
        if not self.initialize_gemini():
            return False
        
        # Calculate how many posts we need
        remaining_posts = self.config.target_samples
        batch_number = 1
        
        while remaining_posts > 0:
            batch_size = min(self.config.batch_size, remaining_posts)
            distribution = self.calculate_distribution(batch_size)
            
            self.logger.info(f"Batch {batch_number}: Generating {batch_size} posts")
            
            # Generate batch
            batch_posts = self.generate_batch(batch_size, distribution)
            
            if batch_posts:
                self.generated_posts.extend(batch_posts)
                remaining_posts -= len(batch_posts)
                self.logger.info(f"Progress: {len(self.generated_posts)}/{self.config.target_samples} posts generated")
            else:
                self.logger.error(f"Failed to generate batch {batch_number}")
                return False
            
            # Rate limiting
            if remaining_posts > 0:
                self.logger.info(f"Waiting {self.config.rate_limit_delay}s for rate limiting...")
                time.sleep(self.config.rate_limit_delay)
            
            batch_number += 1
        
        self.logger.info(f"Dataset generation completed: {len(self.generated_posts)} posts")
        return True
    
    def save_dataset(self) -> bool:
        """Save generated dataset to CSV"""
        try:
            # Ensure output directory exists
            output_path = Path(self.config.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create DataFrame and save
            df = pd.DataFrame(self.generated_posts)
            df.to_csv(output_path, index=False)
            
            # Log statistics
            label_counts = df['label'].value_counts()
            self.logger.info(f"Dataset saved to: {output_path}")
            self.logger.info("Label distribution:")
            for label, count in label_counts.items():
                percentage = (count / len(df)) * 100
                self.logger.info(f"  {label}: {count} ({percentage:.1f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save dataset: {e}")
            return False

def main():
    """Main execution function"""
    # Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment variables")
        print("Please set your Gemini API key in .env file or environment")
        sys.exit(1)
    
    config = Config(
        api_key=api_key,
        target_samples=int(os.getenv('DATASET_SIZE_GEMINI', 5000)),
        rate_limit_delay=float(os.getenv('GEMINI_RATE_LIMIT_DELAY', 1.0))
    )
    
    # Setup logging
    logger = setup_logging(config.log_file)
    
    # Generate dataset
    generator = GeminiDataGenerator(config, logger)
    
    logger.info("="*50)
    logger.info("CONTENT MODERATION ENGINE - GEMINI DATA GENERATION")
    logger.info("="*50)
    
    success = generator.generate_dataset()
    
    if success:
        success = generator.save_dataset()
        
    if success:
        logger.info("✅ Gemini dataset generation completed successfully!")
        logger.info(f"📁 Output file: {config.output_file}")
        logger.info(f"📊 Total samples: {len(generator.generated_posts)}")
    else:
        logger.error("❌ Dataset generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 