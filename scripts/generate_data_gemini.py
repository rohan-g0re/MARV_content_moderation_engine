#!/usr/bin/env python3
"""
Content Moderation Engine - Enhanced Gemini-Based Data Generation Script

This script generates diverse financial social media posts using Google's Gemini API
with varying user expertise levels and comprehensive reasoning for training
advanced content moderation models.

Features:
- Multi-level user expertise (Naive to Expert across finance domains)
- Word count constraints (40-500 words, 30% long-form)
- Inference logic column for model training
- Optimized Gemini prompts with financial domain focus
- Incremental dataset saving after each attempt

Usage:
    python scripts/generate_data_gemini.py

Output:
    data/raw/gemini_dataset.csv with columns: post, label, inference_logic
"""

import os
import sys
import time
import csv
import logging
import random
from typing import List, Dict, Tuple, Optional
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
    batch_size: int = 25  # Reduced for better quality control
    max_retries: int = 3
    min_word_count: int = 40
    max_word_count: int = 500
    long_form_percentage: float = 0.30  # 30% should be 300+ words
    long_form_threshold: int = 300

# User Expertise Levels and Finance Domains
class ExpertiseProfiles:
    """Defines different user expertise levels and financial domains"""
    
    EXPERTISE_LEVELS = {
        'NAIVE': {
            'percentage': 0.40,  # 40% of users
            'description': 'New to finance, basic questions, simple language',
            'characteristics': ['asks basic questions', 'uses simple language', 'seeks explanations', 'may misunderstand concepts']
        },
        'BEGINNER': {
            'percentage': 0.25,  # 25% of users  
            'description': 'Learning finance, some terminology, occasional errors',
            'characteristics': ['learning terminology', 'some knowledge gaps', 'enthusiastic but cautious', 'follows trends']
        },
        'INTERMEDIATE': {
            'percentage': 0.20,  # 20% of users
            'description': 'Decent understanding, uses jargon, shares opinions',
            'characteristics': ['uses financial jargon', 'shares market opinions', 'references specific stocks/crypto', 'debates strategies']
        },
        'ADVANCED': {
            'percentage': 0.10,  # 10% of users
            'description': 'Strong knowledge, technical analysis, detailed insights',
            'characteristics': ['technical analysis', 'detailed market insights', 'complex strategies', 'mentors others']
        },
        'EXPERT': {
            'percentage': 0.05,  # 5% of users
            'description': 'Professional level, industry insights, sophisticated language',
            'characteristics': ['professional insights', 'industry connections', 'sophisticated analysis', 'regulatory knowledge']
        }
    }
    
    FINANCE_DOMAINS = [
        'STOCKS', 'CRYPTO', 'FOREX', 'BONDS', 'DERIVATIVES', 'REAL_ESTATE',
        'COMMODITIES', 'BANKING', 'INSURANCE', 'FINTECH', 'MACROECONOMICS',
        'PERSONAL_FINANCE', 'INVESTING', 'TRADING', 'RETIREMENT_PLANNING'
    ]

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

# Enhanced Content generation prompt templates
class PromptTemplates:
    SYSTEM_PROMPT = """
You are an advanced content generation system for training financial content moderation AI.
Generate realistic social media posts from financial platform users with varying expertise levels.

CRITICAL OUTPUT FORMAT:
Return ONLY valid JSON array with this EXACT structure:
[
    {
        "post": "post content here",
        "label": "CLASSIFICATION_LABEL", 
        "inference_logic": "brief reasoning for classification"
    }
]

CLASSIFICATION SYSTEM (use exactly these labels):
- SAFE: Professional, educational, appropriate financial discussion
- QUESTIONABLE: Heated debates, mild inappropriate language, controversial but legal opinions
- INAPPROPRIATE: Strong profanity, off-topic spam, unprofessional behavior, personal attacks
- HARMFUL: Financial misinformation, harassment, pump/dump schemes, predatory advice
- ILLEGAL: Market manipulation, insider trading, fraud schemes, doxxing, explicit threats

USER EXPERTISE DISTRIBUTION:
- NAIVE (40%): New to finance, asks basic questions, simple language, may misunderstand
- BEGINNER (25%): Learning terminology, enthusiastic, follows trends, some knowledge gaps  
- INTERMEDIATE (20%): Uses jargon, shares opinions, references specific assets, debates strategies
- ADVANCED (10%): Technical analysis, detailed insights, complex strategies, mentors others
- EXPERT (5%): Professional insights, regulatory knowledge, sophisticated analysis

FINANCIAL DOMAINS TO COVER:
Stocks, Crypto, Forex, Bonds, Derivatives, Real Estate, Commodities, Banking, Insurance, 
FinTech, Macroeconomics, Personal Finance, Investing, Trading, Retirement Planning

STRICT CONSTRAINTS:
1. EVERY post must be 40-500 words (count carefully)
2. Exactly 30% of posts must be 300+ words (long-form content)
3. Maximum 5 hashtags per post, realistic usage
4. inference_logic must be under 20 words, no hashtags
5. Posts must sound like REAL social media content from financial platforms
6. Include realistic typos, abbreviations, and social media language patterns
7. Vary writing styles based on expertise level (naive users vs experts)
8. Cover diverse financial topics appropriate to each expertise level
"""

    USER_PROMPT_TEMPLATE = """
Generate {count} diverse financial social media posts following the expertise distribution and classification targets.

TARGET DISTRIBUTION:
- SAFE: {safe_count} posts
- QUESTIONABLE: {questionable_count} posts  
- INAPPROPRIATE: {inappropriate_count} posts
- HARMFUL: {harmful_count} posts
- ILLEGAL: {illegal_count} posts

WORD COUNT REQUIREMENTS:
- {long_form_count} posts must be 300-500 words (long-form content)
- {short_form_count} posts must be 40-299 words (regular content)
- ALL posts must be within 40-500 word range

EXPERTISE LEVEL TARGETS:
- NAIVE: {naive_count} posts (40%)
- BEGINNER: {beginner_count} posts (25%)
- INTERMEDIATE: {intermediate_count} posts (20%)
- ADVANCED: {advanced_count} posts (10%)
- EXPERT: {expert_count} posts (5%)

Create realistic posts that:
1. Match the expertise level of the supposed author
2. Cover diverse financial topics and scenarios
3. Include realistic social media language, abbreviations, and patterns
4. Provide clear, concise inference_logic for each classification
5. Ensure balanced representation across all categories and expertise levels

Remember: Output ONLY the JSON array, no additional text or explanations.
"""

class GeminiDataGenerator:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.generated_posts = []
        self.total_attempts = 0
        
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

    def calculate_distributions(self, total_posts: int) -> Dict[str, Dict[str, int]]:
        """Calculate balanced distributions for labels, expertise, and word counts"""
        
        # Label distribution (realistic for financial platform)
        label_distribution = {
            'SAFE': int(total_posts * 0.40),          # 40% - Most content should be safe
            'QUESTIONABLE': int(total_posts * 0.25),  # 25% - Heated financial debates
            'INAPPROPRIATE': int(total_posts * 0.20), # 20% - Unprofessional behavior
            'HARMFUL': int(total_posts * 0.10),       # 10% - Misinformation, bad advice
            'ILLEGAL': int(total_posts * 0.05)        # 5% - Fraud, manipulation
        }
        
        # Expertise distribution
        expertise_distribution = {
            'NAIVE': int(total_posts * 0.40),         # 40%
            'BEGINNER': int(total_posts * 0.25),      # 25%
            'INTERMEDIATE': int(total_posts * 0.20),  # 20%
            'ADVANCED': int(total_posts * 0.10),      # 10%
            'EXPERT': int(total_posts * 0.05)         # 5%
        }
        
        # Word count distribution
        long_form_count = int(total_posts * self.config.long_form_percentage)
        short_form_count = total_posts - long_form_count
        
        word_count_distribution = {
            'long_form': long_form_count,    # 300-500 words (30%)
            'short_form': short_form_count   # 40-299 words (70%)
        }
        
        # Adjust for rounding errors
        for dist in [label_distribution, expertise_distribution]:
            current_total = sum(dist.values())
            if current_total < total_posts:
                # Add remainder to the largest category
                largest_key = max(dist.keys(), key=lambda k: dist[k])
                dist[largest_key] += total_posts - current_total
        
        return {
            'labels': label_distribution,
            'expertise': expertise_distribution,
            'word_counts': word_count_distribution
        }

    def validate_post_constraints(self, post_data: Dict) -> bool:
        """Validate that a post meets all constraints"""
        try:
            # Check required fields
            required_fields = ['post', 'label', 'inference_logic']
            if not all(field in post_data for field in required_fields):
                return False
            
            post_text = post_data['post']
            inference_logic = post_data['inference_logic']
            
            # Word count validation for post (40-500 words)
            word_count = len(post_text.split())
            if word_count < self.config.min_word_count or word_count > self.config.max_word_count:
                self.logger.debug(f"Post word count {word_count} outside range {self.config.min_word_count}-{self.config.max_word_count}")
                return False
            
            # Inference logic validation (max 20 words, no hashtags)
            logic_word_count = len(inference_logic.split())
            if logic_word_count > 20 or '#' in inference_logic:
                self.logger.debug(f"Inference logic invalid: {logic_word_count} words, contains hashtags: {'#' in inference_logic}")
                return False
            
            # Hashtag validation (max 5 per post)
            hashtag_count = post_text.count('#')
            if hashtag_count > 5:
                self.logger.debug(f"Too many hashtags: {hashtag_count}")
                return False
            
            # Label validation
            valid_labels = ['SAFE', 'QUESTIONABLE', 'INAPPROPRIATE', 'HARMFUL', 'ILLEGAL']
            if post_data['label'] not in valid_labels:
                self.logger.debug(f"Invalid label: {post_data['label']}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Validation error: {e}")
            return False

    def append_to_dataset(self, new_posts: List[Dict]) -> bool:
        """Append new posts to existing dataset file after each attempt"""
        try:
            output_path = Path(self.config.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file exists and has data
            if output_path.exists() and output_path.stat().st_size > 0:
                # Read existing data
                existing_df = pd.read_csv(output_path)
                # Combine with new data
                new_df = pd.DataFrame(new_posts)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                # Create new file
                combined_df = pd.DataFrame(new_posts)
            
            # Save combined data
            combined_df.to_csv(output_path, index=False)
            
            self.logger.info(f"Appended {len(new_posts)} posts to dataset. Total: {len(combined_df)} posts")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to append to dataset: {e}")
            return False

    def generate_batch(self, batch_count: int, distributions: Dict) -> List[Dict]:
        """Generate a batch of posts using Gemini with all constraints"""
        
        label_dist = distributions['labels']
        expertise_dist = distributions['expertise'] 
        word_count_dist = distributions['word_counts']
        
        prompt = PromptTemplates.USER_PROMPT_TEMPLATE.format(
            count=batch_count,
            safe_count=label_dist['SAFE'],
            questionable_count=label_dist['QUESTIONABLE'],
            inappropriate_count=label_dist['INAPPROPRIATE'],
            harmful_count=label_dist['HARMFUL'],
            illegal_count=label_dist['ILLEGAL'],
            long_form_count=word_count_dist['long_form'],
            short_form_count=word_count_dist['short_form'],
            naive_count=expertise_dist['NAIVE'],
            beginner_count=expertise_dist['BEGINNER'],
            intermediate_count=expertise_dist['INTERMEDIATE'],
            advanced_count=expertise_dist['ADVANCED'],
            expert_count=expertise_dist['EXPERT']
        )
        
        for attempt in range(self.config.max_retries):
            try:
                self.total_attempts += 1
                self.logger.info(f"Generating batch of {batch_count} posts (attempt {attempt + 1}, total attempts: {self.total_attempts})")
                
                # Generate content with enhanced system prompt
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
                
                # Validate each post against constraints
                valid_posts = []
                for post_data in posts_data:
                    if self.validate_post_constraints(post_data):
                        valid_posts.append(post_data)
                    else:
                        self.logger.debug(f"Post failed validation: {post_data}")
                
                if valid_posts:
                    self.logger.info(f"Successfully generated {len(valid_posts)} valid posts out of {len(posts_data)} total")
                    
                    # Append to dataset immediately after each successful attempt
                    self.append_to_dataset(valid_posts)
                    
                    return valid_posts
                else:
                    raise ValueError("No valid posts generated")
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.rate_limit_delay * 2)  # Longer delay on error
                    
        self.logger.error(f"Failed to generate batch after {self.config.max_retries} attempts")
        return []

    def generate_dataset(self) -> bool:
        """Generate the complete dataset with enhanced tracking and validation"""
        self.logger.info(f"Starting enhanced dataset generation with target: {self.config.target_samples} posts")
        
        if not self.initialize_gemini():
            return False
        
        # Calculate how many posts we need
        remaining_posts = self.config.target_samples
        batch_number = 1
        successful_batches = 0
        
        while remaining_posts > 0:
            batch_size = min(self.config.batch_size, remaining_posts)
            distributions = self.calculate_distributions(batch_size)
            
            self.logger.info(f"Batch {batch_number}: Generating {batch_size} posts")
            self.logger.info(f"Target distributions - Labels: {distributions['labels']}")
            self.logger.info(f"Expertise: {distributions['expertise']}")
            self.logger.info(f"Word counts: {distributions['word_counts']}")
            
            # Generate batch
            batch_posts = self.generate_batch(batch_size, distributions)
            
            if batch_posts:
                self.generated_posts.extend(batch_posts)
                remaining_posts -= len(batch_posts)
                successful_batches += 1
                
                # Calculate word count distribution for this batch
                word_counts = [len(post['post'].split()) for post in batch_posts]
                long_form_actual = sum(1 for wc in word_counts if wc >= self.config.long_form_threshold)
                
                self.logger.info(f"Batch {batch_number} completed successfully!")
                self.logger.info(f"Progress: {len(self.generated_posts)}/{self.config.target_samples} posts generated")
                self.logger.info(f"Word count stats - Min: {min(word_counts)}, Max: {max(word_counts)}, Long-form: {long_form_actual}/{len(batch_posts)}")
                
            else:
                self.logger.error(f"Failed to generate batch {batch_number}")
                # If we have some data already, consider partial success
                if len(self.generated_posts) > 0:
                    self.logger.warning(f"Stopping generation early due to failure, but have {len(self.generated_posts)} posts")
                    break
                else:
                    return False  # Complete failure
            
            # Rate limiting
            if remaining_posts > 0:
                self.logger.info(f"Waiting {self.config.rate_limit_delay}s for rate limiting...")
                time.sleep(self.config.rate_limit_delay)
            
            batch_number += 1
        
        # Final statistics
        total_generated = len(self.generated_posts)
        self.logger.info(f"Dataset generation completed: {total_generated} posts from {successful_batches} successful batches")
        self.logger.info(f"Total API attempts made: {self.total_attempts}")
        
        if total_generated > 0:
            self.log_final_statistics()
        
        return total_generated > 0
    
    def log_final_statistics(self):
        """Log comprehensive statistics about the generated dataset"""
        if not self.generated_posts:
            return
        
        # Label distribution
        label_counts = {}
        for post in self.generated_posts:
            label = post['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Word count analysis
        word_counts = [len(post['post'].split()) for post in self.generated_posts]
        long_form_count = sum(1 for wc in word_counts if wc >= self.config.long_form_threshold)
        
        # Hashtag analysis
        hashtag_counts = [post['post'].count('#') for post in self.generated_posts]
        
        # Inference logic analysis
        inference_word_counts = [len(post['inference_logic'].split()) for post in self.generated_posts]
        
        total_posts = len(self.generated_posts)
        
        self.logger.info("="*60)
        self.logger.info("FINAL DATASET STATISTICS")
        self.logger.info("="*60)
        
        self.logger.info(f"Total posts generated: {total_posts}")
        self.logger.info(f"Total API attempts: {self.total_attempts}")
        self.logger.info(f"Success rate: {(total_posts/self.total_attempts)*100:.1f}% (considering failed attempts)")
        
        self.logger.info("\nLabel Distribution:")
        for label, count in sorted(label_counts.items()):
            percentage = (count / total_posts) * 100
            self.logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
        self.logger.info(f"\nWord Count Analysis:")
        self.logger.info(f"  Min words: {min(word_counts)}")
        self.logger.info(f"  Max words: {max(word_counts)}")
        self.logger.info(f"  Average words: {sum(word_counts)/len(word_counts):.1f}")
        self.logger.info(f"  Long-form posts (300+ words): {long_form_count} ({(long_form_count/total_posts)*100:.1f}%)")
        self.logger.info(f"  Target long-form: {self.config.long_form_percentage*100:.1f}%")
        
        self.logger.info(f"\nContent Quality Metrics:")
        self.logger.info(f"  Posts with hashtags: {sum(1 for hc in hashtag_counts if hc > 0)}")
        self.logger.info(f"  Max hashtags in any post: {max(hashtag_counts)}")
        self.logger.info(f"  Average inference logic length: {sum(inference_word_counts)/len(inference_word_counts):.1f} words")
        self.logger.info(f"  Max inference logic length: {max(inference_word_counts)} words")
        
        # Check constraint compliance
        constraint_violations = 0
        for post in self.generated_posts:
            word_count = len(post['post'].split())
            hashtag_count = post['post'].count('#')
            inference_words = len(post['inference_logic'].split())
            
            if (word_count < self.config.min_word_count or 
                word_count > self.config.max_word_count or
                hashtag_count > 5 or
                inference_words > 20 or
                '#' in post['inference_logic']):
                constraint_violations += 1
        
        compliance_rate = ((total_posts - constraint_violations) / total_posts) * 100
        self.logger.info(f"\nConstraint Compliance: {compliance_rate:.1f}% ({constraint_violations} violations)")
        
        self.logger.info("="*60)

    def save_dataset(self) -> bool:
        """Save generated dataset to CSV with enhanced metadata"""
        try:
            # The dataset should already be saved incrementally, 
            # but we'll do a final verification and metadata update
            output_path = Path(self.config.output_file)
            
            if not output_path.exists():
                self.logger.error("Dataset file not found - incremental saving may have failed")
                return False
            
            # Read the current dataset
            df = pd.read_csv(output_path)
            
            # Verify it matches our in-memory data
            if len(df) != len(self.generated_posts):
                self.logger.warning(f"Dataset file has {len(df)} posts but memory has {len(self.generated_posts)}")
            
            # Log final statistics
            label_counts = df['label'].value_counts()
            self.logger.info(f"Final dataset saved to: {output_path}")
            self.logger.info("Final label distribution:")
            for label, count in label_counts.items():
                percentage = (count / len(df)) * 100
                self.logger.info(f"  {label}: {count} ({percentage:.1f}%)")
            
            # Save metadata file
            metadata = {
                'total_posts': len(df),
                'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_api_attempts': self.total_attempts,
                'config': {
                    'model_name': self.config.model_name,
                    'target_samples': self.config.target_samples,
                    'batch_size': self.config.batch_size,
                    'min_word_count': self.config.min_word_count,
                    'max_word_count': self.config.max_word_count,
                    'long_form_percentage': self.config.long_form_percentage
                },
                'label_distribution': label_counts.to_dict(),
                'word_count_stats': {
                    'min': int(df['post'].apply(lambda x: len(x.split())).min()),
                    'max': int(df['post'].apply(lambda x: len(x.split())).max()),
                    'mean': float(df['post'].apply(lambda x: len(x.split())).mean()),
                    'long_form_count': int(df['post'].apply(lambda x: len(x.split()) >= self.config.long_form_threshold).sum())
                }
            }
            
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Dataset metadata saved to: {metadata_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save dataset: {e}")
            return False

def main():
    """Main execution function with enhanced error handling and reporting"""
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
    
    logger.info("="*60)
    logger.info("ENHANCED CONTENT MODERATION ENGINE - GEMINI DATA GENERATION")
    logger.info("="*60)
    logger.info(f"Target samples: {config.target_samples}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Word count range: {config.min_word_count}-{config.max_word_count}")
    logger.info(f"Long-form target: {config.long_form_percentage*100:.1f}% (300+ words)")
    logger.info(f"Rate limit delay: {config.rate_limit_delay}s")
    logger.info("="*60)
    
    start_time = time.time()
    success = generator.generate_dataset()
    end_time = time.time()
    
    # Save and report final results
    if len(generator.generated_posts) > 0:
        save_success = generator.save_dataset()
        
        duration = end_time - start_time
        posts_per_minute = (len(generator.generated_posts) / duration) * 60
        
        if save_success:
            if success:
                logger.info("✅ Enhanced Gemini dataset generation completed successfully!")
            else:
                logger.info("⚠️ Gemini dataset generation incomplete (likely quota limit) but data saved!")
            
            logger.info(f"📁 Output file: {config.output_file}")
            logger.info(f"📊 Total samples: {len(generator.generated_posts)}")
            logger.info(f"⏱️  Generation time: {duration:.1f} seconds")
            logger.info(f"🚀 Generation rate: {posts_per_minute:.1f} posts/minute")
            logger.info(f"🎯 API efficiency: {(len(generator.generated_posts)/generator.total_attempts)*100:.1f}%")
            
        else:
            logger.error("❌ Failed to save dataset!")
            sys.exit(1)
    else:
        logger.error("❌ No data generated!")
        sys.exit(1)

if __name__ == "__main__":
    main() 