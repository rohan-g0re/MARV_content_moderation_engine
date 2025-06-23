#!/usr/bin/env python3
"""
Content Moderation Engine - Synthetic Data Generation Script

This script generates diverse social media posts using predefined templates
and pattern combinations to create a large, balanced dataset without relying
on external APIs. Designed to run for hours and generate thousands of samples.

Usage:
    python scripts/generate_data_synthetic.py

Output:
    data/raw/synthetic_dataset.csv with columns: post, label
"""

import os
import sys
import time
import csv
import logging
import random
import re
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import itertools

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from dotenv import load_dotenv
    import pandas as pd
except ImportError as e:
    print(f"Missing required packages. Please install: {e}")
    print("Run: pip install python-dotenv pandas")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configuration
@dataclass
class Config:
    target_samples: int = 10000
    output_file: str = "data/raw/synthetic_dataset.csv"
    log_file: str = "logs/synthetic_generation.log"
    checkpoint_interval: int = 1000
    checkpoint_file: str = "data/raw/.synthetic_checkpoint.csv"

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

class ContentTemplates:
    """Predefined content templates for generating synthetic posts"""
    
    # Safe content templates
    SAFE_TEMPLATES = [
        "Just finished reading {book_title} and I loved it! {positive_emotion}",
        "Great weather today for {activity}. Anyone else enjoying the {weather}?",
        "Learning about {topic} in {subject} class. It's fascinating how {fact}.",
        "My {pet_type} {pet_name} just {cute_action}. So adorable! {emoji}",
        "Excited to announce that I {achievement}! Thank you everyone for your support.",
        "Does anyone have recommendations for {item_type}? Looking for something {quality}.",
        "Happy {holiday}! Hope everyone is {activity} with family and friends.",
        "Just watched {movie_title} and it was {positive_adjective}! {rating}/10",
        "Working on my {hobby} project. Here's my progress so far: {description}",
        "Grateful for {thing} today. Sometimes we need to appreciate {value}."
    ]
    
    # Questionable content templates  
    QUESTIONABLE_TEMPLATES = [
        "I can't believe {politician} said that. Politics is getting crazy these days.",
        "This {controversial_topic} debate is getting heated. Where do you stand?",
        "Ugh, {complaint} again. This is getting {mild_profanity} annoying.",
        "Hot take: {controversial_opinion}. Probably unpopular but whatever.",
        "{mild_profanity}, why is {service} so {negative_adjective} lately?",
        "Am I the only one who thinks {controversial_statement}?",
        "This {event} controversy is {emotion}. People need to {action}.",
        "Honestly, {group} really needs to {criticism}. Just saying.",
        "Why do people {annoying_behavior}? It's {emotion} annoying.",
        "Unpopular opinion: {controversial_view}. Fight me."
    ]
    
    # Inappropriate content templates
    INAPPROPRIATE_TEMPLATES = [
        "{strong_profanity} this {thing}. I'm done with this {expletive}.",
        "Anyone else think {group} are complete {insult}? Just me?",
        "{sexual_reference} {inappropriate_comment}. {vulgar_emoji}",
        "This {event} is {strong_profanity} stupid. Who {insult} thought this was good?",
        "{profanity_string} I hate {thing} so much. Makes me want to {violent_metaphor}.",
        "Looking for {adult_content}. DM me if you have {inappropriate_request}.",
        "{curse_word} off to everyone who {action}. You're all {insult}.",
        "Why are {group} such {derogatory_term}? Seriously {expletive} annoying.",
        "NSFW: {adult_reference} {inappropriate_content}",
        "{strong_curse} everything about {situation}. Complete {vulgar_noun}."
    ]
    
    # Harmful content templates
    HARMFUL_TEMPLATES = [
        "People like {target_group} should {harmful_suggestion}. They're ruining {thing}.",
        "I know where {person} lives. Time to {threatening_action}.",
        "Here's {person}'s personal info: {fake_details}. Do what you want with it.",
        "{target_group} are {hate_speech_term} and don't deserve {rights}.",
        "Spreading this {misinformation} because people need to know the truth about {topic}.",
        "If you see {person}, {threatening_instruction}. They deserve it.",
        "DIY {dangerous_activity}: {harmful_instructions} (Don't blame me if {consequence})",
        "{hate_symbol} {group} will never {goal}. We will {threatening_action}.",
        "Secret method to {illegal_activity}: {dangerous_instructions}",
        "Everyone should {harmful_action} against {target_group}. They're {dehumanizing_term}."
    ]
    
    # Illegal content templates
    ILLEGAL_TEMPLATES = [
        "Selling {illegal_substance} cheap. DM for {transaction_details}. Discreet delivery.",
        "I'm going to {specific_threat} {target} at {location} on {date}.",
        "How to make {weapon_explosive}: {dangerous_instructions}. Use responsibly.",
        "Looking for {illegal_service}. Will pay {amount}. No questions asked.",
        "Here's {victim}'s address: {address}. Time to {violent_action}.",
        "Join us in {illegal_activity} against {target}. Bring {weapon_type}.",
        "Free {illegal_content}! Download here: {malicious_link}",
        "I know {target} will be at {location} alone. Perfect time for {criminal_intent}.",
        "Hiring for {illegal_job}. Must be willing to {criminal_activity}. Good pay.",
        "How to {criminal_activity} without getting caught: {illegal_instructions}"
    ]

class ContentVariables:
    """Variables and word lists for template substitution"""
    
    VARIABLES = {
        # Safe content variables
        'book_title': ['The Great Gatsby', 'To Kill a Mockingbird', 'Dune', 'Pride and Prejudice'],
        'positive_emotion': ['Amazing!', 'So good!', 'Loved it!', '🔥', '❤️'],
        'activity': ['hiking', 'running', 'reading', 'gardening', 'cooking'],
        'weather': ['sunshine', 'breeze', 'clear skies', 'warmth', 'cool air'],
        'topic': ['quantum physics', 'ancient history', 'marine biology', 'space exploration'],
        'subject': ['science', 'history', 'literature', 'mathematics', 'philosophy'],
        'fact': ['interconnected everything is', 'complex the systems are', 'beautiful nature is'],
        'pet_type': ['dog', 'cat', 'bird', 'hamster', 'rabbit'],
        'pet_name': ['Max', 'Luna', 'Charlie', 'Bella', 'Rocky'],
        'cute_action': ['learned a new trick', 'fell asleep in my lap', 'brought me their toy'],
        'emoji': ['😍', '🥰', '😊', '❤️', '🐕'],
        'achievement': ['got promoted', 'graduated', 'ran my first marathon', 'published my paper'],
        'item_type': ['books', 'restaurants', 'movies', 'gadgets', 'apps'],
        'quality': ['affordable', 'reliable', 'highly rated', 'beginner-friendly'],
        'holiday': ['birthday', 'New Year', 'graduation day', 'anniversary'],
        'movie_title': ['Inception', 'The Matrix', 'Interstellar', 'Parasite'],
        'positive_adjective': ['amazing', 'fantastic', 'brilliant', 'outstanding'],
        'rating': ['9', '8', '10', '7'],
        'hobby': ['photography', 'painting', 'coding', 'woodworking'],
        'description': ['coming along nicely', 'almost finished', 'more complex than expected'],
        'thing': ['good friends', 'sunny weather', 'opportunities', 'health'],
        'value': ['small moments', 'what we have', 'our relationships'],
        
        # Questionable content variables
        'politician': ['the president', 'the senator', 'the mayor', 'that politician'],
        'controversial_topic': ['immigration', 'healthcare', 'tax policy', 'education reform'],
        'mild_profanity': ['damn', 'crap', 'hell', 'BS'],
        'controversial_opinion': ['pineapple belongs on pizza', 'remote work is overrated'],
        'complaint': ['traffic', 'weather', 'my boss', 'this app'],
        'service': ['customer service', 'public transport', 'internet', 'delivery'],
        'negative_adjective': ['terrible', 'awful', 'broken', 'useless'],
        'controversial_statement': ['social media is toxic', 'TV news is biased'],
        'event': ['election', 'policy change', 'court decision', 'news story'],
        'emotion': ['frustrating', 'concerning', 'disappointing', 'annoying'],
        'action': ['wake up', 'get informed', 'take action', 'speak up'],
        'group': ['politicians', 'celebrities', 'influencers', 'corporations'],
        'criticism': ['be more transparent', 'listen to people', 'do better'],
        'annoying_behavior': ['cut in line', 'talk loudly on phones', 'ignore rules'],
        'controversial_view': ['college is overrated', 'social media should be regulated'],
        
        # Inappropriate content variables (sanitized)
        'strong_profanity': ['[PROFANITY]', '[CURSE]', '[EXPLETIVE]'],
        'insult': ['[INSULT]', '[DEROGATORY_TERM]', '[OFFENSIVE_WORD]'],
        'sexual_reference': ['[SEXUAL_CONTENT]', '[ADULT_REFERENCE]'],
        'inappropriate_comment': ['[VULGAR_CONTENT]', '[SEXUAL_JOKE]'],
        'vulgar_emoji': ['[INAPPROPRIATE_EMOJI]'],
        'expletive': ['garbage', 'trash', 'nonsense'],
        'profanity_string': ['[MULTIPLE_CURSE_WORDS]'],
        'thing': ['situation', 'policy', 'decision', 'rule'],
        'violent_metaphor': ['scream', 'give up', 'complain more'],
        'adult_content': ['[ADULT_CONTENT]'],
        'inappropriate_request': ['[INAPPROPRIATE_REQUEST]'],
        'curse_word': ['[CURSE_WORD]'],
        'derogatory_term': ['[DEROGATORY_TERM]'],
        'adult_reference': ['[ADULT_REFERENCE]'],
        'inappropriate_content': ['[INAPPROPRIATE_CONTENT]'],
        'strong_curse': ['[STRONG_PROFANITY]'],
        'situation': ['policy', 'decision', 'situation', 'rule'],
        'vulgar_noun': ['[VULGAR_TERM]'],
        
        # Harmful content variables (sanitized)
        'target_group': ['[TARGET_GROUP]', '[DEMOGRAPHIC]'],
        'harmful_suggestion': ['[HARMFUL_SUGGESTION]'],
        'threatening_action': ['[THREATENING_ACTION]'],
        'person': ['[PERSON_NAME]', '[INDIVIDUAL]'],
        'fake_details': ['[PERSONAL_INFO]'],
        'hate_speech_term': ['[HATE_SPEECH]'],
        'rights': ['basic rights', 'equal treatment'],
        'misinformation': ['[FALSE_INFORMATION]'],
        'threatening_instruction': ['[THREATENING_INSTRUCTION]'],
        'dangerous_activity': ['[DANGEROUS_ACTIVITY]'],
        'harmful_instructions': ['[HARMFUL_INSTRUCTIONS]'],
        'consequence': ['[NEGATIVE_CONSEQUENCE]'],
        'hate_symbol': ['[HATE_SYMBOL]'],
        'goal': ['[GOAL]'],
        'illegal_activity': ['[ILLEGAL_ACTIVITY]'],
        'dangerous_instructions': ['[DANGEROUS_INSTRUCTIONS]'],
        'harmful_action': ['[HARMFUL_ACTION]'],
        'dehumanizing_term': ['[DEHUMANIZING_TERM]'],
        
        # Illegal content variables (sanitized)
        'illegal_substance': ['[ILLEGAL_SUBSTANCE]'],
        'transaction_details': ['[TRANSACTION_DETAILS]'],
        'specific_threat': ['[SPECIFIC_THREAT]'],
        'target': ['[TARGET]'],
        'location': ['[LOCATION]'],
        'date': ['[DATE]'],
        'weapon_explosive': ['[WEAPON/EXPLOSIVE]'],
        'illegal_service': ['[ILLEGAL_SERVICE]'],
        'amount': ['[AMOUNT]'],
        'victim': ['[VICTIM]'],
        'address': ['[ADDRESS]'],
        'violent_action': ['[VIOLENT_ACTION]'],
        'weapon_type': ['[WEAPON_TYPE]'],
        'illegal_content': ['[ILLEGAL_CONTENT]'],
        'malicious_link': ['[MALICIOUS_LINK]'],
        'criminal_intent': ['[CRIMINAL_INTENT]'],
        'illegal_job': ['[ILLEGAL_JOB]'],
        'criminal_activity': ['[CRIMINAL_ACTIVITY]'],
        'illegal_instructions': ['[ILLEGAL_INSTRUCTIONS]']
    }

class SyntheticDataGenerator:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.generated_posts = []
        self.templates = ContentTemplates()
        self.variables = ContentVariables.VARIABLES
        
    def load_checkpoint(self) -> List[Dict]:
        """Load existing checkpoint if available"""
        checkpoint_path = Path(self.config.checkpoint_file)
        if checkpoint_path.exists():
            try:
                df = pd.read_csv(checkpoint_path)
                posts = df.to_dict('records')
                self.logger.info(f"Loaded checkpoint with {len(posts)} existing posts")
                return posts
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
        return []
    
    def save_checkpoint(self, posts: List[Dict]) -> None:
        """Save checkpoint for resume capability"""
        try:
            checkpoint_path = Path(self.config.checkpoint_file)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame(posts)
            df.to_csv(checkpoint_path, index=False)
            self.logger.info(f"Checkpoint saved with {len(posts)} posts")
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def substitute_template(self, template: str) -> str:
        """Substitute variables in template with random values"""
        result = template
        
        # Find all variables in template
        variables_in_template = re.findall(r'\{(\w+)\}', template)
        
        for var in variables_in_template:
            if var in self.variables:
                value = random.choice(self.variables[var])
                result = result.replace(f'{{{var}}}', value)
            else:
                # Keep placeholder if variable not found
                result = result.replace(f'{{{var}}}', f'[{var.upper()}]')
        
        return result
    
    def add_variation(self, post: str) -> str:
        """Add random variations to make posts more diverse"""
        # Simple variations
        if random.random() < 0.1:
            post = post + '!'
        if random.random() < 0.05:
            post = post.upper()
        if random.random() < 0.1:
            post = post.replace('.', '...')
        
        return post
    
    def generate_posts_by_label(self, label: str, count: int) -> List[Dict]:
        """Generate posts for a specific label"""
        posts = []
        
        if label == 'SAFE':
            templates = self.templates.SAFE_TEMPLATES
        elif label == 'QUESTIONABLE':
            templates = self.templates.QUESTIONABLE_TEMPLATES
        elif label == 'INAPPROPRIATE':
            templates = self.templates.INAPPROPRIATE_TEMPLATES
        elif label == 'HARMFUL':
            templates = self.templates.HARMFUL_TEMPLATES
        elif label == 'ILLEGAL':
            templates = self.templates.ILLEGAL_TEMPLATES
        else:
            self.logger.error(f"Unknown label: {label}")
            return []
        
        for i in range(count):
            template = random.choice(templates)
            post_text = self.substitute_template(template)
            post_text = self.add_variation(post_text)
            
            posts.append({
                'post': post_text,
                'label': label
            })
        
        return posts
    
    def calculate_distribution(self, total_posts: int) -> Dict[str, int]:
        """Calculate balanced distribution across labels"""
        distribution = {
            'SAFE': int(total_posts * 0.35),          # 35%
            'QUESTIONABLE': int(total_posts * 0.25),   # 25%
            'INAPPROPRIATE': int(total_posts * 0.20),  # 20%
            'HARMFUL': int(total_posts * 0.15),        # 15%
            'ILLEGAL': int(total_posts * 0.05)         # 5%
        }
        
        # Adjust for rounding errors
        current_total = sum(distribution.values())
        if current_total < total_posts:
            distribution['SAFE'] += total_posts - current_total
            
        return distribution
    
    def generate_dataset(self) -> bool:
        """Generate the complete synthetic dataset"""
        self.logger.info(f"Starting synthetic dataset generation with target: {self.config.target_samples} posts")
        
        # Load existing checkpoint
        self.generated_posts = self.load_checkpoint()
        already_generated = len(self.generated_posts)
        remaining_posts = self.config.target_samples - already_generated
        
        if remaining_posts <= 0:
            self.logger.info("Target already reached with existing checkpoint")
            return True
        
        self.logger.info(f"Need to generate {remaining_posts} more posts")
        
        # Calculate distribution for remaining posts
        distribution = self.calculate_distribution(remaining_posts)
        
        self.logger.info(f"Distribution for remaining posts: {distribution}")
        
        # Generate posts for each label
        for label, count in distribution.items():
            if count > 0:
                self.logger.info(f"Generating {count} posts for label: {label}")
                
                # Generate in smaller batches to save checkpoints
                batch_size = min(200, count)
                generated_for_label = 0
                
                while generated_for_label < count:
                    current_batch_size = min(batch_size, count - generated_for_label)
                    batch_posts = self.generate_posts_by_label(label, current_batch_size)
                    
                    self.generated_posts.extend(batch_posts)
                    generated_for_label += current_batch_size
                    
                    # Progress update
                    total_generated = len(self.generated_posts)
                    progress = (total_generated / self.config.target_samples) * 100
                    self.logger.info(f"Progress: {total_generated}/{self.config.target_samples} ({progress:.1f}%)")
                    
                    # Save checkpoint periodically
                    if total_generated % self.config.checkpoint_interval == 0:
                        self.save_checkpoint(self.generated_posts)
                    
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.01)
        
        self.logger.info(f"Synthetic dataset generation completed: {len(self.generated_posts)} posts")
        return True
    
    def save_dataset(self) -> bool:
        """Save generated dataset to CSV"""
        try:
            # Ensure output directory exists
            output_path = Path(self.config.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Shuffle the posts for better distribution
            random.shuffle(self.generated_posts)
            
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
            
            # Clean up checkpoint file
            checkpoint_path = Path(self.config.checkpoint_file)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                self.logger.info("Checkpoint file cleaned up")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save dataset: {e}")
            return False

def main():
    """Main execution function"""
    config = Config(
        target_samples=int(os.getenv('DATASET_SIZE_SYNTHETIC', 10000))
    )
    
    # Setup logging
    logger = setup_logging(config.log_file)
    
    # Generate dataset
    generator = SyntheticDataGenerator(config, logger)
    
    logger.info("="*50)
    logger.info("CONTENT MODERATION ENGINE - SYNTHETIC DATA GENERATION")
    logger.info("="*50)
    
    try:
        success = generator.generate_dataset()
        
        if success:
            success = generator.save_dataset()
            
        if success:
            logger.info("✅ Synthetic dataset generation completed successfully!")
            logger.info(f"📁 Output file: {config.output_file}")
            logger.info(f"📊 Total samples: {len(generator.generated_posts)}")
        else:
            logger.error("❌ Dataset generation failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user. Saving checkpoint...")
        generator.save_checkpoint(generator.generated_posts)
        logger.info("Checkpoint saved. You can resume later by running the script again.")
        sys.exit(0)

if __name__ == "__main__":
    main() 