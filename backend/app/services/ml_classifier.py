"""
ML Classification service for content moderation.
Uses unitary/toxic-bert model to classify words by toxicity/severity.
"""

import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import warnings

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToxicityClassifier:
    """
    Toxicity classifier using unitary/toxic-bert model.
    Classifies words into 3 severity tiers based on toxicity scores.
    """
    
    def __init__(self, model_name: str = "unitary/toxic-bert", batch_size: int = 32):
        """
        Initialize the toxicity classifier.
        
        Args:
            model_name (str): HuggingFace model name
            batch_size (int): Batch size for processing
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = None
        self.tokenizer = None
        self.model = None
        
        logger.info(f"Initializing ToxicityClassifier with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
    def load_model(self) -> bool:
        """
        Load the toxic-bert model and tokenizer.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Loading tokenizer and model...")
            
            # Load tokenizer and model separately for better control
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline for easier inference
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _extract_toxicity_score(self, model_output: List[Dict]) -> float:
        """
        Extract toxicity score from model output.
        toxic-bert returns multiple toxicity categories, we'll use a composite score.
        
        Args:
            model_output (List[Dict]): Model prediction output
            
        Returns:
            float: Composite toxicity score between 0 and 1
        """
        try:
            # toxic-bert returns multiple categories: toxic, severe_toxic, obscene, threat, insult, identity_hate
            # We'll calculate a composite score using weighted combination
            
            score_weights = {
                'toxic': 1.0,           # Base toxicity
                'severe_toxic': 1.5,    # More severe
                'obscene': 0.8,         # Moderately weighted
                'threat': 1.2,          # High weight for threats
                'insult': 0.6,          # Lower weight
                'identity_hate': 1.3    # High weight for hate
            }
            
            composite_score = 0.0
            total_weight = 0.0
            
            for label_score in model_output:
                label = label_score['label']
                score = label_score['score']
                
                if label in score_weights:
                    weight = score_weights[label]
                    composite_score += score * weight
                    total_weight += weight
            
            # Normalize by total weight to keep score between 0-1
            if total_weight > 0:
                normalized_score = composite_score / total_weight
                return min(normalized_score, 1.0)  # Cap at 1.0
            else:
                logger.warning("No recognized toxicity labels found in model output")
                return 0.0
            
        except Exception as e:
            logger.error(f"Error extracting toxicity score: {e}")
            return 0.0
    
    def _score_to_tier(self, toxicity_score: float) -> int:
        """
        Convert toxicity score to severity tier.
        
        Args:
            toxicity_score (float): Toxicity score between 0 and 1
            
        Returns:
            int: Tier (1=High, 2=Medium, 3=Low severity)
        """
        # Tier mapping based on composite toxicity score
        # Adjusted thresholds based on toxic-bert's typical score ranges
        if toxicity_score >= 0.5:
            return 1  # High severity - immediate blocking
        elif toxicity_score >= 0.2:
            return 2  # Medium severity - requires review
        else:
            return 3  # Low severity - monitoring only
    
    def _classify_batch(self, words_batch: List[str]) -> List[Dict]:
        """
        Classify a batch of words.
        
        Args:
            words_batch (List[str]): Batch of words to classify
            
        Returns:
            List[Dict]: Classification results for the batch
        """
        try:
            # Get predictions for the batch
            predictions = self.classifier(words_batch)
            
            results = []
            for i, word in enumerate(words_batch):
                toxicity_score = self._extract_toxicity_score(predictions[i])
                tier = self._score_to_tier(toxicity_score)
                
                # Convert to 0-100 scale for database storage
                severity_score = int(toxicity_score * 100)
                
                results.append({
                    'word': word,
                    'toxicity_score': toxicity_score,
                    'severity_score': severity_score,
                    'tier': tier
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error classifying batch: {e}")
            # Return default low-severity results for the batch
            return [
                {
                    'word': word,
                    'toxicity_score': 0.0,
                    'severity_score': 0,
                    'tier': 3
                }
                for word in words_batch
            ]
    
    def classify_words(self, words: List[str]) -> List[Dict]:
        """
        Classify a list of words into severity tiers.
        
        Args:
            words (List[str]): List of words to classify
            
        Returns:
            List[Dict]: Classification results with scores and tiers
        """
        if not self.classifier:
            logger.error("Model not loaded. Call load_model() first.")
            return []
        
        if not words:
            logger.warning("Empty words list provided")
            return []
        
        logger.info(f"Classifying {len(words)} words in batches of {self.batch_size}")
        
        all_results = []
        
        # Process words in batches with progress bar
        for i in tqdm(range(0, len(words), self.batch_size), desc="Processing batches"):
            batch = words[i:i + self.batch_size]
            batch_results = self._classify_batch(batch)
            all_results.extend(batch_results)
        
        # Log statistics
        tier_counts = {1: 0, 2: 0, 3: 0}
        for result in all_results:
            tier_counts[result['tier']] += 1
        
        logger.info(f"Classification complete!")
        logger.info(f"Tier 1 (High): {tier_counts[1]} words")
        logger.info(f"Tier 2 (Medium): {tier_counts[2]} words")
        logger.info(f"Tier 3 (Low): {tier_counts[3]} words")
        
        return all_results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dict: Model information
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'batch_size': self.batch_size,
            'model_loaded': self.classifier is not None,
            'cuda_available': torch.cuda.is_available()
        }

def classify_words_from_json(json_file_path: str, batch_size: int = 32) -> List[Dict]:
    """
    Convenience function to classify words directly from JSON file.
    
    Args:
        json_file_path (str): Path to JSON file containing words
        batch_size (int): Batch size for processing
        
    Returns:
        List[Dict]: Classification results
    """
    # Import here to avoid circular imports
    from ..utils.file_operations import load_words_from_json
    
    try:
        # Load words from JSON
        words = load_words_from_json(json_file_path)
        if not words:
            logger.error("No words loaded from JSON file")
            return []
        
        # Initialize classifier
        classifier = ToxicityClassifier(batch_size=batch_size)
        
        # Load model
        if not classifier.load_model():
            logger.error("Failed to load toxicity model")
            return []
        
        # Classify words
        results = classifier.classify_words(words)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in classify_words_from_json: {e}")
        return [] 