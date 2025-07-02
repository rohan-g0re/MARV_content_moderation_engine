#!/usr/bin/env python3
"""
Script to train a custom fraud detection model for the moderation pipeline
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_fraud_dataset():
    """Create a synthetic fraud detection dataset"""
    
    # Fraudulent content examples
    fraud_examples = [
        "New loophole lets you legally double your bank balance in 7 days",
        "High quality counterfeit bills available, ping me for details",
        "Get rich quick with this investment opportunity",
        "Investment opportunity with guaranteed returns",
        "Counterfeit cash operation, undetectable bills",
        "Double your money in 24 hours, guaranteed",
        "Legal loophole to avoid taxes completely",
        "High quality fake money, contact me",
        "Get rich quick scheme, easy money",
        "Investment scam with fake returns",
        "Counterfeit currency for sale",
        "Fake bills that pass all tests",
        "Undetectable counterfeit money",
        "Legal tax evasion methods",
        "Quick money making scheme",
        "Investment fraud opportunity",
        "Fake currency production",
        "Counterfeit bills for sale",
        "Get rich overnight scheme",
        "Fake investment returns"
    ]
    
    # Legitimate content examples
    legitimate_examples = [
        "This is a legitimate business post about investments",
        "I'm looking for investment advice for my retirement",
        "How to invest in index funds for beginners",
        "Best practices for financial planning",
        "Understanding compound interest",
        "Diversification strategies for portfolios",
        "Retirement planning tips",
        "How to save money effectively",
        "Investment strategies for long-term growth",
        "Financial literacy education",
        "Budgeting techniques for families",
        "Understanding mutual funds",
        "How to start investing in stocks",
        "Financial planning for young professionals",
        "Retirement account options",
        "Investment risk management",
        "How to read financial statements",
        "Tax planning strategies",
        "Estate planning basics",
        "Insurance planning for families"
    ]
    
    # Create dataset
    texts = fraud_examples + legitimate_examples
    labels = [1] * len(fraud_examples) + [0] * len(legitimate_examples)  # 1=fraud, 0=legitimate
    
    return Dataset.from_dict({
        "text": texts,
        "label": labels
    })

def train_fraud_model(output_dir="models/fraud-detection-finetuned"):
    """Train a custom fraud detection model"""
    
    logger.info("Creating fraud detection dataset...")
    dataset = create_fraud_dataset()
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.2)
    
    logger.info(f"Training set size: {len(dataset['train'])}")
    logger.info(f"Test set size: {len(dataset['test'])}")
    
    # Load model and tokenizer
    model_name = "distilbert-base-uncased"  # Good balance of speed and performance
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2  # Binary classification: fraud vs legitimate
    )
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train the model
    logger.info("Starting model training...")
    trainer.train()
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate the model
    logger.info("Evaluating model...")
    results = trainer.evaluate()
    logger.info(f"Evaluation results: {results}")
    
    logger.info(f"✅ Model trained and saved to {output_dir}")
    return output_dir

def test_fraud_model(model_path):
    """Test the trained fraud detection model"""
    
    from transformers import pipeline
    
    # Load the trained model
    classifier = pipeline("text-classification", model=model_path)
    
    # Test cases
    test_cases = [
        "New loophole lets you legally double your bank balance in 7 days",
        "High quality counterfeit bills available, ping me for details",
        "This is a legitimate business post about investments",
        "I'm looking for investment advice for my retirement"
    ]
    
    print("Testing trained fraud detection model:")
    print("=" * 50)
    
    for text in test_cases:
        result = classifier(text)
        label = "FRAUD" if result[0]['label'] == 'LABEL_1' else "LEGITIMATE"
        confidence = result[0]['score']
        
        print(f"Text: {text}")
        print(f"Prediction: {label} (confidence: {confidence:.3f})")
        print("-" * 50)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a custom fraud detection model")
    parser.add_argument("--output-dir", default="models/fraud-detection-finetuned", 
                       help="Output directory for the trained model")
    parser.add_argument("--test-only", action="store_true", 
                       help="Only test an existing model")
    
    args = parser.parse_args()
    
    if args.test_only:
        test_fraud_model(args.output_dir)
    else:
        model_path = train_fraud_model(args.output_dir)
        print(f"\nModel training complete! Test it with:")
        print(f"python scripts/train_fraud_model.py --test-only --output-dir {model_path}") 