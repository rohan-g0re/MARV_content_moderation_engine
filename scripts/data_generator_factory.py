#!/usr/bin/env python3
"""
Content Moderation Engine - Data Generator Factory

This module provides a unified interface for both Gemini and Synthetic data generators
while keeping them separate for prompt tuning and maintenance benefits.

Usage:
    python scripts/data_generator_factory.py --type gemini --samples 1000
    python scripts/data_generator_factory.py --type synthetic --samples 10000
    python scripts/data_generator_factory.py --type combined --samples 15000
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# Import both generators
from generate_data_gemini import GeminiDataGenerator, Config as GeminiConfig
from generate_data_synthetic import SyntheticDataGenerator, Config as SyntheticConfig

@dataclass
class GeneratorConfig:
    """Unified configuration for data generators"""
    generator_type: str  # 'gemini', 'synthetic', 'combined'
    total_samples: int
    gemini_samples: Optional[int] = None
    synthetic_samples: Optional[int] = None
    output_dir: str = "data/raw"
    
    def __post_init__(self):
        if self.generator_type == 'combined':
            if self.gemini_samples is None:
                self.gemini_samples = int(self.total_samples * 0.3)  # 30% Gemini
            if self.synthetic_samples is None:
                self.synthetic_samples = self.total_samples - self.gemini_samples

class DataGeneratorFactory:
    """Factory class for creating and managing data generators"""
    
    def __init__(self, config: GeneratorConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.generators = {}
    
    def create_gemini_generator(self) -> GeminiDataGenerator:
        """Create Gemini data generator"""
        gemini_config = GeminiConfig(
            api_key="",  # Will be loaded from env
            target_samples=self.config.gemini_samples or self.config.total_samples,
            output_file=f"{self.config.output_dir}/gemini_dataset.csv",
            log_file="logs/gemini_generation.log"
        )
        
        generator = GeminiDataGenerator(gemini_config, self.logger)
        self.generators['gemini'] = generator
        return generator
    
    def create_synthetic_generator(self) -> SyntheticDataGenerator:
        """Create Synthetic data generator"""
        synthetic_config = SyntheticConfig(
            target_samples=self.config.synthetic_samples or self.config.total_samples,
            output_file=f"{self.config.output_dir}/synthetic_dataset.csv",
            log_file="logs/synthetic_generation.log"
        )
        
        generator = SyntheticDataGenerator(synthetic_config, self.logger)
        self.generators['synthetic'] = generator
        return generator
    
    def generate_combined_dataset(self) -> bool:
        """Generate combined dataset from both generators"""
        self.logger.info(f"Starting combined dataset generation: {self.config.total_samples} total samples")
        self.logger.info(f"Gemini: {self.config.gemini_samples}, Synthetic: {self.config.synthetic_samples}")
        
        success = True
        
        # Generate Gemini data first (higher quality)
        if self.config.gemini_samples and self.config.gemini_samples > 0:
            self.logger.info("="*50)
            self.logger.info("GENERATING GEMINI DATASET")
            self.logger.info("="*50)
            
            gemini_gen = self.create_gemini_generator()
            if not gemini_gen.generate_dataset():
                self.logger.error("Gemini generation failed!")
                success = False
            else:
                gemini_gen.save_dataset()
        
        # Generate Synthetic data
        if self.config.synthetic_samples and self.config.synthetic_samples > 0:
            self.logger.info("="*50)
            self.logger.info("GENERATING SYNTHETIC DATASET")
            self.logger.info("="*50)
            
            synthetic_gen = self.create_synthetic_generator()
            if not synthetic_gen.generate_dataset():
                self.logger.error("Synthetic generation failed!")
                success = False
            else:
                synthetic_gen.save_dataset()
        
        # Combine datasets if both succeeded
        if success and self.config.generator_type == 'combined':
            success = self.combine_datasets()
        
        return success
    
    def combine_datasets(self) -> bool:
        """Combine Gemini and Synthetic datasets into a single file"""
        try:
            import pandas as pd
            
            gemini_file = f"{self.config.output_dir}/gemini_dataset.csv"
            synthetic_file = f"{self.config.output_dir}/synthetic_dataset.csv"
            combined_file = f"{self.config.output_dir}/combined_dataset.csv"
            
            # Load datasets
            gemini_df = pd.read_csv(gemini_file) if Path(gemini_file).exists() else pd.DataFrame()
            synthetic_df = pd.read_csv(synthetic_file) if Path(synthetic_file).exists() else pd.DataFrame()
            
            # Combine datasets
            combined_df = pd.concat([gemini_df, synthetic_df], ignore_index=True)
            
            # Shuffle the combined dataset
            combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Save combined dataset
            combined_df.to_csv(combined_file, index=False)
            
            # Log statistics
            self.logger.info("="*50)
            self.logger.info("COMBINED DATASET STATISTICS")
            self.logger.info("="*50)
            self.logger.info(f"Total samples: {len(combined_df)}")
            
            if len(gemini_df) > 0:
                self.logger.info(f"Gemini samples: {len(gemini_df)}")
            if len(synthetic_df) > 0:
                self.logger.info(f"Synthetic samples: {len(synthetic_df)}")
            
            # Label distribution
            label_counts = combined_df['label'].value_counts()
            self.logger.info("\nLabel Distribution:")
            for label, count in label_counts.items():
                percentage = (count / len(combined_df)) * 100
                self.logger.info(f"  {label}: {count} ({percentage:.1f}%)")
            
            self.logger.info(f"\nCombined dataset saved to: {combined_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to combine datasets: {e}")
            return False
    
    def generate_dataset(self) -> bool:
        """Generate dataset based on configuration"""
        if self.config.generator_type == 'gemini':
            generator = self.create_gemini_generator()
            success = generator.generate_dataset()
            if success:
                generator.save_dataset()
            return success
            
        elif self.config.generator_type == 'synthetic':
            generator = self.create_synthetic_generator()
            success = generator.generate_dataset()
            if success:
                generator.save_dataset()
            return success
            
        elif self.config.generator_type == 'combined':
            return self.generate_combined_dataset()
        
        else:
            self.logger.error(f"Unknown generator type: {self.config.generator_type}")
            return False

def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/data_generation.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Content Moderation Engine - Data Generator")
    parser.add_argument('--type', choices=['gemini', 'synthetic', 'combined'], 
                       default='combined', help='Generator type')
    parser.add_argument('--samples', type=int, default=15000, 
                       help='Total number of samples to generate')
    parser.add_argument('--gemini-samples', type=int, 
                       help='Number of Gemini samples (for combined mode)')
    parser.add_argument('--synthetic-samples', type=int, 
                       help='Number of Synthetic samples (for combined mode)')
    parser.add_argument('--output-dir', default='data/raw', 
                       help='Output directory for datasets')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Create configuration
    config = GeneratorConfig(
        generator_type=args.type,
        total_samples=args.samples,
        gemini_samples=args.gemini_samples,
        synthetic_samples=args.synthetic_samples,
        output_dir=args.output_dir
    )
    
    # Create factory and generate dataset
    factory = DataGeneratorFactory(config, logger)
    
    logger.info("="*60)
    logger.info("CONTENT MODERATION ENGINE - DATA GENERATOR FACTORY")
    logger.info("="*60)
    logger.info(f"Generator Type: {config.generator_type}")
    logger.info(f"Total Samples: {config.total_samples}")
    
    if config.generator_type == 'combined':
        logger.info(f"Gemini Samples: {config.gemini_samples}")
        logger.info(f"Synthetic Samples: {config.synthetic_samples}")
    
    try:
        success = factory.generate_dataset()
        
        if success:
            logger.info("✅ Dataset generation completed successfully!")
        else:
            logger.error("❌ Dataset generation failed!")
            exit(1)
            
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user.")
        exit(0)

if __name__ == "__main__":
    main() 