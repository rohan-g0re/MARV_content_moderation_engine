#!/usr/bin/env python3
"""
Comprehensive Content Analysis Tool for GuardianAI Content Moderation System

This single file provides complete bulk content analysis functionality:
- Import posts from Excel/CSV files
- Analyze each post through the moderation pipeline
- Generate detailed output with all metrics
- Create comprehensive visualizations
- Measure performance and escalation frequency
- User-friendly interface with progress tracking
"""

import time
import json
import statistics
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from logger import get_logger
from app.core.moderation import GuardianModerationEngine

logger = get_logger("content_analyzer", "content_analysis.log")

class ContentAnalyzer:
    """Comprehensive content analysis for content moderation system"""
    
    def __init__(self):
        self.moderation_engine = GuardianModerationEngine()
        self.results = []
        self.performance_stats = defaultdict(list)
        self.escalation_stats = defaultdict(int)
        self.model_decision_stats = defaultdict(int)
        self.band_distribution = defaultdict(int)
        self.confidence_data = []
        
        # Supported file formats
        self.supported_formats = ['.csv', '.xlsx', '.xls']
        
        # Output columns mapping
        self.band_mapping = {
            "SAFE": "Safe",
            "FLAG_LOW": "Flag Low", 
            "FLAG_MEDIUM": "Flag Medium",
            "FLAG_HIGH": "Flag High",
            "BLOCK": "Block"
        }
        
        # Model decision mapping
        self.model_mapping = {
            "rule-based": "Rules",
            "detoxify": "ML Model (Detoxify)",
            "finbert": "ML Model (FinBERT)"
        }
    
    def load_content_from_file(self, file_path: str) -> pd.DataFrame:
        """Load content from Excel or CSV file"""
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path_obj.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format. Supported: {self.supported_formats}")
        
        try:
            if path_obj.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:  # Excel files
                df = pd.read_excel(file_path)
            
            logger.info(f"Loaded {len(df)} posts from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise
    
    def analyze_single_post(self, content: str, post_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a single post through the moderation pipeline"""
        start_time = time.time()
        
        try:
            # Run moderation
            result = self.moderation_engine.moderate_content(content)
            processing_time = time.time() - start_time
            
            # Determine final status
            final_status = self._determine_final_status(result)
            
            # Determine which model made the final decision
            decision_model = self.model_mapping.get(result.stage, "Unknown")
            
            # Check if escalated
            escalated = self._is_escalated(result)
            
            analysis_result = {
                "post_id": post_id or f"post_{len(self.results) + 1}",
                "content": content[:200] + "..." if len(content) > 200 else content,
                "full_content": content,
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time": processing_time,
                "final_status": final_status,
                "band": result.band,
                "action": result.action,
                "threat_level": result.threat_level,
                "confidence": result.confidence,
                "stage": result.stage,
                "decision_model": decision_model,
                "reason": result.reason,
                "escalated": escalated,
                "accepted": result.accepted
            }
            
            # Record statistics
            self.performance_stats["processing_times"].append(processing_time)
            self.confidence_data.append({
                "post_id": analysis_result["post_id"],
                "confidence": result.confidence,
                "threat_level": result.threat_level,
                "stage": result.stage,
                "band": result.band
            })
            
            if escalated:
                self.escalation_stats["total_escalations"] += 1
                self.escalation_stats[result.stage] += 1
            
            self.model_decision_stats[decision_model] += 1
            self.band_distribution[result.band] += 1
            
            logger.info(f"Analyzed post {analysis_result['post_id']} - "
                       f"Status: {final_status}, Time: {processing_time:.3f}s, "
                       f"Model: {decision_model}")
            
            return analysis_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error analyzing post {post_id}: {e}")
            return {
                "post_id": post_id or f"post_{len(self.results) + 1}",
                "content": content[:200] + "..." if len(content) > 200 else content,
                "full_content": content,
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time": processing_time,
                "error": str(e),
                "final_status": "Error",
                "band": "ERROR",
                "action": "ERROR",
                "threat_level": "unknown",
                "confidence": 0.0,
                "stage": "error",
                "decision_model": "Error",
                "reason": f"Analysis failed: {e}",
                "escalated": False,
                "accepted": False
            }
    
    def _determine_final_status(self, result) -> str:
        """Determine the final status based on moderation result"""
        if result.band == "SAFE":
            return "Safe"
        elif result.band == "FLAG_LOW":
            return "Flag Low"
        elif result.band == "FLAG_MEDIUM":
            return "Flag Medium"
        elif result.band == "FLAG_HIGH":
            return "Flag High"
        elif result.band == "BLOCK":
            return "Block"
        else:
            return "Unknown"
    
    def _is_escalated(self, result) -> bool:
        """Determine if content was escalated"""
        escalation_indicators = [
            result.threat_level in ["medium", "high"],
            result.confidence < 0.7,
            result.stage in ["detoxify", "finbert"],
            result.action in ["FLAG_LOW", "FLAG_MEDIUM", "FLAG_HIGH", "BLOCK"]
        ]
        return any(escalation_indicators)
    
    def analyze_bulk_content(self, file_path: str, content_column: str = "content", 
                           id_column: Optional[str] = None) -> pd.DataFrame:
        """Analyze bulk content from file"""
        logger.info(f"Starting bulk content analysis from {file_path}")
        
        # Load data
        df = self.load_content_from_file(file_path)
        
        # Validate columns
        if content_column not in df.columns:
            raise ValueError(f"Content column '{content_column}' not found in file. Available columns: {list(df.columns)}")
        
        if id_column and id_column not in df.columns:
            logger.warning(f"ID column '{id_column}' not found, using auto-generated IDs")
            id_column = None
        
        # Analyze each post
        analysis_results = []
        total_posts = len(df)
        
        print(f"📊 Analyzing {total_posts} posts...")
        
        for index, row in df.iterrows():
            content = str(row[content_column])
            post_id = str(row[id_column]) if id_column else None
            
            # Skip empty content
            if not content or content.strip() == "":
                logger.warning(f"Skipping empty content at row {index + 1}")
                continue
            
            result = self.analyze_single_post(content, post_id)
            analysis_results.append(result)
            
            # Progress indicator
            if (index + 1) % 10 == 0 or (index + 1) == total_posts:
                print(f"   Processed {index + 1}/{total_posts} posts...")
        
        # Create results DataFrame
        results_df = pd.DataFrame(analysis_results)
        
        # Add analysis metadata
        results_df['analysis_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Bulk analysis completed. Processed {len(results_df)} posts")
        
        return results_df
    
    def generate_analysis_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        logger.info("Generating analysis report...")
        
        # Calculate statistics
        total_posts = len(results_df)
        # Check if 'error' column exists and count successful analyses
        if 'error' in results_df.columns:
            successful_analyses = len(results_df[results_df['error'].isna()])
        else:
            successful_analyses = total_posts  # If no error column, assume all successful
        
        # Performance analysis
        processing_times = results_df['processing_time'].dropna()
        performance_stats = {
            "total_posts": total_posts,
            "successful_analyses": successful_analyses,
            "failed_analyses": total_posts - successful_analyses,
            "success_rate": (successful_analyses / total_posts * 100) if total_posts > 0 else 0,
            "avg_processing_time": processing_times.mean() if len(processing_times) > 0 else 0,
            "min_processing_time": processing_times.min() if len(processing_times) > 0 else 0,
            "max_processing_time": processing_times.max() if len(processing_times) > 0 else 0,
            "std_processing_time": processing_times.std() if len(processing_times) > 1 else 0,
            "throughput": (successful_analyses / processing_times.sum()) if processing_times.sum() > 0 else 0
        }
        
        # Escalation analysis
        escalated_posts = results_df[results_df['escalated'] == True]
        escalation_stats = {
            "total_escalations": len(escalated_posts),
            "escalation_rate": (len(escalated_posts) / successful_analyses * 100) if successful_analyses > 0 else 0,
            "escalations_by_model": dict(self.escalation_stats)
        }
        
        # Band distribution
        band_stats = results_df['band'].value_counts().to_dict()
        
        # Model decision analysis
        model_stats = results_df['decision_model'].value_counts().to_dict()
        
        # Confidence analysis
        confidence_stats = results_df['confidence'].describe().to_dict()
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "performance": performance_stats,
            "escalation": escalation_stats,
            "band_distribution": band_stats,
            "model_decisions": model_stats,
            "confidence_stats": confidence_stats,
            "detailed_results": results_df.to_dict('records')
        }
        
        return report
    
    def save_results(self, results_df: pd.DataFrame, output_prefix: str = "content_analysis"):
        """Save analysis results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to Excel
        excel_file = f"{output_prefix}_results_{timestamp}.xlsx"
        results_df.to_excel(excel_file, index=False, engine='openpyxl')
        logger.info(f"Detailed results saved to {excel_file}")
        
        # Save summary to JSON
        report = self.generate_analysis_report(results_df)
        json_file = f"{output_prefix}_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Analysis report saved to {json_file}")
        
        return excel_file, json_file
    
    def create_visualizations(self, results_df: pd.DataFrame, output_prefix: str = "content_analysis"):
        """Create comprehensive visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Band Distribution
        plt.subplot(3, 3, 1)
        band_counts = results_df['band'].value_counts()
        colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1']
        plt.pie(band_counts.values, labels=band_counts.index, autopct='%1.1f%%', colors=colors[:len(band_counts)])
        plt.title('Content Distribution by Band', fontsize=14, fontweight='bold')
        
        # 2. Model Decision Distribution
        plt.subplot(3, 3, 2)
        model_counts = results_df['decision_model'].value_counts()
        plt.bar(model_counts.index, model_counts.values, color='skyblue', alpha=0.7)
        plt.title('Decisions by Model', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylabel('Number of Posts')
        
        # 3. Processing Time Distribution
        plt.subplot(3, 3, 3)
        processing_times = results_df['processing_time'].dropna()
        plt.hist(processing_times, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        plt.title('Processing Time Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Processing Time (seconds)')
        plt.ylabel('Frequency')
        
        # 4. Confidence Score Distribution
        plt.subplot(3, 3, 4)
        confidence_scores = results_df['confidence'].dropna()
        plt.hist(confidence_scores, bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
        plt.title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        
        # 5. Confidence Heatmap by Band and Model
        plt.subplot(3, 3, 5)
        heatmap_data = results_df.groupby(['band', 'decision_model'])['confidence'].mean().unstack(fill_value=0)
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn_r', center=0.5, fmt='.3f')
        plt.title('Average Confidence by Band and Model', fontsize=14, fontweight='bold')
        
        # 6. Processing Time by Model
        plt.subplot(3, 3, 6)
        time_by_model = results_df.groupby('decision_model')['processing_time'].mean()
        plt.bar(time_by_model.index, time_by_model.values, color='orange', alpha=0.7)
        plt.title('Average Processing Time by Model', fontsize=14, fontweight='bold')
        plt.ylabel('Average Time (seconds)')
        plt.xticks(rotation=45)
        
        # 7. Escalation Analysis
        plt.subplot(3, 3, 7)
        escalated = results_df[results_df['escalated'] == True]
        escalation_by_model = escalated['decision_model'].value_counts()
        plt.pie(escalation_by_model.values, labels=escalation_by_model.index, autopct='%1.1f%%')
        plt.title('Escalations by Model', fontsize=14, fontweight='bold')
        
        # 8. Threat Level Distribution
        plt.subplot(3, 3, 8)
        threat_counts = results_df['threat_level'].value_counts()
        colors = ['#28a745', '#ffc107', '#dc3545']
        plt.bar(threat_counts.index, threat_counts.values, color=colors[:len(threat_counts)], alpha=0.7)
        plt.title('Threat Level Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Posts')
        
        # 9. Processing Time vs Confidence
        plt.subplot(3, 3, 9)
        plt.scatter(results_df['processing_time'], results_df['confidence'], alpha=0.6, color='purple')
        plt.xlabel('Processing Time (seconds)')
        plt.ylabel('Confidence Score')
        plt.title('Processing Time vs Confidence', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the comprehensive visualization
        viz_file = f"{output_prefix}_visualizations_{timestamp}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {viz_file}")
        return viz_file
    
    def print_summary(self, results_df: pd.DataFrame):
        """Print a comprehensive summary of the analysis"""
        print("\n" + "="*80)
        print("📊 CONTENT ANALYSIS SUMMARY")
        print("="*80)
        
        # Basic statistics
        total_posts = len(results_df)
        # Check if 'error' column exists and count successful analyses
        if 'error' in results_df.columns:
            successful_analyses = len(results_df[results_df['error'].isna()])
        else:
            successful_analyses = total_posts  # If no error column, assume all successful
        
        print(f"\n📈 OVERVIEW:")
        print(f"   Total Posts Analyzed: {total_posts}")
        print(f"   Successful Analyses: {successful_analyses}")
        print(f"   Failed Analyses: {total_posts - successful_analyses}")
        print(f"   Success Rate: {(successful_analyses / total_posts * 100):.1f}%")
        
        # Performance metrics
        processing_times = results_df['processing_time'].dropna()
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   Average Latency: {processing_times.mean():.3f} seconds")
        print(f"   Min Processing Time: {processing_times.min():.3f} seconds")
        print(f"   Max Processing Time: {processing_times.max():.3f} seconds")
        print(f"   Total Processing Time: {processing_times.sum():.2f} seconds")
        print(f"   Throughput: {(successful_analyses / processing_times.sum()):.2f} posts/second")
        
        # Band distribution
        print(f"\n🎯 CONTENT CLASSIFICATION:")
        band_counts = results_df['band'].value_counts()
        for band, count in band_counts.items():
            percentage = (count / total_posts) * 100
            print(f"   {band}: {count} posts ({percentage:.1f}%)")
        
        # Model decisions
        print(f"\n🤖 MODEL DECISIONS:")
        model_counts = results_df['decision_model'].value_counts()
        for model, count in model_counts.items():
            percentage = (count / total_posts) * 100
            print(f"   {model}: {count} decisions ({percentage:.1f}%)")
        
        # Escalation analysis
        escalated_posts = results_df[results_df['escalated'] == True]
        print(f"\n🚨 ESCALATION ANALYSIS:")
        print(f"   Total Escalations: {len(escalated_posts)}")
        print(f"   Escalation Frequency: {(len(escalated_posts) / successful_analyses * 100):.1f}%")
        
        # Confidence analysis
        confidence_scores = results_df['confidence'].dropna()
        print(f"\n🎯 CONFIDENCE ANALYSIS:")
        print(f"   Average Confidence: {confidence_scores.mean():.3f}")
        print(f"   Min Confidence: {confidence_scores.min():.3f}")
        print(f"   Max Confidence: {confidence_scores.max():.3f}")
        
        print("\n" + "="*80)

def main():
    """Main function with user-friendly interface"""
    print("=" * 80)
    print("📊 COMPREHENSIVE CONTENT ANALYSIS TOOL")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Get file name and search in current directory
        print("📁 FILE INPUT:")
        file_name = input("Enter the name of your Excel/CSV file: ").strip()
        
        if not file_name:
            print("❌ No file name provided. Exiting.")
            return
        
        # Search for file in current directory
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, file_name)
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_name}")
            print(f"💡 Searched in: {current_dir}")
            print("💡 Make sure the file is in the same folder as this script.")
            return
        
        # Initialize analyzer
        print("\n🔧 Initializing content analyzer...")
        analyzer = ContentAnalyzer()
        
        # Get column configuration
        print("\n📋 COLUMN CONFIGURATION:")
        content_column = input("Enter the column name containing the content (default: 'content'): ").strip()
        if not content_column:
            content_column = "content"
        
        id_column = input("Enter the column name for post IDs (optional, press Enter to skip): ").strip()
        if not id_column:
            id_column = None
        
        print(f"\n✅ Configuration:")
        print(f"   📄 File: {file_path}")
        print(f"   📝 Content Column: {content_column}")
        print(f"   🆔 ID Column: {id_column or 'Auto-generated'}")
        
        # Confirm before proceeding
        print(f"\n🚀 Ready to start analysis!")
        confirm = input("Press Enter to continue or 'q' to quit: ").strip().lower()
        if confirm == 'q':
            print("❌ Analysis cancelled by user.")
            return
        
        # Start analysis
        print(f"\n🔄 Starting content analysis...")
        print("   This may take a while depending on the number of posts...")
        print()
        
        start_time = datetime.now()
        
        # Analyze bulk content
        results_df = analyzer.analyze_bulk_content(file_path, content_column, id_column)
        
        analysis_time = datetime.now() - start_time
        
        print(f"\n✅ Analysis completed in {analysis_time.total_seconds():.2f} seconds!")
        
        # Save results
        print(f"\n💾 Saving results...")
        excel_file, json_file = analyzer.save_results(results_df)
        
        # Create visualizations
        print(f"📊 Creating visualizations...")
        viz_file = analyzer.create_visualizations(results_df)
        
        # Print summary
        analyzer.print_summary(results_df)
        
        # Show file locations
        print(f"\n📁 GENERATED FILES:")
        print(f"   📊 Detailed Results: {excel_file}")
        print(f"   📄 Analysis Report: {json_file}")
        print(f"   📈 Visualizations: {viz_file}")
        
        # Performance insights
        total_posts = len(results_df)
        # Check if 'error' column exists and count successful analyses
        if 'error' in results_df.columns:
            successful_analyses = len(results_df[results_df['error'].isna()])
        else:
            successful_analyses = total_posts  # If no error column, assume all successful
        processing_times = results_df['processing_time'].dropna()
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   📈 Success Rate: {(successful_analyses / total_posts * 100):.1f}%")
        print(f"   ⚡ Average Latency: {processing_times.mean():.3f} seconds")
        print(f"   🚨 Escalation Frequency: {(len(results_df[results_df['escalated'] == True]) / successful_analyses * 100):.1f}%")
        print(f"   📊 Throughput: {(successful_analyses / processing_times.sum()):.2f} posts/second")
        
        # Band distribution summary
        band_counts = results_df['band'].value_counts()
        print(f"\n🎯 CONTENT CLASSIFICATION SUMMARY:")
        for band, count in band_counts.head(3).items():
            percentage = (count / total_posts) * 100
            print(f"   {band}: {count} posts ({percentage:.1f}%)")
        
        print(f"\n✅ Content analysis completed successfully!")
        print("=" * 80)
        
        logger.info("Content analysis completed successfully")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        logger.error(f"File not found: {e}")
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        logger.error(f"Invalid input: {e}")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        logger.error(f"Content analysis failed: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   - Make sure your file is in CSV, XLSX, or XLS format")
        print("   - Check that the content column name is correct")
        print("   - Ensure the file is not corrupted or password-protected")
        print("   - Try with a smaller file first to test")

if __name__ == "__main__":
    main() 