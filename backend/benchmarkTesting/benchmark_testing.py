#!/usr/bin/env python3
"""
Day 7-8: AI Integration Testing Framework

This module provides comprehensive testing for the content moderation system:
- Benchmark test posts (clean, spam, fraud, profane, nuanced)
- Latency measurement and performance analysis
- Escalation frequency tracking
- Confidence heatmap generation
- Detailed reporting with logs
"""

import time
import json
import statistics
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from logger import get_logger
from app.core.moderation import GuardianModerationEngine

logger = get_logger("benchmark", "benchmark_testing.log")

def test_imports() -> bool:
    """Test that all required imports work correctly"""
    
    print("Testing imports...")
    
    try:
        from logger import get_logger
        print("✅ logger import successful")
        
        from app.core.moderation import GuardianModerationEngine
        print("✅ GuardianModerationEngine import successful")
        
        # Test logger creation
        test_logger = get_logger("test", "test_imports.log")
        test_logger.info("Import test successful")
        print("✅ Logger creation successful")
        
        # Test moderation engine creation
        engine = GuardianModerationEngine()
        print("✅ Moderation engine creation successful")
        
        print("\n🎉 All imports and initializations successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

class BenchmarkTestSuite:
    """Comprehensive benchmark testing for content moderation system"""
    
    def __init__(self):
        self.moderation_engine = GuardianModerationEngine()
        self.results = []
        self.latency_stats = defaultdict(list)
        self.escalation_stats = defaultdict(int)
        self.confidence_data = []
        
        # Test categories and sample content
        self.test_categories = {
            "clean": [
                "Hello, how are you today?",
                "I love this beautiful weather we're having.",
                "Thank you for your help with the project.",
                "The new restaurant downtown is amazing!",
                "I'm looking forward to the weekend."
            ],
            "spam": [
                "CLICK HERE TO WIN $1000 NOW!!!",
                "Make money fast from home - guaranteed results!",
                "FREE VIAGRA - LIMITED TIME OFFER!!!",
                "You've won a prize! Claim it now!",
                "Double your money in 24 hours - guaranteed!"
            ],
            "fraud": [
                "I need your bank account details to transfer inheritance money",
                "I'm a Nigerian prince and need your help with $10 million",
                "Your account has been compromised, please verify your details",
                "You've won the lottery! Send $500 processing fee to claim",
                "I can help you get rich quick with this investment opportunity"
            ],
            "profane": [
                "You're a complete idiot and I hate you",
                "This is the worst piece of garbage I've ever seen",
                "Go to hell and never come back",
                "I hope you die in a fire",
                "You're so stupid it's unbelievable"
            ],
            "nuanced": [
                "The stock market crash could be seen as both a disaster and opportunity",
                "While the policy has good intentions, it may have unintended consequences",
                "The debate about climate change involves complex scientific and economic factors",
                "This decision affects multiple stakeholders with competing interests",
                "The situation requires careful consideration of ethical implications"
            ]
        }
    
    def run_single_test(self, content: str, category: str) -> Dict[str, Any]:
        """Run a single moderation test and record results"""
        start_time = time.time()
        
        try:
            result = self.moderation_engine.moderate_content(content)
            processing_time = time.time() - start_time
            
            test_result = {
                "timestamp": datetime.utcnow().isoformat(),
                "category": category,
                "content": content[:100] + "..." if len(content) > 100 else content,
                "accepted": result.accepted,
                "reason": result.reason,
                "threat_level": result.threat_level,
                "confidence": result.confidence,
                "stage": result.stage,
                "band": result.band,
                "action": result.action,
                "processing_time": processing_time,
                "escalated": self._is_escalated(result)
            }
            
            # Record statistics
            self.latency_stats[category].append(processing_time)
            self.confidence_data.append({
                "category": category,
                "confidence": result.confidence,
                "threat_level": result.threat_level,
                "stage": result.stage
            })
            
            if test_result["escalated"]:
                self.escalation_stats[category] += 1
            
            logger.info(f"Test completed - Category: {category}, Result: {result.accepted}, "
                       f"Time: {processing_time:.3f}s, Confidence: {result.confidence}")
            
            return test_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Test failed for category {category}: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "category": category,
                "content": content[:100] + "..." if len(content) > 100 else content,
                "error": str(e),
                "processing_time": processing_time
            }
    
    def _is_escalated(self, result) -> bool:
        """Determine if content was escalated based on moderation result"""
        # Escalation criteria
        escalation_indicators = [
            result.threat_level in ["medium", "high"],
            result.confidence < 0.7,
            result.stage in ["detoxify", "finbert"],
            result.action in ["FLAG_LOW", "FLAG_MEDIUM", "FLAG_HIGH", "BLOCK"]
        ]
        return any(escalation_indicators)
    
    def run_benchmark_suite(self, iterations: int = 3) -> Dict[str, Any]:
        """Run complete benchmark test suite"""
        logger.info(f"Starting benchmark test suite with {iterations} iterations per category")
        
        start_time = time.time()
        
        for iteration in range(iterations):
            logger.info(f"Running iteration {iteration + 1}/{iterations}")
            
            for category, test_posts in self.test_categories.items():
                logger.info(f"Testing category: {category}")
                
                for post in test_posts:
                    result = self.run_single_test(post, category)
                    self.results.append(result)
                    
                    # Small delay to avoid overwhelming the system
                    time.sleep(0.1)
        
        total_time = time.time() - start_time
        logger.info(f"Benchmark suite completed in {total_time:.2f} seconds")
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        logger.info("Generating benchmark report...")
        
        # Calculate statistics
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if "error" not in r])
        
        # Latency analysis
        latency_summary = {}
        for category, times in self.latency_stats.items():
            if times:
                latency_summary[category] = {
                    "count": len(times),
                    "avg_latency": statistics.mean(times),
                    "min_latency": min(times),
                    "max_latency": max(times),
                    "std_dev": statistics.stdev(times) if len(times) > 1 else 0
                }
        
        # Overall latency stats
        all_latencies = [r["processing_time"] for r in self.results if "processing_time" in r]
        overall_latency = {
            "avg_latency": statistics.mean(all_latencies) if all_latencies else 0,
            "min_latency": min(all_latencies) if all_latencies else 0,
            "max_latency": max(all_latencies) if all_latencies else 0,
            "total_requests": len(all_latencies)
        }
        
        # Escalation analysis
        total_escalations = sum(self.escalation_stats.values())
        escalation_rate = (total_escalations / successful_tests * 100) if successful_tests > 0 else 0
        
        # Accuracy analysis by category
        accuracy_by_category = {}
        for category in self.test_categories.keys():
            category_results = [r for r in self.results if r.get("category") == category and "error" not in r]
            if category_results:
                # Define expected outcomes
                expected_rejected = category in ["spam", "fraud", "profane"]
                correct_decisions = sum(1 for r in category_results if r["accepted"] != expected_rejected)
                accuracy_by_category[category] = {
                    "accuracy": (correct_decisions / len(category_results)) * 100,
                    "total_tests": len(category_results),
                    "correct_decisions": correct_decisions
                }
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "performance": {
                "overall_latency": overall_latency,
                "latency_by_category": latency_summary
            },
            "escalation": {
                "total_escalations": total_escalations,
                "escalation_rate": escalation_rate,
                "escalations_by_category": dict(self.escalation_stats)
            },
            "accuracy": accuracy_by_category,
            "confidence_data": self.confidence_data,
            "detailed_results": self.results
        }
        
        # Save report to file
        self._save_report(report)
        
        # Generate visualizations
        self._generate_confidence_heatmap()
        self._generate_latency_chart()
        
        logger.info("Benchmark report generated successfully")
        return report
    
    def _save_report(self, report: Dict[str, Any]):
        """Save benchmark report to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"benchmark_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Benchmark report saved to {report_file}")
    
    def _generate_confidence_heatmap(self):
        """Generate confidence heatmap visualization"""
        try:
            if not self.confidence_data:
                logger.warning("No confidence data available for heatmap")
                return
            
            # Create DataFrame for heatmap
            df = pd.DataFrame(self.confidence_data)
            
            # Create pivot table for heatmap
            pivot_data = df.pivot_table(
                values='confidence',
                index='threat_level',
                columns='category',
                aggfunc='mean'
            ).fillna(0)
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_data, annot=True, cmap='RdYlGn_r', center=0.5, 
                       fmt='.3f', cbar_kws={'label': 'Average Confidence'})
            plt.title('Confidence Heatmap by Category and Threat Level')
            plt.xlabel('Content Category')
            plt.ylabel('Threat Level')
            plt.tight_layout()
            
            # Save heatmap
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            heatmap_file = f"confidence_heatmap_{timestamp}.png"
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confidence heatmap saved to {heatmap_file}")
            
        except Exception as e:
            logger.error(f"Error generating confidence heatmap: {e}")
    
    def _generate_latency_chart(self):
        """Generate latency comparison chart"""
        try:
            if not self.latency_stats:
                logger.warning("No latency data available for chart")
                return
            
            # Prepare data for plotting
            categories = list(self.latency_stats.keys())
            avg_latencies = [statistics.mean(times) for times in self.latency_stats.values()]
            
            # Create bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.bar(categories, avg_latencies, color='skyblue', alpha=0.7)
            
            # Add value labels on bars
            for bar, latency in zip(bars, avg_latencies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{latency:.3f}s', ha='center', va='bottom')
            
            plt.title('Average Processing Latency by Content Category')
            plt.xlabel('Content Category')
            plt.ylabel('Average Latency (seconds)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_file = f"latency_chart_{timestamp}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Latency chart saved to {chart_file}")
            
        except Exception as e:
            logger.error(f"Error generating latency chart: {e}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a human-readable summary of the benchmark results"""
        print("\n" + "="*60)
        print("CONTENT MODERATION BENCHMARK TEST RESULTS")
        print("="*60)
        
        summary = report["summary"]
        print(f"\n📊 TEST SUMMARY:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Successful: {summary['successful_tests']}")
        print(f"   Failed: {summary['failed_tests']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        
        performance = report["performance"]
        overall = performance["overall_latency"]
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Average Latency: {overall['avg_latency']:.3f}s")
        print(f"   Min Latency: {overall['min_latency']:.3f}s")
        print(f"   Max Latency: {overall['max_latency']:.3f}s")
        print(f"   Total Requests: {overall['total_requests']}")
        
        escalation = report["escalation"]
        print(f"\n🚨 ESCALATION ANALYSIS:")
        print(f"   Total Escalations: {escalation['total_escalations']}")
        print(f"   Escalation Rate: {escalation['escalation_rate']:.1f}%")
        
        print(f"\n📈 ACCURACY BY CATEGORY:")
        for category, stats in report["accuracy"].items():
            print(f"   {category.upper()}: {stats['accuracy']:.1f}% "
                  f"({stats['correct_decisions']}/{stats['total_tests']})")
        
        print("\n" + "="*60)

def main():
    """Main function to run benchmark testing with import verification"""
    print("🧪 DAY 7-8: AI INTEGRATION TESTING")
    print("=" * 50)
    
    # First, test imports
    print("📋 Testing imports and dependencies...")
    if not test_imports():
        print("❌ Import test failed. Please check your setup.")
        return
    
    print("\n🚀 Starting benchmark testing...")
    logger.info("Starting Day 7-8 AI Integration Testing")
    
    try:
        # Create and run benchmark suite
        benchmark = BenchmarkTestSuite()
        report = benchmark.run_benchmark_suite(iterations=2)  # 2 iterations per category
        
        # Print summary
        benchmark.print_summary(report)
        
        print("\n✅ Benchmark testing completed successfully!")
        logger.info("Benchmark testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during benchmark testing: {e}")
        logger.error(f"Benchmark testing failed: {e}")

if __name__ == "__main__":
    main() 