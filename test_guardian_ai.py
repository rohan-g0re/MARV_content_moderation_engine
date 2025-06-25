#!/usr/bin/env python3
"""
Test Script for GuardianAI Content Moderation Engine

This script tests the GuardianAI system to verify it works correctly
and demonstrates its capabilities as specified in the project brief.
"""

import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from app.core.guardian_ai import GuardianAI
from app.core.config import get_default_config

def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_guardian_ai():
    """Test the GuardianAI system with various content types"""
    logger = setup_logging()
    
    # Initialize GuardianAI
    config = get_default_config()
    guardian = GuardianAI(config, logger)
    
    # Test cases covering different threat levels
    test_cases = [
        {
            "content": "This is a safe and normal post about finance and investing.",
            "description": "Safe content - should be accepted"
        },
        {
            "content": "I love learning about stocks and crypto! Great investment advice here.",
            "description": "Safe financial content"
        },
        {
            "content": "This system is so damn annoying and stupid.",
            "description": "Mild profanity - should be flagged"
        },
        {
            "content": "I hate this piece of shit system and want to kill it.",
            "description": "Strong profanity and violent language - should be blocked"
        },
        {
            "content": "Here's how to hack into the system and steal money.",
            "description": "Illegal activities - should be escalated"
        },
        {
            "content": "Kill yourself and commit suicide, you worthless piece of shit.",
            "description": "Critical threat - should be escalated"
        },
        {
            "content": "Just sharing some investment advice - buy low, sell high! #investing #finance",
            "description": "Safe content with hashtags"
        }
    ]
    
    print("="*80)
    print("GUARDIANAI CONTENT MODERATION ENGINE - TEST RESULTS")
    print("="*80)
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case['description']}")
        print("-" * 60)
        print(f"Content: {test_case['content']}")
        
        try:
            # Perform moderation
            result = guardian.moderate_content(test_case['content'], f"test_{i+1}")
            
            # Display results
            print(f"Threat Level: {result.threat_level.name}")
            print(f"Action: {result.action.value}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Processing Time: {result.processing_time_ms:.2f}ms")
            print(f"Explanation: {result.explanation}")
            
            if result.layer_results:
                print("Layer Results:")
                for layer, layer_result in result.layer_results.items():
                    print(f"  {layer}: {layer_result.get('threat_score', 0):.3f}")
            
            results.append({
                "test_id": i+1,
                "threat_level": result.threat_level.name,
                "action": result.action.value,
                "confidence": result.confidence,
                "processing_time_ms": result.processing_time_ms
            })
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "test_id": i+1,
                "error": str(e)
            })
    
    # Display summary statistics
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    successful_tests = [r for r in results if "error" not in r]
    failed_tests = [r for r in results if "error" in r]
    
    print(f"Total Tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if successful_tests:
        # Action distribution
        action_counts = {}
        threat_counts = {}
        total_processing_time = 0
        
        for result in successful_tests:
            action = result["action"]
            threat = result["threat_level"]
            
            action_counts[action] = action_counts.get(action, 0) + 1
            threat_counts[threat] = threat_counts.get(threat, 0) + 1
            total_processing_time += result["processing_time_ms"]
        
        print(f"\nAction Distribution:")
        for action, count in action_counts.items():
            percentage = (count / len(successful_tests)) * 100
            print(f"  {action}: {count} ({percentage:.1f}%)")
        
        print(f"\nThreat Level Distribution:")
        for threat, count in threat_counts.items():
            percentage = (count / len(successful_tests)) * 100
            print(f"  {threat}: {count} ({percentage:.1f}%)")
        
        avg_processing_time = total_processing_time / len(successful_tests)
        print(f"\nAverage Processing Time: {avg_processing_time:.2f}ms")
    
    # Get system statistics
    print(f"\nSystem Statistics:")
    stats = guardian.get_statistics()
    print(f"  Total Processed: {stats['total_processed']}")
    print(f"  Layer Usage: {stats['layer_usage']}")
    
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)
    
    return len(failed_tests) == 0

if __name__ == "__main__":
    success = test_guardian_ai()
    sys.exit(0 if success else 1) 