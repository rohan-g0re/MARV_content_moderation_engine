#!/usr/bin/env python3
"""
Test script for CiferAI integration in the moderation pipeline
"""

import logging
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.core.moderation import create_default_pipeline, GuardianModerationEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ciferai_integration():
    """Test CiferAI integration with various fraud scenarios"""
    
    print("🔍 Testing CiferAI Integration in GuardianAI Moderation Pipeline")
    print("=" * 70)
    
    # Test cases with known fraud patterns
    test_cases = [
        {
            "text": "New loophole lets you legally double your bank balance in 7 days",
            "expected": "BLOCK",
            "description": "High-confidence fraud: legal loophole + money doubling"
        },
        {
            "text": "High quality counterfeit bills available, ping me for details",
            "expected": "BLOCK", 
            "description": "High-confidence fraud: counterfeit + contact request"
        },
        {
            "text": "Get rich quick with this investment opportunity",
            "expected": "ESCALATE",
            "description": "Medium-confidence fraud: get rich quick + investment"
        },
        {
            "text": "Investment opportunity with guaranteed returns",
            "expected": "ESCALATE",
            "description": "Medium-confidence fraud: guaranteed returns"
        },
        {
            "text": "This is a legitimate business post about investments",
            "expected": "ACCEPT",
            "description": "Legitimate content: normal business language"
        },
        {
            "text": "I'm looking for investment advice for my retirement",
            "expected": "ACCEPT", 
            "description": "Legitimate content: personal investment inquiry"
        }
    ]
    
    try:
        # Create pipeline
        print("📦 Initializing moderation pipeline...")
        pipeline = create_default_pipeline()
        
        # Check model status
        engine = GuardianModerationEngine()
        status = engine.get_model_status()
        print(f"✅ Pipeline initialized with {status['pipeline_stages']} stages")
        print(f"📋 Stages: {status['stage_names']}")
        print()
        
        # Test each case
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"🧪 Test {i}: {case['description']}")
            print(f"   Text: '{case['text']}'")
            
            # Process through pipeline
            result = pipeline.process(case['text'])
            
            # Check if decision matches expected
            decision_correct = result.decision == case['expected']
            status_icon = "✅" if decision_correct else "❌"
            
            print(f"   Decision: {result.decision} (expected: {case['expected']}) {status_icon}")
            print(f"   Stage: {result.stage}")
            print(f"   Reason: {result.reason}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Threat Level: {result.threat_level}")
            
            # Show stage breakdown
            stage_results = pipeline.get_stage_results()
            print("   Stage Breakdown:")
            for stage_name, stage_result in stage_results.items():
                print(f"     {stage_name}: {stage_result.decision} ({stage_result.reason})")
            
            results.append({
                "case": case,
                "result": result,
                "correct": decision_correct
            })
            
            print()
        
        # Summary
        print("📊 Test Summary")
        print("=" * 70)
        correct_count = sum(1 for r in results if r['correct'])
        total_count = len(results)
        
        print(f"Total Tests: {total_count}")
        print(f"Correct Decisions: {correct_count}")
        print(f"Accuracy: {correct_count/total_count*100:.1f}%")
        
        # Detailed breakdown
        print("\nDetailed Results:")
        for i, result in enumerate(results, 1):
            status = "✅ PASS" if result['correct'] else "❌ FAIL"
            print(f"  Test {i}: {status} - {result['case']['description']}")
        
        # CiferAI specific info
        print("\n🔍 CiferAI Integration Status:")
        stage3b_results = [r for r in results if 'stage3b' in r['result'].stage]
        if stage3b_results:
            print("✅ Stage 3b (Fraud Classifier) is active")
            cifer_reasons = [r['result'].reason for r in stage3b_results]
            print(f"   Detection methods used: {set(cifer_reasons)}")
        else:
            print("⚠️  Stage 3b not reached in any test case")
        
        return correct_count == total_count
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        print(f"❌ Test failed: {e}")
        return False

def test_individual_stages():
    """Test individual stages to verify CiferAI integration"""
    
    print("\n🔧 Testing Individual Stages")
    print("=" * 70)
    
    try:
        from backend.app.core.moderation import Stage3bFraudClassifier
        
        # Test fraud classifier directly
        fraud_classifier = Stage3bFraudClassifier()
        
        test_text = "High quality counterfeit bills available, ping me for details"
        result = fraud_classifier.process(test_text)
        
        print(f"Fraud Classifier Test:")
        print(f"  Input: '{test_text}'")
        print(f"  Decision: {result.decision}")
        print(f"  Reason: {result.reason}")
        print(f"  Confidence: {result.confidence:.3f}")
        
        # Check if CiferAI is being used
        if "hf_" in result.reason or "api_" in result.reason:
            print("✅ CiferAI model is being used")
        elif "heuristic_" in result.reason:
            print("⚠️  Using heuristic fallback (CiferAI not available)")
        else:
            print("❓ Unknown detection method")
            
    except Exception as e:
        logger.error(f"Individual stage test failed: {e}")
        print(f"❌ Individual stage test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting CiferAI Integration Tests")
    print("=" * 70)
    
    # Run main integration test
    success = test_ciferai_integration()
    
    # Run individual stage test
    test_individual_stages()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 All tests passed! CiferAI integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n💡 Next Steps:")
    print("1. Review the test results above")
    print("2. Check logs for any CiferAI model loading issues")
    print("3. Adjust thresholds if needed based on your use case")
    print("4. Test with your own content") 