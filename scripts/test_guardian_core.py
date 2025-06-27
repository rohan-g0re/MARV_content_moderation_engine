#!/usr/bin/env python3
"""
Test script for GuardianAI Core Pipeline
Demonstrates the moderate_content() entrypoint with structured output
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from guardian_core import GuardianAICore, ContentType, ThreatLevel, ModerationAction

def test_guardian_core():
    """Test the GuardianAI Core Pipeline"""
    print("🛡️ GuardianAI Core Pipeline Test")
    print("=" * 50)
    
    # Initialize GuardianAI Core
    guardian = GuardianAICore()
    
    # Test cases with different threat levels
    test_cases = [
        {
            "content": "This is a wonderful community! I love being here.",
            "expected_level": ThreatLevel.LOW,
            "description": "Positive content"
        },
        {
            "content": "I'm frustrated with the service quality.",
            "expected_level": ThreatLevel.LOW,
            "description": "Mild negative sentiment"
        },
        {
            "content": "This damn service is terrible and I hate it!",
            "expected_level": ThreatLevel.MEDIUM,
            "description": "Strong negative language"
        },
        {
            "content": "I'm going to hack into your system and destroy everything!",
            "expected_level": ThreatLevel.HIGH,
            "description": "Threatening content"
        },
        {
            "content": "Send me your credit card details and social security number for verification.",
            "expected_level": ThreatLevel.HIGH,
            "description": "Financial fraud attempt"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"Content: {test_case['content']}")
        print("-" * 60)
        
        # Run moderation
        result = guardian.moderate_content(
            content=test_case['content'],
            content_type=ContentType.TEXT,
            content_id=f"test_case_{i}"
        )
        
        # Display results
        print(f"🆔 Content ID: {result.content_id}")
        print(f"⚠️ Threat Level: {result.threat_level.value.upper()}")
        print(f"🛡️ Action: {result.action.value.upper()}")
        print(f"📊 Confidence: {result.confidence:.2f}")
        print(f"⏱️ Processing Time: {result.processing_time_ms}ms")
        print(f"💬 Explanation: {result.explanation}")
        
        # Check if result matches expectation
        expected = test_case['expected_level']
        actual = result.threat_level
        status = "✅ PASS" if actual == expected else "❌ FAIL"
        print(f"🎯 Expected: {expected.value.upper()}, Got: {actual.value.upper()} - {status}")
        
        results.append({
            "test_id": i,
            "description": test_case['description'],
            "expected": expected,
            "actual": actual,
            "passed": actual == expected,
            "result": result
        })
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    # Detailed results
    print("\n📋 DETAILED RESULTS:")
    for result in results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"Test {result['test_id']}: {result['description']} - {status}")
        if not result['passed']:
            print(f"  Expected: {result['expected'].value}, Got: {result['actual'].value}")
    
    guardian.close()
    
    return passed == total

def demo_structured_output():
    """Demonstrate the structured output format"""
    print("\n" + "=" * 50)
    print("🔍 STRUCTURED OUTPUT DEMO")
    print("=" * 50)
    
    guardian = GuardianAICore()
    
    # Test with a complex case
    content = "I'm extremely angry and I want to hack your system to steal money!"
    
    print(f"📝 Input Content: {content}")
    print("-" * 50)
    
    result = guardian.moderate_content(
        content=content,
        content_type=ContentType.TEXT,
        content_id="demo_001"
    )
    
    # Show structured output
    print("📊 STRUCTURED RESULT:")
    print(f"  Content ID: {result.content_id}")
    print(f"  Threat Level: {result.threat_level.value}")
    print(f"  Action: {result.action.value}")
    print(f"  Explanation: {result.explanation}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Processing Time: {result.processing_time_ms}ms")
    print(f"  Timestamp: {result.timestamp}")
    
    print("\n📈 METADATA:")
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")
    
    guardian.close()

if __name__ == "__main__":
    print("🚀 Starting GuardianAI Core Pipeline Tests...")
    
    # Run basic tests
    success = test_guardian_core()
    
    # Run structured output demo
    demo_structured_output()
    
    if success:
        print("\n🎉 All tests passed! GuardianAI Core Pipeline is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")
    
    print("\n✅ Test completed!") 