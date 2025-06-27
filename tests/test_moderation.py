"""
Test script for MARV Content Moderation Engine
"""

import sys
import asyncio
import json
import time
from pathlib import Path
import requests

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.moderation_service import ModerationService
from app.models.schemas import ModerationRequest, ContentType


class ModerationTester:
    """Test class for moderation system"""
    
    def __init__(self):
        self.moderation_service = ModerationService()
        self.test_results = []
    
    async def run_all_tests(self):
        """Run all moderation tests"""
        print("Starting MARV Content Moderation Engine Tests")
        print("=" * 60)
        
        # Test cases
        test_cases = [
            {
                "name": "Clean Content",
                "content": "Hello, this is a friendly message about technology and innovation.",
                "expected_action": "accept",
                "expected_threat": "low"
            },
            {
                "name": "Mild Profanity",
                "content": "This is damn annoying, but I'll deal with it.",
                "expected_action": "flag",
                "expected_threat": "medium"
            },
            {
                "name": "Strong Profanity",
                "content": "This is complete shit and I hate it!",
                "expected_action": "block",
                "expected_threat": "high"
            },
            {
                "name": "Violent Threat",
                "content": "I'm going to kill you if you don't stop.",
                "expected_action": "block",
                "expected_threat": "critical"
            },
            {
                "name": "Financial Fraud",
                "content": "Send me your credit card number and social security for verification.",
                "expected_action": "block",
                "expected_threat": "high"
            },
            {
                "name": "Spam Content",
                "content": "Click here to make money fast! Limited time offer!",
                "expected_action": "flag",
                "expected_threat": "medium"
            },
            {
                "name": "URL Detection",
                "content": "Check out this website: https://example.com",
                "expected_action": "flag",
                "expected_threat": "medium"
            },
            {
                "name": "Email Detection",
                "content": "Contact me at user@example.com for more information.",
                "expected_action": "flag",
                "expected_threat": "low"
            },
            {
                "name": "Mixed Content",
                "content": "This is a damn good deal! Send money to user@example.com and visit https://scam.com",
                "expected_action": "block",
                "expected_threat": "high"
            },
            {
                "name": "Complex Analysis",
                "content": "The market is showing negative trends with potential financial risks. Investors should be cautious about their portfolio decisions.",
                "expected_action": "flag",
                "expected_threat": "medium"
            }
        ]
        
        # Run tests
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['name']}")
            print("-" * 40)
            
            result = await self.run_single_test(test_case)
            self.test_results.append(result)
            
            # Print result
            self.print_test_result(result)
        
        # Print summary
        self.print_summary()
    
    async def run_single_test(self, test_case):
        """Run a single test case"""
        start_time = time.time()
        
        try:
            # Create moderation request
            request = ModerationRequest(
                content=test_case["content"],
                content_type=ContentType.TEXT,
                user_id="test_user"
            )
            
            # Run moderation
            result = await self.moderation_service.moderate_content(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Determine test success
            action_match = result.action.value == test_case["expected_action"]
            threat_match = result.threat_level.value == test_case["expected_threat"]
            
            return {
                "test_case": test_case,
                "result": result,
                "processing_time": processing_time,
                "success": action_match and threat_match,
                "action_match": action_match,
                "threat_match": threat_match,
                "error": None
            }
            
        except Exception as e:
            return {
                "test_case": test_case,
                "result": None,
                "processing_time": time.time() - start_time,
                "success": False,
                "action_match": False,
                "threat_match": False,
                "error": str(e)
            }
    
    def print_test_result(self, test_result):
        """Print individual test result"""
        if test_result["error"]:
            print(f"Error: {test_result['error']}")
            return
        
        result = test_result["result"]
        test_case = test_result["test_case"]
        
        print(f"Content: {test_case['content'][:50]}...")
        print(f"Action: {result.action.value} (expected: {test_case['expected_action']})")
        print(f"Threat: {result.threat_level.value} (expected: {test_case['expected_threat']})")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Processing Time: {result.processing_time_ms}ms")
        print(f"Explanation: {result.explanation}")
        
        if test_result["success"]:
            print("PASS")
        else:
            print("FAIL")
            if not test_result["action_match"]:
                print(f"   - Action mismatch: got {result.action.value}, expected {test_case['expected_action']}")
            if not test_result["threat_match"]:
                print(f"   - Threat mismatch: got {result.threat_level.value}, expected {test_case['expected_threat']}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Performance metrics
        processing_times = [r["processing_time"] for r in self.test_results if r["result"]]
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            max_time = max(processing_times)
            min_time = min(processing_times)
            
            print(f"\nPerformance Metrics:")
            print(f"Average Processing Time: {avg_time:.3f}s")
            print(f"Fastest: {min_time:.3f}s")
            print(f"Slowest: {max_time:.3f}s")
        
        # Failed tests
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for i, result in enumerate(self.test_results, 1):
                if not result["success"]:
                    print(f"  {i}. {result['test_case']['name']}")
                    if result["error"]:
                        print(f"     Error: {result['error']}")
        
        print("\n🎯 Test completed!")


async def test_api_endpoints():
    """Test API endpoints"""
    print("\n🌐 Testing API Endpoints")
    print("=" * 40)
    
    base_url = "http://localhost:8000"
    
    async with requests.AsyncSession() as session:
        # Test health endpoint
        try:
            response = await session.get(f"{base_url}/health")
            if response.status_code == 200:
                health_data = await response.json()
                print("✅ Health endpoint: OK")
                print(f"   Status: {health_data.get('status', 'unknown')}")
                print(f"   Database: {health_data.get('database', 'unknown')}")
                print(f"   ML Models: {health_data.get('ml_models', 'unknown')}")
            else:
                print(f"❌ Health endpoint: {response.status_code}")
        except Exception as e:
            print(f"❌ Health endpoint: {e}")
        
        # Test moderation endpoint
        try:
            test_data = {
                "content": "This is a test message for API testing.",
                "content_type": "text",
                "user_id": "api_test_user"
            }
            
            response = await session.post(f"{base_url}/api/v1/moderate", json=test_data)
            if response.status_code == 200:
                result = await response.json()
                print(f"✅ Moderation endpoint: OK (Action: {result['action']})")
            else:
                print(f"❌ Moderation endpoint: {response.status_code}")
        except Exception as e:
            print(f"❌ Moderation endpoint: {e}")
        
        # Test posts endpoint
        try:
            response = await session.get(f"{base_url}/api/v1/posts?limit=5")
            if response.status_code == 200:
                posts_data = await response.json()
                posts_count = len(posts_data.get("posts", []))
                print(f"✅ Posts endpoint: OK ({posts_count} posts)")
            else:
                print(f"❌ Posts endpoint: {response.status_code}")
        except Exception as e:
            print(f"❌ Posts endpoint: {e}")


async def test_individual_services():
    """Test individual services"""
    print("\n🔧 Testing Individual Services")
    print("=" * 40)
    
    # Test rule service
    try:
        from app.services.rule_service import RuleService
        rule_service = RuleService()
        
        test_content = "This is a test with damn and hack words."
        result = await rule_service.analyze(test_content)
        
        print(f"✅ Rule Service: OK")
        print(f"   - Total severity: {result['total_severity']}")
        print(f"   - Matches: {len(result['matches'])}")
        
    except Exception as e:
        print(f"❌ Rule Service: {e}")
    
    # Test ML service
    try:
        from app.services.ml_service import MLService
        ml_service = MLService()
        
        # Wait for models to load
        await asyncio.sleep(2)
        
        test_content = "This is a test message for ML analysis."
        result = await ml_service.analyze(test_content)
        
        print(f"✅ ML Service: OK")
        print(f"   - Models loaded: {result.get('models_loaded', False)}")
        print(f"   - Overall score: {result.get('overall_score', 0):.3f}")
        
    except Exception as e:
        print(f"❌ ML Service: {e}")
    
    # Test LLM service
    try:
        from app.services.llm_service import LLMService
        llm_service = LLMService()
        
        test_content = "This is a test message for LLM analysis."
        result = await llm_service.test_connection()
        
        print(f"✅ LLM Service: {result['status']}")
        
    except Exception as e:
        print(f"❌ LLM Service: {e}")


def main():
    """Run the test suite"""
    print("MARV Content Moderation Engine - Test Suite")
    print("=" * 60)
    
    # Test configuration
    test_config = {
        "api_url": "http://localhost:8000",
        "test_content": [
            {
                "content": "This is a normal, safe message.",
                "expected_action": "accept",
                "description": "Safe content"
            },
            {
                "content": "This message contains some profanity like damn and hell.",
                "expected_action": "reject",
                "description": "Profanity detection"
            },
            {
                "content": "I want to invest in this amazing opportunity! Guaranteed returns!",
                "expected_action": "reject", 
                "description": "Financial fraud detection"
            },
            {
                "content": "Check out this website: http://example.com",
                "expected_action": "reject",
                "description": "URL detection"
            },
            {
                "content": "Contact me at test@example.com or call 555-123-4567",
                "expected_action": "reject",
                "description": "Personal info detection"
            }
        ]
    }
    
    print(f"Testing API at: {test_config['api_url']}")
    print()
    
    # Test API health
    print("1. Testing API health...")
    try:
        response = requests.get(f"{test_config['api_url']}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Database: {health_data.get('database', 'unknown')}")
            print(f"   ML Models: {health_data.get('ml_models', 'unknown')}")
            print("   Health check: PASSED")
        else:
            print(f"   Health check: FAILED (Status: {response.status_code})")
    except Exception as e:
        print(f"   Health check: FAILED ({str(e)})")
    
    print()
    
    # Test moderation endpoint
    print("2. Testing moderation endpoint...")
    passed_tests = 0
    total_tests = len(test_config["test_content"])
    
    for i, test_case in enumerate(test_config["test_content"], 1):
        print(f"   Test {i}/{total_tests}: {test_case['description']}")
        print(f"   Content: {test_case['content'][:50]}...")
        
        try:
            response = requests.post(
                f"{test_config['api_url']}/api/v1/moderate",
                json={"content": test_case["content"]},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                action = result.get("action", "unknown")
                explanation = result.get("explanation", "No explanation")
                
                print(f"   Result: {action}")
                print(f"   Explanation: {explanation[:100]}...")
                
                if action == test_case["expected_action"]:
                    print("   Status: PASSED")
                    passed_tests += 1
                else:
                    print(f"   Status: FAILED (Expected: {test_case['expected_action']}, Got: {action})")
            else:
                print(f"   Status: FAILED (HTTP {response.status_code})")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"   Status: FAILED ({str(e)})")
        
        print()
    
    # Test posts endpoint
    print("3. Testing posts endpoint...")
    try:
        response = requests.get(f"{test_config['api_url']}/api/v1/posts?limit=5", timeout=5)
        if response.status_code == 200:
            posts_data = response.json()
            posts_count = len(posts_data.get("posts", []))
            print(f"   Retrieved {posts_count} posts")
            print("   Posts endpoint: PASSED")
        else:
            print(f"   Posts endpoint: FAILED (Status: {response.status_code})")
    except Exception as e:
        print(f"   Posts endpoint: FAILED ({str(e)})")
    
    print()
    
    # Summary
    print("Test Summary")
    print("=" * 60)
    print(f"Moderation tests: {passed_tests}/{total_tests} passed")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("All tests passed! System is working correctly.")
    else:
        print("Some tests failed. Check the system configuration.")


if __name__ == "__main__":
    main() 