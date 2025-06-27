#!/usr/bin/env python3
"""
Quick test script for MARV Content Moderation Engine
Run this to verify the system is working correctly
"""

import sys
import asyncio
import json
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from app.core.config import settings
        print("✅ Config: OK")
    except Exception as e:
        print(f"❌ Config: {e}")
        return False
    
    try:
        from app.core.database import get_db, create_tables
        print("✅ Database: OK")
    except Exception as e:
        print(f"❌ Database: {e}")
        return False
    
    try:
        from app.models.schemas import ModerationRequest, ModerationResult
        print("✅ Schemas: OK")
    except Exception as e:
        print(f"❌ Schemas: {e}")
        return False
    
    try:
        from app.services.rule_service import RuleService
        print("✅ Rule Service: OK")
    except Exception as e:
        print(f"❌ Rule Service: {e}")
        return False
    
    try:
        from app.services.ml_service import MLService
        print("✅ ML Service: OK")
    except Exception as e:
        print(f"❌ ML Service: {e}")
        return False
    
    try:
        from app.services.llm_service import LLMService
        print("✅ LLM Service: OK")
    except Exception as e:
        print(f"❌ LLM Service: {e}")
        return False
    
    try:
        from app.services.moderation_service import ModerationService
        print("✅ Moderation Service: OK")
    except Exception as e:
        print(f"❌ Moderation Service: {e}")
        return False
    
    return True

async def test_rule_service():
    """Test rule-based filtering"""
    print("\n🧪 Testing Rule Service...")
    
    try:
        from app.services.rule_service import RuleService
        
        rule_service = RuleService()
        
        # Test content
        test_content = "This is a test message with damn and hack words."
        result = await rule_service.analyze(test_content)
        
        print(f"✅ Rule analysis completed")
        print(f"   - Total severity: {result['total_severity']}")
        print(f"   - Confidence: {result['confidence']:.2f}")
        print(f"   - Matches: {len(result['matches'])}")
        
        for match in result['matches']:
            print(f"     - {match['type']}: '{match['matched_text']}' (severity: {match['severity']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Rule service test failed: {e}")
        return False

async def test_ml_service():
    """Test ML service (if models are available)"""
    print("\n🤖 Testing ML Service...")
    
    try:
        from app.services.ml_service import MLService
        
        ml_service = MLService()
        
        # Wait for models to load
        await asyncio.sleep(2)
        
        test_content = "This is a test message for ML analysis."
        result = await ml_service.analyze(test_content)
        
        print(f"✅ ML analysis completed")
        print(f"   - Models loaded: {result.get('models_loaded', False)}")
        print(f"   - Overall score: {result.get('overall_score', 0):.3f}")
        print(f"   - Toxicity score: {result.get('toxicity_score', 0):.3f}")
        print(f"   - Fraud score: {result.get('fraud_score', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML service test failed: {e}")
        return False

async def test_llm_service():
    """Test LLM service"""
    print("\n🧠 Testing LLM Service...")
    
    try:
        from app.services.llm_service import LLMService
        
        llm_service = LLMService()
        
        # Test connection
        result = await llm_service.test_connection()
        
        print(f"✅ LLM service test completed")
        print(f"   - Status: {result['status']}")
        
        if result['status'] == 'connected':
            print(f"   - Available models: {result.get('available_models', [])}")
            print(f"   - Target model: {result.get('target_model', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM service test failed: {e}")
        return False

async def test_moderation_pipeline():
    """Test the complete moderation pipeline"""
    print("\n🚀 Testing Complete Moderation Pipeline...")
    
    try:
        from app.services.moderation_service import ModerationService
        from app.models.schemas import ModerationRequest, ContentType
        
        moderation_service = ModerationService()
        
        # Test cases
        test_cases = [
            {
                "name": "Clean Content",
                "content": "Hello, this is a friendly message about technology.",
                "expected_action": "accept"
            },
            {
                "name": "Profanity",
                "content": "This is damn annoying!",
                "expected_action": "flag"
            },
            {
                "name": "Threat",
                "content": "I'm going to kill you!",
                "expected_action": "block"
            },
            {
                "name": "Fraud",
                "content": "Send me your credit card number.",
                "expected_action": "block"
            }
        ]
        
        for test_case in test_cases:
            print(f"\n📝 Testing: {test_case['name']}")
            
            request = ModerationRequest(
                content=test_case["content"],
                content_type=ContentType.TEXT,
                user_id="test_user"
            )
            
            result = await moderation_service.moderate_content(request)
            
            print(f"   - Action: {result.action}")
            print(f"   - Threat Level: {result.threat_level}")
            print(f"   - Confidence: {result.confidence:.2f}")
            print(f"   - Processing Time: {result.processing_time_ms}ms")
            print(f"   - Explanation: {result.explanation}")
            
            # Check if action matches expectation
            if result.action.value == test_case["expected_action"]:
                print(f"   ✅ Expected action: {test_case['expected_action']}")
            else:
                print(f"   ⚠️ Unexpected action: got {result.action.value}, expected {test_case['expected_action']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Moderation pipeline test failed: {e}")
        return False

async def test_database():
    """Test database functionality"""
    print("\n🗄️ Testing Database...")
    
    try:
        from app.core.database import create_tables, SessionLocal
        from app.models.post import Post, ModerationRule
        
        # Create tables
        create_tables()
        print("✅ Database tables created")
        
        # Test database connection
        db = SessionLocal()
        
        # Count existing data
        post_count = db.query(Post).count()
        rule_count = db.query(ModerationRule).count()
        
        print(f"✅ Database connection successful")
        print(f"   - Posts: {post_count}")
        print(f"   - Rules: {rule_count}")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_frontend():
    """Test if frontend files exist"""
    print("\n🌐 Testing Frontend...")
    
    frontend_path = Path("frontend/index.html")
    if frontend_path.exists():
        print("✅ Frontend file exists")
        return True
    else:
        print("❌ Frontend file not found")
        return False

async def main():
    """Run all tests"""
    print("🧪 MARV Content Moderation Engine - System Test")
    print("=" * 60)
    
    results = {}
    
    # Test imports
    results["imports"] = test_imports()
    
    # Test database
    results["database"] = await test_database()
    
    # Test rule service
    results["rule_service"] = await test_rule_service()
    
    # Test ML service
    results["ml_service"] = await test_ml_service()
    
    # Test LLM service
    results["llm_service"] = await test_llm_service()
    
    # Test moderation pipeline
    results["moderation_pipeline"] = await test_moderation_pipeline()
    
    # Test frontend
    results["frontend"] = test_frontend()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Start the server: uvicorn app.main:app --reload")
        print("2. Open frontend: frontend/index.html")
        print("3. Visit API docs: http://localhost:8000/docs")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 