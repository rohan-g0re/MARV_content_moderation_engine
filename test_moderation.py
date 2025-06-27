#!/usr/bin/env python3
"""
Comprehensive Test Script for GuardianAI Content Moderation System v2.0

Tests the consolidated three-stage moderation pipeline:
1. Rule-based filtering
2. Detoxify AI toxicity detection  
3. FinBERT financial fraud detection

PowerShell Commands for manual testing:
# Test moderation
Invoke-RestMethod -Uri "http://localhost:8000/moderate" -Method POST -ContentType "application/json" -Body '{"content": "You are a scammer and I hate this!"}'

# Get all posts
Invoke-RestMethod -Uri "http://localhost:8000/posts" -Method GET

# Get statistics
Invoke-RestMethod -Uri "http://localhost:8000/stats" -Method GET
"""

import requests
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoint
API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint and model status"""
    try:
        logger.info("🔍 Testing health endpoint...")
        response = requests.get(f"{API_URL}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ Health check passed")
            logger.info(f"Models status: {data.get('models', {})}")
            return True
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        logger.error(f"❌ Health check failed: {e}")
        logger.error("Make sure the backend is running: python backend/main.py")
        return False

def test_moderation():
    """Test the enhanced moderation endpoint with various content"""
    
    test_cases = [
        {
            "content": "This is a normal post about technology and innovation.",
            "expected": "accepted",
            "description": "Normal technology discussion"
        },
        {
            "content": "You are a scammer and I hate this!",
            "expected": "rejected",
            "description": "Rule-based rejection (keywords)"
        },
        {
            "content": "I love this amazing product, it's fantastic!",
            "expected": "accepted", 
            "description": "Positive content"
        },
        {
            "content": "This is fraud and illegal financial activity for sure.",
            "expected": "rejected",
            "description": "Financial fraud detection"
        },
        {
            "content": "Check out this website: http://spam-site.com for free money",
            "expected": "rejected",
            "description": "URL pattern detection"
        },
        {
            "content": "Contact me at spam@badsite.com for quick cash schemes",
            "expected": "rejected",
            "description": "Email pattern detection"
        },
        {
            "content": "I will kill your account if you don't pay",
            "expected": "rejected",
            "description": "Violence/threat pattern detection"
        },
        {
            "content": "The stock market analysis shows positive trends this quarter.",
            "expected": "accepted",
            "description": "Professional financial discussion"
        },
        {
            "content": "Call me at 1234567890 for exclusive investment opportunities",
            "expected": "rejected", 
            "description": "Phone number detection"
        },
        {
            "content": "🚀 Excited about the new tech developments! #innovation #tech",
            "expected": "accepted",
            "description": "Social media style positive content"
        }
    ]
    
    logger.info("🧪 Testing Enhanced Content Moderation System")
    logger.info("=" * 70)
    
    passed_tests = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Test {i}/{total_tests}: {test_case['description']}")
        logger.info(f"Content: \"{test_case['content']}\"")
        
        try:
            response = requests.post(
                f"{API_URL}/moderate",
                json={"content": test_case["content"]},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Enhanced logging with new fields
                logger.info(f"✅ Status: {response.status_code}")
                logger.info(f"   Accepted: {data['accepted']}")
                logger.info(f"   Reason: {data['reason']}")
                logger.info(f"   Threat Level: {data.get('threat_level', 'N/A')}")
                logger.info(f"   Confidence: {data.get('confidence', 'N/A')}")
                logger.info(f"   Stage: {data.get('stage', 'N/A')}")
                logger.info(f"   Action: {data.get('action', 'N/A')}")
                logger.info(f"   Post ID: {data['id']}")
                
                # Check if result matches expectation
                actual_result = "accepted" if data["accepted"] else "rejected"
                if actual_result == test_case["expected"]:
                    logger.info("   ✅ Result matches expectation")
                    passed_tests += 1
                else:
                    logger.warning(f"   ⚠️ Expected {test_case['expected']}, got {actual_result}")
                    
            else:
                logger.error(f"❌ Request failed: {response.status_code}")
                logger.error(f"   Response: {response.text}")
                
        except requests.RequestException as e:
            logger.error(f"❌ Request failed: {e}")
        
        logger.info("-" * 70)
        time.sleep(0.5)  # Small delay between requests
    
    # Summary
    success_rate = (passed_tests / total_tests) * 100
    logger.info(f"📊 Test Summary: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        logger.info("🎉 Moderation system working well!")
    else:
        logger.warning("⚠️ Some tests failed - check moderation logic")
    
    return success_rate >= 80

def test_get_posts():
    """Test getting all posts with enhanced information"""
    try:
        logger.info("📄 Testing get posts endpoint...")
        response = requests.get(f"{API_URL}/posts", timeout=10)
        
        if response.status_code == 200:
            posts = response.json()
            logger.info(f"✅ Retrieved {len(posts)} posts")
            
            if posts:
                # Show details of the most recent post
                latest_post = posts[0]
                logger.info("Latest post details:")
                logger.info(f"   ID: {latest_post['id']}")
                logger.info(f"   Content: \"{latest_post['content'][:50]}...\"")
                logger.info(f"   Accepted: {latest_post['accepted']}")
                logger.info(f"   Reason: {latest_post['reason']}")
                logger.info(f"   Threat Level: {latest_post.get('threat_level', 'N/A')}")
                logger.info(f"   Stage: {latest_post.get('stage', 'N/A')}")
                logger.info(f"   Created: {latest_post['created_at']}")
            
            return True
        else:
            logger.error(f"❌ Failed to get posts: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        logger.error(f"❌ Failed to get posts: {e}")
        return False

def test_stats():
    """Test the new statistics endpoint"""
    try:
        logger.info("📊 Testing statistics endpoint...")
        response = requests.get(f"{API_URL}/stats", timeout=10)
        
        if response.status_code == 200:
            stats = response.json()
            logger.info("✅ Statistics retrieved successfully:")
            logger.info(f"   Total posts: {stats.get('total_posts', 0)}")
            logger.info(f"   Accepted: {stats.get('accepted', 0)}")
            logger.info(f"   Rejected: {stats.get('rejected', 0)}")
            logger.info(f"   Acceptance rate: {stats.get('acceptance_rate', 0):.1f}%")
            
            # Rejection breakdown by stage
            rejection_stats = stats.get('rejection_by_stage', {})
            if rejection_stats:
                logger.info("   Rejection by stage:")
                for stage, count in rejection_stats.items():
                    logger.info(f"     {stage}: {count}")
            
            # Threat level distribution
            threat_stats = stats.get('threat_levels', {})
            if threat_stats:
                logger.info("   Threat level distribution:")
                for level, count in threat_stats.items():
                    logger.info(f"     {level}: {count}")
            
            # Model status
            models = stats.get('models', {})
            logger.info(f"   Models loaded: Detoxify={models.get('detoxify_loaded', False)}, FinBERT={models.get('finbert_loaded', False)}")
            
            return True
        else:
            logger.error(f"❌ Failed to get stats: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        logger.error(f"❌ Failed to get stats: {e}")
        return False

def test_admin_endpoints():
    """Test admin endpoints for keyword management"""
    try:
        logger.info("🔧 Testing admin endpoints...")
        
        # Test keyword reload
        response = requests.post(f"{API_URL}/admin/reload-keywords", timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Keywords reloaded: {data.get('count', 0)} keywords")
        else:
            logger.warning(f"⚠️ Keyword reload failed: {response.status_code}")
        
        # Test threshold update
        response = requests.post(
            f"{API_URL}/admin/update-thresholds",
            params={
                "toxicity_threshold": 0.5,
                "finbert_threshold": 0.7
            },
            timeout=10
        )
        if response.status_code == 200:
            logger.info("✅ Thresholds updated successfully")
        else:
            logger.warning(f"⚠️ Threshold update failed: {response.status_code}")
            
        return True
        
    except requests.RequestException as e:
        logger.error(f"❌ Admin endpoint tests failed: {e}")
        return False

def main():
    """Main test execution"""
    logger.info("🚀 Starting GuardianAI Content Moderation Tests v2.0...")
    logger.info("=" * 70)
    
    start_time = time.time()
    all_passed = True
    
    # Test sequence
    tests = [
        ("Health Check", test_health),
        ("Moderation Pipeline", test_moderation),
        ("Get Posts", test_get_posts),
        ("Statistics", test_stats),
        ("Admin Endpoints", test_admin_endpoints)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} tests...")
        try:
            result = test_func()
            if not result:
                all_passed = False
                logger.error(f"❌ {test_name} tests failed")
            else:
                logger.info(f"✅ {test_name} tests passed")
        except Exception as e:
            logger.error(f"❌ {test_name} tests crashed: {e}")
            all_passed = False
    
    # Final summary
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info("\n" + "=" * 70)
    if all_passed:
        logger.info("🎉 All tests completed successfully!")
        logger.info("✨ GuardianAI Content Moderation System v2.0 is working correctly!")
    else:
        logger.error("❌ Some tests failed. Check the logs above for details.")
    
    logger.info(f"⏱️  Total test duration: {duration:.2f} seconds")
    logger.info("\n📖 Manual Testing Commands:")
    logger.info('Invoke-RestMethod -Uri "http://localhost:8000/moderate" -Method POST -ContentType "application/json" -Body \'{"content": "test content"}\'')
    logger.info('Invoke-RestMethod -Uri "http://localhost:8000/posts" -Method GET')
    logger.info('Invoke-RestMethod -Uri "http://localhost:8000/stats" -Method GET')
    logger.info("\n🌐 Frontend: Open frontend/index.html in your browser")
    logger.info("📚 API Documentation: http://localhost:8000/docs")

if __name__ == "__main__":
    main() 