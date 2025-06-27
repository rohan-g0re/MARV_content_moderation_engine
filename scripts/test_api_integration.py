#!/usr/bin/env python3
"""
Test API integration and database connectivity
"""

import sys
import requests
import json
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

def test_api_health():
    """Test if the API is running and healthy"""
    print("🔍 Testing API health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy: {data['status']}")
            print(f"   Database: {data.get('database_status', 'unknown')}")
            print(f"   ML Models: {data.get('ml_models_status', {})}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API connection failed: {e}")
        return False

def test_moderation_endpoint():
    """Test the moderation endpoint"""
    print("\n🔍 Testing moderation endpoint...")
    try:
        test_content = "This is a test message with some concerning words like damn and hack."
        
        response = requests.post(
            "http://localhost:8000/api/v1/moderate",
            headers={"Content-Type": "application/json"},
            json={
                "content": test_content,
                "content_type": "text",
                "user_id": "test_user"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Moderation endpoint working!")
            print(f"   Action: {result.get('action', 'unknown')}")
            print(f"   Threat Level: {result.get('threat_level', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")
            print(f"   Processing Time: {result.get('processing_time_ms', 0)}ms")
            return True
        else:
            print(f"❌ Moderation endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Moderation endpoint error: {e}")
        return False

def test_posts_endpoint():
    """Test the posts endpoint"""
    print("\n🔍 Testing posts endpoint...")
    try:
        response = requests.get("http://localhost:8000/api/v1/posts?limit=5", timeout=5)
        
        if response.status_code == 200:
            posts = response.json()
            print(f"✅ Posts endpoint working! Found {len(posts)} posts")
            return True
        else:
            print(f"❌ Posts endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Posts endpoint error: {e}")
        return False

def check_database():
    """Check database files"""
    print("\n🔍 Checking database files...")
    
    # Check production database
    marv_db = Path("marv.db")
    if marv_db.exists():
        print(f"✅ Production database exists: {marv_db} ({marv_db.stat().st_size} bytes)")
    else:
        print("❌ Production database not found: marv.db")
    
    # Check legacy database
    legacy_db = Path("backend/moderation.db")
    if legacy_db.exists():
        print(f"⚠️ Legacy database exists: {legacy_db} ({legacy_db.stat().st_size} bytes)")
    else:
        print("ℹ️ No legacy database found")

def check_words_json():
    """Check words.json files"""
    print("\n🔍 Checking words.json files...")
    
    words_files = [
        Path("backend/words.json"),
        Path("data/words.json")
    ]
    
    for words_file in words_files:
        if words_file.exists():
            try:
                with open(words_file, 'r', encoding='utf-8') as f:
                    keywords = json.load(f)
                print(f"✅ {words_file}: {len(keywords)} keywords")
            except Exception as e:
                print(f"❌ {words_file}: Error reading - {e}")
        else:
            print(f"ℹ️ {words_file}: Not found")

def main():
    """Main test function"""
    print("🧪 MARV Content Moderation Engine - Integration Test")
    print("=" * 60)
    
    # Check files
    check_database()
    check_words_json()
    
    # Test API
    if not test_api_health():
        print("\n❌ API is not running. Please start the backend first:")
        print("   python start_system.py")
        return
    
    # Test endpoints
    moderation_ok = test_moderation_endpoint()
    posts_ok = test_posts_endpoint()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    if moderation_ok and posts_ok:
        print("✅ All tests passed! Your system is working correctly.")
        print("\n🎯 Next steps:")
        print("1. Open frontend/index.html in your browser")
        print("2. Try submitting content for moderation")
        print("3. Check the posts table for results")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure the backend is running: python start_system.py")
        print("2. Check if port 8000 is available")
        print("3. Verify database setup: python scripts/setup_database.py --test")

if __name__ == "__main__":
    main() 