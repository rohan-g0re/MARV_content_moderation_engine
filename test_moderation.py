#!/usr/bin/env python3
"""
Test script for Content Moderation System

PowerShell Commands for manual testing:
# Test moderation
Invoke-RestMethod -Uri "http://localhost:8000/moderate" -Method POST -ContentType "application/json" -Body '{"content": "You are a scammer and I hate this!"}'

# Get all posts
Invoke-RestMethod -Uri "http://localhost:8000/posts" -Method GET

# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET
"""

import requests
import json
import time

# API endpoint
API_URL = "http://localhost:8000"

def test_moderation():
    """Test the moderation endpoint with various content"""
    
    test_cases = [
        {
            "content": "This is a normal post about technology.",
            "expected": "accepted"
        },
        {
            "content": "You are a scammer and I hate this!",
            "expected": "rejected"
        },
        {
            "content": "I love this product, it's amazing!",
            "expected": "accepted"
        },
        {
            "content": "This is fraud and illegal activity.",
            "expected": "rejected"
        },
        {
            "content": "Check out this website: http://spam.com",
            "expected": "rejected"
        },
        {
            "content": "This contains some bad words that should be filtered.",
            "expected": "rejected"
        }
    ]
    
    print("🧪 Testing Content Moderation System")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['content'][:50]}...")
        
        try:
            response = requests.post(
                f"{API_URL}/moderate",
                json={"content": test_case["content"]},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                status = "✅ PASS" if result["accepted"] == (test_case["expected"] == "accepted") else "❌ FAIL"
                print(f"   Status: {status}")
                print(f"   Result: {'Accepted' if result['accepted'] else 'Rejected'}")
                print(f"   Reason: {result['reason']}")
                print(f"   ID: {result['id']}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("   ❌ Connection Error: Make sure the backend is running on http://localhost:8000")
            break
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Testing completed!")

def test_get_posts():
    """Test getting all posts"""
    print("\n📊 Testing GET /posts endpoint")
    print("-" * 30)
    
    try:
        response = requests.get(f"{API_URL}/posts")
        
        if response.status_code == 200:
            posts = response.json()
            print(f"✅ Found {len(posts)} posts in database")
            
            if posts:
                print("\nRecent posts:")
                for post in posts[:3]:  # Show last 3 posts
                    status = "✅ Accepted" if post["accepted"] else "❌ Rejected"
                    print(f"   ID {post['id']}: {status} - {post['reason']}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the backend is running")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_health():
    """Test health endpoint"""
    print("\n🏥 Testing health endpoint")
    print("-" * 30)
    
    try:
        response = requests.get(f"{API_URL}/health")
        
        if response.status_code == 200:
            print("✅ Backend is healthy!")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Backend not running")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Content Moderation Tests...")
    
    # Test health first
    test_health()
    
    # Test moderation
    test_moderation()
    
    # Test getting posts
    test_get_posts()
    
    print("\n✨ All tests completed!")
    print("\nTo run the frontend, open frontend/index.html in your browser")
    print("Or visit: http://localhost:8000/docs for API documentation")
    print("\nPowerShell Commands for manual testing:")
    print('Invoke-RestMethod -Uri "http://localhost:8000/moderate" -Method POST -ContentType "application/json" -Body \'{"content": "test content"}\'') 