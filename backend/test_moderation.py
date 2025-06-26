import asyncio
import httpx
import json

# Test cases
TEST_CASES = [
    {
        "name": "Clean content",
        "content": "Hello, this is a nice post about technology and innovation.",
        "expected": "accept"
    },
    {
        "name": "Profanity filter",
        "content": "This post contains badword1 which should be filtered.",
        "expected": "reject"
    },
    {
        "name": "Spam pattern",
        "content": "Buy now at http://spam.com for amazing deals!",
        "expected": "reject"
    },
    {
        "name": "Email pattern",
        "content": "Contact me at spam@example.com for more information.",
        "expected": "reject"
    },
    {
        "name": "Phone number pattern",
        "content": "Call me at 12345678901 for details.",
        "expected": "reject"
    }
]

async def test_moderation():
    """Test the moderation endpoint with various content types"""
    backend_url = "http://localhost:8000/api/moderate"
    
    print("🧪 Testing Content Moderation System")
    print("=" * 50)
    
    async with httpx.AsyncClient() as client:
        for i, test_case in enumerate(TEST_CASES, 1):
            print(f"\n{i}. Testing: {test_case['name']}")
            print(f"   Content: {test_case['content'][:50]}...")
            
            try:
                response = await client.post(
                    backend_url,
                    json={"content": test_case['content']},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    status = "✅ PASS" if result["accepted"] == (test_case["expected"] == "accept") else "❌ FAIL"
                    print(f"   Status: {status}")
                    print(f"   Result: {'Accepted' if result['accepted'] else 'Rejected'}")
                    print(f"   Reason: {result['reason']}")
                    print(f"   Record ID: {result['id']}")
                else:
                    print(f"   ❌ HTTP Error: {response.status_code}")
                    print(f"   Response: {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎯 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_moderation()) 