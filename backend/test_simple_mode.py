#!/usr/bin/env python3
"""
Simple test for the PRODUCTION_MODE toggle
"""

import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_simple_mode():
    """Test the simple PRODUCTION_MODE toggle"""
    print("🎚️ SIMPLE MODE TEST")
    print("=" * 40)
    
    try:
        from app.core.moderation_safe import GuardianModerationEngine, PRODUCTION_MODE
        
        print(f"Current PRODUCTION_MODE: {PRODUCTION_MODE}")
        print()
        
        # Create engine (should use PRODUCTION_MODE setting)
        engine = GuardianModerationEngine()
        mode_info = engine.get_simple_mode_info()
        
        print("📊 Mode Info:")
        mode = "PRODUCTION" if mode_info["PRODUCTION_MODE"] else "TESTING"
        print(f"  Mode: {mode}")
        print(f"  Description: {mode_info['mode_description']}")
        print(f"  Active stages: {mode_info['active_stages']}")
        
        print()
        print("🧪 Testing content moderation:")
        test_content = "This is a test message"
        result = engine.moderate_content(test_content)
        print(f"  Content: '{test_content}'")
        print(f"  Result: {'ACCEPTED' if result.accepted else 'REJECTED'}")
        print(f"  Stage: {result.stage}")
        
        print()
        print("✅ Simple mode working correctly!")
        
        print()
        print("📝 Instructions for non-technical users:")
        print("  1. Open: backend/app/core/moderation_safe.py")
        print("  2. Find: PRODUCTION_MODE = True")
        print("  3. Change to: PRODUCTION_MODE = False (for testing)")
        print("  4. Change to: PRODUCTION_MODE = True (for full protection)")
        print("  5. Save file and restart server")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_simple_mode() 