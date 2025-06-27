#!/usr/bin/env python3
"""
GuardianAI Integration Example
Shows how to integrate GuardianAI Core Pipeline with FastAPI backend
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from guardian_core import GuardianAICore, ContentType, ModerationResult
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

# === FastAPI Models ===
class ModerationRequest(BaseModel):
    content: str
    content_type: Optional[str] = "text"
    content_id: Optional[str] = None

class ModerationResponse(BaseModel):
    content_id: str
    threat_level: str
    action: str
    explanation: str
    confidence: float
    processing_time_ms: int
    accepted: bool
    metadata: dict

# === GuardianAI Integration ===
class GuardianAIAPI:
    """FastAPI integration for GuardianAI Core Pipeline"""
    
    def __init__(self):
        self.app = FastAPI(
            title="GuardianAI Content Moderation API",
            description="Advanced content moderation with structured output",
            version="1.0.0"
        )
        self.guardian = GuardianAICore()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        def root():
            return {
                "message": "GuardianAI Content Moderation API",
                "version": "1.0.0",
                "status": "running"
            }
        
        @self.app.get("/health")
        def health():
            return {"status": "healthy", "guardian_status": "initialized"}
        
        @self.app.post("/moderate", response_model=ModerationResponse)
        def moderate_content(request: ModerationRequest):
            """Main moderation endpoint using GuardianAI Core Pipeline"""
            try:
                # Convert content type string to enum
                content_type = ContentType.TEXT
                if request.content_type:
                    try:
                        content_type = ContentType(request.content_type)
                    except ValueError:
                        content_type = ContentType.TEXT
                
                # Run GuardianAI moderation
                result: ModerationResult = self.guardian.moderate_content(
                    content=request.content,
                    content_type=content_type,
                    content_id=request.content_id
                )
                
                # Convert to API response format
                return ModerationResponse(
                    content_id=result.content_id,
                    threat_level=result.threat_level.value,
                    action=result.action.value,
                    explanation=result.explanation,
                    confidence=result.confidence,
                    processing_time_ms=result.processing_time_ms,
                    accepted=result.action.value in ["accept", "flag"],
                    metadata=result.metadata
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/results/{content_id}")
        def get_moderation_result(content_id: str):
            """Get moderation result by content ID"""
            try:
                # This would query the database for stored results
                # For now, return a placeholder
                return {
                    "content_id": content_id,
                    "message": "Result retrieval not implemented in this demo",
                    "note": "Results are stored in guardian.db"
                }
            except Exception as e:
                raise HTTPException(status_code=404, detail="Result not found")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the FastAPI server"""
        uvicorn.run(self.app, host=host, port=port)
    
    def close(self):
        """Clean up resources"""
        self.guardian.close()

# === CLI Interface ===
def run_cli_demo():
    """Run a CLI demo of the GuardianAI Core Pipeline"""
    print("🛡️ GuardianAI Core Pipeline CLI Demo")
    print("=" * 50)
    
    guardian = GuardianAICore()
    
    while True:
        print("\n📝 Enter content to moderate (or 'quit' to exit):")
        content = input("> ").strip()
        
        if content.lower() in ['quit', 'exit', 'q']:
            break
        
        if not content:
            print("⚠️ Please enter some content")
            continue
        
        print("\n🔄 Processing...")
        
        try:
            result = guardian.moderate_content(content)
            
            print("\n📊 MODERATION RESULT:")
            print(f"🆔 Content ID: {result.content_id}")
            print(f"⚠️ Threat Level: {result.threat_level.value.upper()}")
            print(f"🛡️ Action: {result.action.value.upper()}")
            print(f"📊 Confidence: {result.confidence:.2f}")
            print(f"⏱️ Processing Time: {result.processing_time_ms}ms")
            print(f"💬 Explanation: {result.explanation}")
            
            # Color-coded output
            if result.threat_level.value == "low":
                print("✅ Content appears safe")
            elif result.threat_level.value == "medium":
                print("⚠️ Content flagged for review")
            elif result.threat_level.value == "high":
                print("🚨 Content blocked - high threat detected")
            else:
                print("🚨 CRITICAL THREAT - Immediate action required")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    guardian.close()
    print("\n👋 Goodbye!")

def run_api_server():
    """Run the GuardianAI API server"""
    print("🚀 Starting GuardianAI API Server...")
    api = GuardianAIAPI()
    
    try:
        api.run(host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    finally:
        api.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GuardianAI Core Pipeline")
    parser.add_argument("--mode", choices=["cli", "api"], default="cli",
                       help="Run mode: cli (interactive) or api (server)")
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    
    args = parser.parse_args()
    
    if args.mode == "cli":
        run_cli_demo()
    elif args.mode == "api":
        print(f"🚀 Starting GuardianAI API Server on {args.host}:{args.port}")
        api = GuardianAIAPI()
        try:
            api.run(host=args.host, port=args.port)
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
        finally:
            api.close() 