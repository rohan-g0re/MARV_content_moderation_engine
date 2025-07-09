# GuardianAI Stage Toggles Feature

## Overview

The stage toggles feature allows you to manually control which stages of the moderation pipeline are active. This is useful for testing, debugging, and understanding how different stages contribute to the final moderation decision.

## Pipeline Stages

1. **Stage 1: Rule-based filtering** - Keywords and regex patterns
2. **Stage 2: LGBM machine learning model** - Financial scam/fraud detection
3. **Stage 3: Detoxify AI** - Toxicity detection using AI
4. **Stage 4: FinBERT AI** - Financial fraud detection using AI

## How to Control Stages

### Method 1: Environment Variables

Set environment variables before starting the server:

```bash
# Windows PowerShell
$env:STAGE1_ENABLED="true"
$env:STAGE2_ENABLED="false"
$env:STAGE3_ENABLED="true"
$env:STAGE4_ENABLED="false"
python main.py

# Linux/Mac
export STAGE1_ENABLED=true
export STAGE2_ENABLED=false
export STAGE3_ENABLED=true
export STAGE4_ENABLED=false
python main.py
```

### Method 2: Direct Engine Configuration

```python
from app.core.moderation_safe import GuardianModerationEngine

# Create engine with specific stages enabled
engine = GuardianModerationEngine(
    stage1_enabled=True,
    stage2_enabled=False,
    stage3_enabled=True,
    stage4_enabled=False
)

# Dynamically update toggles
engine.set_stage_toggles(stage1=False, stage2=True)
```

### Method 3: API Endpoints

#### Get current configuration:
```bash
curl -X GET http://localhost:8000/stage-config
```

#### Update stage configuration:
```bash
curl -X POST http://localhost:8000/stage-config \
  -H "Content-Type: application/json" \
  -d '{
    "stage1_enabled": true,
    "stage2_enabled": false,
    "stage3_enabled": true,
    "stage4_enabled": false
  }'
```

#### Test all stage combinations:
```bash
curl -X POST http://localhost:8000/test-stages \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your test content here"
  }'
```

## Available API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stage-config` | GET | Get current stage configuration |
| `/stage-config` | POST | Update stage toggles |
| `/test-stages` | POST | Test content through all stage combinations |

## Testing Script

Run the provided test script to verify the stage toggles functionality:

```bash
cd backend
python test_stage_toggles.py
```

The test script will:
- Test direct engine usage with different stage combinations
- Test environment variable control
- Test API endpoints (requires running server)
- Show usage examples

## Common Testing Scenarios

### Test Individual Stages

1. **Only Rule-based (Stage 1)**:
   ```json
   {
     "stage1_enabled": true,
     "stage2_enabled": false,
     "stage3_enabled": false,
     "stage4_enabled": false
   }
   ```

2. **Only LGBM (Stage 2)**:
   ```json
   {
     "stage1_enabled": false,
     "stage2_enabled": true,
     "stage3_enabled": false,
     "stage4_enabled": false
   }
   ```

3. **All Stages**:
   ```json
   {
     "stage1_enabled": true,
     "stage2_enabled": true,
     "stage3_enabled": true,
     "stage4_enabled": true
   }
   ```

4. **No Stages (Pass Everything)**:
   ```json
   {
     "stage1_enabled": false,
     "stage2_enabled": false,
     "stage3_enabled": false,
     "stage4_enabled": false
   }
   ```

### Test Content Examples

- **Rule-based trigger**: "This is a scam!"
- **Financial content**: "Guaranteed 100% returns on your investment!"
- **Toxic content**: "I hate everyone"
- **Clean content**: "Hello, how are you today?"

## Response Format

When testing stages, you'll see responses like:

```json
{
  "accepted": false,
  "reason": "Rule-based: scam",
  "stage": "rule-based",
  "band": "BLOCK",
  "action": "BLOCK",
  "threat_level": "high",
  "confidence": 1.0
}
```

## Troubleshooting

1. **LGBM not working**: Ensure LightGBM is installed and model files exist
2. **AI stages not working**: Check if transformers and torch are installed
3. **API not responding**: Make sure the server is running on the correct port
4. **Environment variables not working**: Restart the server after setting variables

## Example Workflow

1. Start with all stages enabled to see normal behavior
2. Disable individual stages to see their impact
3. Test specific content that should trigger certain stages
4. Use the `/test-stages` endpoint for comprehensive analysis
5. Compare results between different stage combinations

This feature makes it easy to understand exactly how your content moderation pipeline is working and which stages are most effective for different types of content.