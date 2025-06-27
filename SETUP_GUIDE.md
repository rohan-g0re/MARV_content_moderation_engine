# MARV Content Moderation Engine - Setup Guide

## 🚀 Quick Start (5 minutes)

### 1. Prerequisites
- Python 3.11+
- Git
- 4GB+ RAM (for ML models)

### 2. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/rohan-g0re/MARV_content_moderation_engine.git
cd MARV_content_moderation_engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Initialize Database
```bash
# Setup database with sample rules
python scripts/setup_database.py --test
```

### 4. Start the System
```bash
# Option 1: Use the startup script
python start_system.py

# Option 2: Manual start
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test the System
```bash
# Run comprehensive tests
python test_system.py
```

### 6. Access the System
- **Frontend**: Open `frontend/index.html` in your browser
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
MARV_content_moderation_engine/
├── app/                          # Main application
│   ├── api/v1/                   # API endpoints
│   │   ├── moderation.py         # Main moderation endpoint
│   │   ├── posts.py              # Post management
│   │   └── admin.py              # Admin functions
│   ├── core/                     # Core configuration
│   │   ├── config.py             # Settings management
│   │   └── database.py           # Database connection
│   ├── models/                   # Database models
│   │   ├── post.py               # Post and rule models
│   │   └── schemas.py            # Pydantic schemas
│   ├── services/                 # Business logic
│   │   ├── moderation_service.py # Main pipeline
│   │   ├── rule_service.py       # Rule-based filtering
│   │   ├── ml_service.py         # ML/AI detection
│   │   └── llm_service.py        # LLM integration
│   └── main.py                   # FastAPI application
├── frontend/                     # HTML frontend
│   └── index.html                # Main UI
├── scripts/                      # Utility scripts
│   └── setup_database.py         # Database initialization
├── tests/                        # Test suite
│   └── test_moderation.py        # Comprehensive tests
├── docker/                       # Docker configuration
├── requirements.txt              # Python dependencies
├── start_system.py               # Startup script
└── test_system.py                # Quick test script
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:

```env
# Database
DATABASE_URL=sqlite:///./marv.db

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# ML Models
DETOXIFY_MODEL=original
FINBERT_MODEL=ProsusAI/finbert

# LLM Integration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
ENABLE_LLM=True

# Moderation Thresholds
TOXICITY_THRESHOLD=0.5
FRAUD_THRESHOLD=0.7
RULE_SEVERITY_THRESHOLD=5
```

## 🧠 Moderation Pipeline

The system uses a **3-layer approach**:

### Layer 1: Rule-Based Filtering
- **Keywords**: Profanity, threats, fraud terms
- **Regex**: URLs, emails, phone numbers
- **Severity**: 1-10 scale per rule

### Layer 2: ML/AI Detection
- **Detoxify**: Toxicity detection (0-1 score)
- **FinBERT**: Financial sentiment analysis
- **Confidence**: Model confidence scores

### Layer 3: LLM Escalation (Optional)
- **Ollama/LLaMA 3.1**: Complex reasoning
- **Threshold-based**: Only for uncertain cases
- **Cost-aware**: Minimizes API calls

## 📊 API Endpoints

### Core Endpoints
```bash
# Moderate content
POST /api/v1/moderate
{
  "content": "Text to moderate",
  "content_type": "text",
  "user_id": "user123"
}

# Get posts
GET /api/v1/posts?limit=10

# Get specific post
GET /api/v1/moderate/{post_id}

# Submit feedback
POST /api/v1/moderate/{post_id}/feedback
```

### Admin Endpoints
```bash
# System statistics
GET /api/v1/admin/stats

# Manage rules
GET /api/v1/admin/rules
POST /api/v1/admin/rules
PUT /api/v1/admin/rules/{rule_id}
DELETE /api/v1/admin/rules/{rule_id}
```

### Example Response
```json
{
  "post_id": 1,
  "action": "block",
  "threat_level": "high",
  "confidence": 0.85,
  "explanation": "Toxicity detected: 0.78; Rule violations detected: profanity",
  "processing_time_ms": 45,
  "metadata": {
    "layer_results": {
      "rule": {...},
      "ml": {...},
      "llm": {...}
    }
  }
}
```

## 🧪 Testing

### Quick Test
```bash
# Run comprehensive test suite
python test_system.py
```

### Manual Testing
```bash
# Test moderation endpoint
curl -X POST "http://localhost:8000/api/v1/moderate" \
     -H "Content-Type: application/json" \
     -d '{
       "content": "This is a test message with damn words",
       "content_type": "text",
       "user_id": "test_user"
     }'
```

### Test Cases
The system includes test cases for:
- ✅ Clean content (should accept)
- ⚠️ Mild profanity (should flag)
- ❌ Strong profanity (should block)
- 🚨 Violent threats (should block)
- 💰 Financial fraud (should block)
- 📧 Personal information (should flag)

## 🐳 Docker Deployment

### Development
```bash
# Build and run with Docker Compose
cd docker
docker-compose up --build
```

### Production
```bash
# Build production image
docker build -f docker/Dockerfile -t marv-moderation .

# Run with environment variables
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host/db \
  -e DEBUG=False \
  marv-moderation
```

## 🔄 Adding New Features

### 1. Add New Moderation Rules
```python
# In scripts/setup_database.py
new_rule = {
    "id": "custom_rule_1",
    "rule_type": "custom",
    "pattern": "your_pattern",
    "severity": 5,
    "is_regex": False,
    "description": "Custom rule description",
    "category": "custom"
}
```

### 2. Add New ML Models
```python
# In app/services/ml_service.py
async def _analyze_custom(self, content: str):
    # Your custom ML logic
    return {"score": 0.5, "confidence": 0.8}
```

### 3. Add New API Endpoints
```python
# In app/api/v1/moderation.py
@router.post("/custom-endpoint")
async def custom_endpoint():
    # Your custom logic
    pass
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

#### 2. Database Errors
```bash
# Solution: Reset database
python scripts/setup_database.py --force-reset
```

#### 3. ML Models Not Loading
```bash
# Solution: Check internet connection and try again
# Models are downloaded automatically on first use
```

#### 4. LLM Service Not Working
```bash
# Solution: Install and start Ollama
# Visit: https://ollama.ai
```

#### 5. Port Already in Use
```bash
# Solution: Change port in .env file
API_PORT=8001
```

### Performance Optimization

#### 1. Enable Caching
```python
# Add Redis caching
REDIS_URL=redis://localhost:6379
```

#### 2. Optimize ML Models
```python
# Use quantized models
DETOXIFY_MODEL=quantized
```

#### 3. Database Optimization
```python
# Use PostgreSQL for production
DATABASE_URL=postgresql://user:pass@host/db
```

## 📈 Monitoring

### Health Checks
```bash
# Check system health
curl http://localhost:8000/health
```

### Metrics
```bash
# Get system statistics
curl http://localhost:8000/api/v1/stats
```

### Logs
```bash
# View application logs
tail -f logs/app.log
```

## 🔒 Security

### Production Checklist
- [ ] Change default SECRET_KEY
- [ ] Use HTTPS
- [ ] Enable rate limiting
- [ ] Set up authentication
- [ ] Use PostgreSQL
- [ ] Enable logging
- [ ] Set DEBUG=False

### Security Headers
```python
# Add security middleware
app.add_middleware(SecurityMiddleware)
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black app/
flake8 app/
```

### Code Style
- Use type hints
- Add docstrings
- Follow PEP 8
- Write tests for new features

## 📞 Support

### Getting Help
1. Check the troubleshooting section
2. Review the test cases
3. Check the API documentation
4. Open an issue on GitHub

### Useful Commands
```bash
# Quick system check
python test_system.py

# Start development server
uvicorn app.main:app --reload

# Run specific tests
pytest tests/test_moderation.py -v

# Check database
python scripts/setup_database.py --test
```

---

**🎉 You're ready to build safe online communities with MARV!** 