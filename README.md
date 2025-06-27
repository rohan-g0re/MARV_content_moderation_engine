# MARV Content Moderation Engine

A production-ready, multi-layered content moderation system with rule-based filtering, ML/AI detection, and LLM escalation.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend UI   │───▶│   FastAPI Backend│───▶│  Moderation     │
│   (React/HTML)  │    │   (Python 3.11+) │    │  Pipeline       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
            ┌───────▼──────┐   ┌───────▼──────┐
            │   Database   │   │   ML Models  │
            │ (SQLite/PG)  │   │(Detoxify/    │
            └──────────────┘   │ FinBERT/LLM) │
                               └──────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ (for React frontend)
- Docker (optional, for containerization)

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/rohan-g0re/MARV_content_moderation_engine.git
cd MARV_content_moderation_engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize Database
```bash
# Setup SQLite database with sample rules
python scripts/setup_database.py

# Or for PostgreSQL (production)
export DATABASE_URL="postgresql://user:pass@localhost/marv_db"
python scripts/setup_database.py --postgres
```

### 3. Start Backend API
```bash
# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Start Frontend (Optional)
```bash
cd frontend
npm install
npm start
```

## 📁 Project Structure

```
MARV_content_moderation_engine/
├── app/                          # FastAPI application
│   ├── api/                      # API routes
│   │   ├── v1/
│   │   │   ├── moderation.py     # Main moderation endpoint
│   │   │   ├── posts.py          # Post management
│   │   │   └── admin.py          # Admin endpoints
│   │   └── moderation.py         # Moderation result model
│   ├── core/                     # Core configuration
│   │   ├── config.py             # Settings management
│   │   ├── database.py           # Database connection
│   │   └── security.py           # Authentication
│   ├── models/                   # Database models
│   │   ├── post.py               # Post model
│   │   └── user.py               # User model
│   ├── services/                 # Business logic
│   │   ├── moderation_service.py # Main moderation pipeline
│   │   ├── ml_service.py         # ML model integration
│   │   ├── llm_service.py        # LLM integration
│   │   └── rule_service.py       # Rule-based filtering
│   └── utils/                    # Utilities
│       ├── text_processing.py    # Text normalization
│       └── validators.py         # Input validation
├── frontend/                     # React frontend
│   ├── src/
│   │   ├── components/           # React components
│   │   ├── pages/                # Page components
│   │   └── services/             # API client
├── scripts/                      # Utility scripts
│   ├── setup_database.py         # Database initialization
│   ├── generate_posts.py         # Test data generation
│   └── benchmark.py              # Performance testing
├── tests/                        # Test suite
│   ├── test_moderation.py        # Moderation tests
│   ├── test_api.py               # API tests
│   └── test_ml.py                # ML model tests
├── data/                         # Data files
│   ├── rules/                    # Moderation rules
│   ├── models/                   # ML model files
│   └── samples/                  # Test samples
├── docs/                         # Documentation
├── config/                       # Configuration files
└── docker/                       # Docker configuration
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:

```env
# Database
DATABASE_URL=sqlite:///./marv.db
# DATABASE_URL=postgresql://user:pass@localhost/marv_db

# ML Models
DETOXIFY_MODEL=original
FINBERT_MODEL=ProsusAI/finbert

# LLM Integration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Security
SECRET_KEY=your-secret-key-here
```

## 🧠 Moderation Pipeline

The system uses a multi-layered approach:

### Layer 1: Rule-Based Filtering
- **Regex patterns**: URLs, emails, phone numbers
- **Keyword matching**: Profanity, threats, spam
- **Severity scoring**: 1-10 scale per rule

### Layer 2: ML/AI Detection
- **Detoxify**: Toxicity detection (0-1 score)
- **FinBERT**: Financial sentiment analysis
- **Custom models**: Domain-specific classifiers

### Layer 3: LLM Escalation (Optional)
- **Ollama/LLaMA 3.1**: Complex reasoning for borderline cases
- **Threshold-based**: Only for uncertain cases (10-15% of posts)
- **Cost-aware**: Minimizes API calls

### Decision Logic
```python
def determine_action(threat_level, ml_scores, llm_feedback):
    if threat_level == "CRITICAL":
        return "BLOCK"
    elif threat_level == "HIGH":
        return "QUARANTINE"
    elif threat_level == "MEDIUM":
        return "FLAG"
    else:
        return "ACCEPT"
```

## 📊 API Endpoints

### Core Endpoints
- `POST /api/v1/moderate` - Main moderation endpoint
- `GET /api/v1/posts` - List all posts
- `GET /api/v1/posts/{id}` - Get specific post
- `POST /api/v1/posts/{id}/feedback` - Submit feedback

### Admin Endpoints
- `GET /api/v1/admin/stats` - System statistics
- `POST /api/v1/admin/rules` - Add/modify rules
- `GET /api/v1/admin/logs` - System logs

### Example Usage
```bash
# Moderate content
curl -X POST "http://localhost:8000/api/v1/moderate" \
     -H "Content-Type: application/json" \
     -d '{
       "content": "This is a test post",
       "user_id": "user123",
       "content_type": "text"
     }'

# Get moderation result
{
  "post_id": "post_123",
  "action": "ACCEPT",
  "threat_level": "LOW",
  "confidence": 0.95,
  "explanation": "Content passed all checks",
  "processing_time_ms": 45,
  "metadata": {
    "rule_matches": 0,
    "toxicity_score": 0.1,
    "fraud_score": 0.05
  }
}
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Generate Test Data
```bash
python scripts/generate_posts.py --count 100 --output data/samples/test_posts.json
```

### Performance Benchmark
```bash
python scripts/benchmark.py --posts data/samples/test_posts.json
```

### Manual Testing
```bash
# Start the API
uvicorn app.main:app --reload

# Test with sample posts
python scripts/test_moderation.py
```

## 🐳 Docker Deployment

### Development
```bash
docker-compose -f docker/docker-compose.dev.yml up --build
```

### Production
```bash
docker-compose -f docker/docker-compose.prod.yml up --build
```

## 📈 Monitoring & Logging

### Metrics
- Processing time per post
- Accuracy by content type
- LLM escalation frequency
- Error rates

### Logs
- Structured JSON logging
- Request/response tracking
- Error stack traces
- Performance metrics

## 🔄 Adding New Moderation Layers

### 1. Create Service
```python
# app/services/custom_service.py
class CustomModerationService:
    def __init__(self):
        self.model = load_model()
    
    async def analyze(self, content: str) -> Dict:
        # Your custom logic
        return {"score": 0.5, "confidence": 0.8}
```

### 2. Integrate into Pipeline
```python
# app/services/moderation_service.py
class ModerationService:
    def __init__(self):
        self.custom_service = CustomModerationService()
    
    async def moderate_content(self, content: str) -> ModerationResult:
        # Add to pipeline
        custom_result = await self.custom_service.analyze(content)
        # ... rest of pipeline
```

### 3. Update Configuration
```python
# app/core/config.py
class Settings:
    ENABLE_CUSTOM_MODERATION: bool = True
    CUSTOM_MODERATION_WEIGHT: float = 0.3
```

## 🔒 Security Considerations

- Input validation and sanitization
- Rate limiting on API endpoints
- Authentication for admin functions
- Secure database connections
- Model file integrity checks

## 🚀 Performance Optimization

- Async processing for ML models
- Database connection pooling
- Caching for frequent queries
- Model quantization for faster inference
- Horizontal scaling with load balancers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation in `/docs`
- Review the test examples

---

**Built with ❤️ for safe online communities** 