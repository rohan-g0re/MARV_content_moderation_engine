# 🧩 Content Moderation System

A simple, clean content moderation system with three-layer filtering: rule-based, Detoxify toxicity detection, and FinBERT fraud detection.

## ✨ Features

- **Three-Layer Moderation Pipeline**:
  1. **Rule-based filtering** (keywords from `words.json` + regex patterns)
  2. **Detoxify** toxicity detection
  3. **FinBERT** financial fraud detection

- **Clean UI**: Simple HTML frontend with real-time results
- **Database Storage**: SQLite with all moderation history
- **RESTful API**: FastAPI backend with automatic documentation

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Run the Backend

```powershell
cd backend
python main.py
```

The API will be available at: http://localhost:8000

### 3. Open the Frontend

Open `frontend/index.html` in your browser, or serve it with:

```powershell
# Using Python
python -m http.server 8080

# Then visit: http://localhost:8080/frontend/index.html
```

### 4. Test the System

```powershell
python test_moderation.py
```

## 📋 API Endpoints

### POST /moderate
Moderate content through the three-layer pipeline.

**Request:**
```json
{
  "content": "Your text content here"
}
```

**Response:**
```json
{
  "accepted": false,
  "reason": "Rule-based: scammer",
  "id": 1
}
```

### GET /posts
Get all moderated posts with their results.

### GET /health
Health check endpoint.

## 🧪 Testing

### Manual Testing with PowerShell

```powershell
# Test moderation
Invoke-RestMethod -Uri "http://localhost:8000/moderate" -Method POST -ContentType "application/json" -Body '{"content": "You are a scammer and I hate this!"}'

# Get all posts
Invoke-RestMethod -Uri "http://localhost:8000/posts" -Method GET
```

### Alternative using curl (if available)

```powershell
# Test moderation
curl -X POST http://localhost:8000/moderate -H "Content-Type: application/json" -d '{"content": "You are a scammer and I hate this!"}'

# Get all posts
curl http://localhost:8000/posts
```

### Test Cases

The system includes test cases for:
- Normal content (should be accepted)
- Rule-based violations (profanity from `words.json`)
- Toxic content (high toxicity scores)
- Financial fraud indicators
- URL/email patterns

## 📁 Project Structure

```
├── backend/
│   └── main.py              # FastAPI backend
├── frontend/
│   └── index.html           # Simple HTML UI
├── words.json               # Profanity keywords (2700+ words)
├── requirements.txt         # Python dependencies
├── test_moderation.py       # Test script
└── README.md               # This file
```

## ⚙️ Configuration

### Keywords File
The system uses `words.json` which contains 2700+ profanity and inappropriate words. You can:
- Add new words to the JSON array
- Remove words you don't want to filter
- Replace with your own custom list

### Model Thresholds
Modify thresholds in `backend/main.py`:
- Detoxify toxicity threshold: `toxicity_score > 0.5`
- FinBERT fraud threshold: `confidence > 0.7`

## 🔧 Troubleshooting

### Common Issues

1. **Models not loading**: The system will fall back to rule-based filtering
2. **Database errors**: SQLite file will be created automatically
3. **CORS issues**: Backend includes CORS middleware for all origins

### Manual Database Inspection

```powershell
# Using SQLite CLI (if installed)
sqlite3 moderation.db
.tables
SELECT * FROM posts ORDER BY created_at DESC;
```

## 📊 Database Schema

```sql
CREATE TABLE posts (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    accepted BOOLEAN NOT NULL,
    reason TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🎯 Moderation Logic

1. **Rule-based Filter**: Checks 2700+ keywords from `words.json` and regex patterns
2. **Detoxify**: Uses HuggingFace `unitary/toxic-bert` model
3. **FinBERT**: Uses `ProsusAI/finbert` for financial sentiment

If any layer rejects the content, the post is rejected with the specific reason.

## 📝 License

MIT License - feel free to use and modify! 