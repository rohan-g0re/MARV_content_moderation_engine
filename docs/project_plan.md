# Content Moderation Engine - Project Plan & Technical Roadmap

## 🎯 Project Overview

**Client:** UnBound X  
**Objective:** Build a cost-efficient, scalable content moderation system using layered filtering  
**Timeline:** 13-day sprint  
**Delivery:** Daily commits + final walkthrough + comprehensive documentation

### Core Mission
Design a system that triages user-generated posts using deterministic filtering, machine learning, and lightweight agentic layers to ensure cost-aware content moderation.

---

## 🏗️ Architecture Blueprint

### System Components
1. **Feed Generator** - Synthetic post creation for training/testing
2. **Database-Driven Rule Filter** - Fast keyword/pattern lookup
3. **AI Layer 1** - Llama 3.1 via Ollama for nuanced analysis
4. **Agentic Layer** - Advanced reasoning for edge cases
5. **Feedback & Override System** - Human-in-the-loop corrections

### Data Flow Pipeline
```
User Post → Rule Filter → AI Layer 1 → Agentic Layer → Decision + Explanation
                ↓              ↓           ↓
           Fast Block     Escalation   Final Review
```

---

## 🛠️ Technical Stack & Architecture

### Backend Stack
- **Language:** Python 3.11+
- **Framework:** FastAPI (async/await support)
- **Database:** SQLite (dev) → PostgreSQL (production)
- **ML Pipeline:** HuggingFace Transformers
- **LLM Integration:** Ollama + Llama 3.1

### Frontend Stack
- **Framework:** React 18+
- **UI Components:** Modern, accessible design
- **State Management:** Context API / Redux (if needed)
- **API Integration:** Axios/Fetch for backend communication

### AI/ML Components
- **Content Generation:** Google Gemini API
- **Text Classification:** Transformer models
- **Embeddings:** sentence-transformers
- **Traditional ML:** scikit-learn for baseline models

### Infrastructure
- **Development:** Local Ollama + SQLite
- **Production Ready:** PostgreSQL + cloud deployment
- **Monitoring:** Structured logging + metrics

---

## 📅 Detailed Sprint Plan

### **Day 1: Feed Generation** ✅ (Current)
**Focus:** Data foundation for training and testing

**Deliverables:**
- ✅ Modular project structure
- 🔄 Gemini-based data generation script
- 🔄 Python synthetic data generation script
- 🔄 10,000+ labeled samples total
- ✅ Development environment setup

**Technical Requirements:**
- Schema: `Post (text, varying length)` + `Label (5-7 band classification)`
- Classification bands: `SAFE`, `QUESTIONABLE`, `INAPPROPRIATE`, `HARMFUL`, `ILLEGAL`
- 1-second delay between Gemini API calls
- Balanced dataset across all severity levels

**Success Criteria:**
- 2 working scripts generating diverse content
- Consistent data format and quality
- Ready for immediate ML training

---

### **Day 2-3: Rule-Based Filtering (GuardianAI v1)**
**Focus:** Fast, deterministic content filtering

**Deliverables:**
- DatabaseFilter class with SQLite backend
- Regex patterns for common violations
- Keyword scoring system
- Profanity and severity lookup tables

**Technical Implementation:**
- SQL-based keyword matching for speed
- Configurable scoring thresholds
- Pattern matching for URLs, emails, phone numbers
- Expandable dictionary system

---

### **Day 4: Basic Post Viewer UI**
**Focus:** User interface foundation

**Deliverables:**
- React app with post submission form
- Moderation result display (Accept/Flag/Block)
- Basic admin dashboard
- API integration layer

**UI Requirements:**
- Clean, modern interface
- Real-time moderation feedback
- Admin override capabilities
- Responsive design

---

### **Day 5: GuardianAI Core Pipeline**
**Focus:** Unified moderation controller

**Deliverables:**
- `moderate_content()` main entrypoint
- Structured output format
- Pipeline orchestration
- Result explanation system

**Core Features:**
- Threat level scoring (0-5 scale)
- Action recommendation logic
- Human-readable explanations
- Confidence scoring

---

### **Day 6: LLM Escalation Logic**
**Focus:** AI-powered nuanced analysis

**Deliverables:**
- Llama 3.1 integration via Ollama
- Escalation threshold system
- Hybrid decision logic
- Cost optimization (use LLM for ~10-15% of posts)

**Implementation:**
- Borderline case detection
- LLM prompt engineering
- Fallback mechanisms
- Performance monitoring

---

### **Day 7-8: AI Integration Testing**
**Focus:** Comprehensive system validation

**Deliverables:**
- Benchmark test suite
- Latency measurement tools
- Escalation frequency analysis
- Confidence heatmap generation

**Test Categories:**
- Clean content (various topics)
- Spam detection
- Fraud attempts
- Profanity variants
- Nuanced edge cases

---

### **Day 9: Phrase Suggestions (Optional)**
**Focus:** User experience enhancement

**Deliverables:**
- Alternative phrase suggestions
- Context-aware recommendations
- UI integration for suggestions
- User feedback loop

---

### **Day 10: Dictionary Expansion**
**Focus:** Automated content understanding improvement

**Deliverables:**
- FastText/Word2Vec synonym discovery
- Candidate term generation
- Moderator review interface
- Dictionary update pipeline

---

### **Day 11: Feedback & Override System**
**Focus:** Human-in-the-loop learning

**Deliverables:**
- User feedback capture
- Admin override interface
- Correction logging system
- Model improvement pipeline

---

### **Day 12: QA Pass & Logging**
**Focus:** Production readiness

**Deliverables:**
- Full regression test suite
- Comprehensive error handling
- Production logging setup
- LLM fallback recovery

---

### **Day 13: Documentation & Demo**
**Focus:** Project completion and handoff

**Deliverables:**
- Technical documentation
- System architecture diagrams
- API documentation
- Live demonstration
- Clean GitHub repository

---

## 🎯 Success Metrics

### Technical Metrics
- **Accuracy:** 95%+ correct classification on test set
- **Latency:** <200ms average response time for rule-based filtering
- **Cost Efficiency:** LLM usage for only 10-15% of posts
- **Scalability:** Handle 1000+ posts/minute

### Business Metrics
- **False Positive Rate:** <5% for legitimate content
- **False Negative Rate:** <2% for harmful content
- **User Experience:** Clear explanations for all moderation actions
- **Maintainability:** Easy to add new rules and patterns

---

## 🔧 Development Standards

### Code Quality
- Type hints for all Python functions
- Comprehensive docstrings
- Unit tests for all core functionality
- Black formatting + flake8 linting

### Documentation
- API documentation with examples
- Architecture decision records
- Setup and deployment guides
- Troubleshooting documentation

### Git Workflow
- Feature branches for each day's work
- Detailed commit messages
- Daily progress commits
- Clean history for final handoff

---

## 🚀 Future Enhancements (Post-Sprint)

### Phase 2 Possibilities
- Multi-language support
- Image/video content moderation
- Real-time streaming processing
- Advanced ML model fine-tuning
- Enterprise authentication
- Detailed analytics dashboard

### Scalability Considerations
- Microservices architecture
- Container deployment
- Load balancing
- Caching strategies
- Database sharding

---

**Document Version:** 1.0  
**Last Updated:** Day 1 - Session 1  
**Next Review:** End of Day 3  

**Note for Future AI Agents:** This document serves as the complete technical roadmap. Check `docs/progress.md` for daily updates and `docs/errors.md` for troubleshooting history. 