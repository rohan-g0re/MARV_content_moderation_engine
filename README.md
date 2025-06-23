# Content Moderation Engine

**Project Brief: Content Moderation Engine**

**Client:** UnBound X  
**Team:** Intern Development Cohort  
**Delivery Format:** Daily commits + final walkthrough + documentation

**Objective**

Design and build a cost-efficient, scalable content moderation system using multiple filtering layers. The system should triage user-generated posts based on severity using a mix of deterministic filtering, machine learning, and lightweight Agentic layer

**Architecture Overview**

The pipeline will consist of:

1. **Feed Generator** (synthetic posts)
2. **Database-Driven Rule Filter** (fast keyword lookup)
3. **AI Layer 1** (e.g. Llama 3.1 via Ollama for nuance)
4. **Agentic Layer (not sure about its working - help me understand what options do I have here )**
5. **Feedback & Override System** (user/admin moderation)

All logic will flow through a unified moderation controller (GuardianAI) to ensure explainability and tunable thresholds.

**Technical Stack**

| Layer | Tools / Frameworks |
| :---- | :---- |
| Language | Python 3.11+ |
| Backend | FastAPI |
| ML / AI | HuggingFace Transformers |
| Database | **IF NEEDED** - SQLite for dev, PostgreSQL for production-ready handoff |
| Frontend (Optional) | React (basic UI for submission/viewer) |
| LLM Integration | Ollama w/ Llama 3.1 (local) or lightweight hosted LLM  |
| Agentic Integration  | Not decided yet |

**Timeline (13-Day Sprint)**

**Day 1: Feed Generation**

* Write a content generator script using random mixing of good/malicious phrases
* 2 scripts to be written for content generation:
  * First: A gemini based script that has a very string funneled system prompt that will help generate a good mixture of pure to malicious to complete explicit and illegal posts
  * Second: As gemini apis will hit a limit we need to have a python script that can run for hours to generate posts with similar balance 
* Deliverable: 2 different datasets (by 2 different generator scripts) of atleast 10,000 samples in total

**Day 2–3: Rule-Based Filtering (GuardianAI v1)**

* Implement database-backed profanity and severity lookup (SQLite) → SUGGEST ME BETTER STORAGE TYPES
* Deliverable: DatabaseFilter with regex, keywords, scoring

**Day 4: Basic Post Viewer UI**

* Create or mock a web form for post submission
* Render moderation action (Accept, Flag, Block)
* Deliverable: Lightweight UI + API integration

**Day 5: GuardianAI Core Pipeline**

* Combine DatabaseFilter + moderation router
* Return structured output (threat level, action, explanation)
* Deliverable: moderate_content() entrypoint with result class

**Day 6: LLM Escalation Logic**

* Connect to Llama via API (borderline cases only)
* Add thresholds: block if LLM threat ≥ 3, else use feedback
* Deliverable: Hybrid decision logic tested across samples

**Day 7–8: AI Integration Testing**

* Run benchmark test posts (clean, spam, fraud, profane, nuanced)
* Measure average latency + escalation frequency
* Deliverable: Report logs with confidence heatmap

**Day 9: Phrase Suggestions (Optional)**

* LLM should suggest safer alternatives when posts are blocked
* Deliverable: Explanation + suggestion visible in UI response

**Day 10: Dictionary Expansion with Word Embeddings**

* Use FastText/Word2Vec to discover synonyms
* Build candidate list for moderator review
* Deliverable: Updated dictionary + tool for vetting terms

**Day 11: Feedback & Override System**

* Capture user/admin feedback on flagged posts
* Add manual override route in admin view
* Deliverable: Logs with correction indicators

**Day 12: QA Pass + Logging**

* Run full regression test on all flows
* Deliverable: logs, error handling, LLM fallback recovery

**Day 13: Documentation + Demo**

* Technical README, system diagrams, routes, thresholds
* Live walkthrough by dev team
* Deliverable: GitHub repo with clean commit history + presentation

**Success Criteria**

* Content is flagged accurately by rules, ML, and AI layers
* LLM is used for ~10–15% of posts (cost-aware usage)
* System returns action + human-readable explanation per post
* Interns demonstrate understanding of layered moderation strategy

## Setup Instructions

1. **Create Virtual Environment:**
   ```bash
   python -m venv content_moderation_env
   content_moderation_env\Scripts\activate  # Windows
   # source content_moderation_env/bin/activate  # Linux/Mac
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   - Copy `.env.example` to `.env`
   - Add your Gemini API key and other required credentials

4. **Run Data Generation:**
   ```bash
   python scripts/generate_data_gemini.py
   python scripts/generate_data_synthetic.py
   ```

5. **Start Development Server:**
   ```bash
   uvicorn backend.app.main:app --reload
   ```

6. **Frontend Development:**
   ```bash
   cd frontend
   npm install
   npm start
   ``` 