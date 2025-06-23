**Project Brief: Content Moderation Engine**

**Client:** UnBound X  
**Team:** Intern Development Cohort  
**Delivery Format:** Daily commits \+ final walkthrough \+ documentation

**Objective**

Design and build a cost-efficient, scalable content moderation system using multiple filtering layers. The system should triage user-generated posts based on severity using a mix of deterministic filtering, machine learning, and lightweight Agentic layer

**Architecture Overview**

The pipeline will consist of:

1. **Feed Generator** (synthetic posts)

2. **Database-Driven Rule Filter** (fast keyword lookup)

3. **AI Layer 1** (e.g. Llama 3.1 via Ollama for nuance)

4. **Agentic Layer (not sure about its working \- help me understand what options do I have here )**

5. **Feedback & Override System** (user/admin moderation)

All logic will flow through a unified moderation controller (GuardianAI) to ensure explainability and tunable thresholds.

**Technical Stack**

| Layer | Tools / Frameworks |
| :---- | :---- |
| Language | Python 3.11+ |
| Backend | FastAPI |
| ML / AI | HuggingFace Transformers |
| Database | **IF NEEDED** \- SQLite for dev, PostgreSQL for production-ready handoff |
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

* Plan for today: 

- DOCUMENTS TO BE CREATED \- keep them TO THE POINT such that CLAUDE can understand it better \- you decide the file format:  
  - We have to make a daily progress document  
  - A plan document (which will be constantly updated for the deliverables achieved, the roadmap, and the technical routing of thec complete repo → make it such that I can add it to a new chat in cursor and it can understand COMPLETELY what are we working on and whatsthe progress)  
  - an error document (which will be constantly updated when we make will be trying to fix an error → make it such that I can add it to a new chat in cursor and it can understand what errors we ran in the past and how did we solve it)

\*\*\*

- write 2 scripts, run both of them to generate data  
- both the datasets should have a same model which will be “Post (can be of varying length and with mixed maturity), label (good, on edge , or whatever \- the band of labels should be 5 or 7-band thing which could help in easy classification and segregation of posts goodness/badness)”  \- so basically only 2 columns → also the gemini api stalls a lot so every api call must have some delay (1 sec break) in between so that it does not stall   
- As in future we will be creating a web app using react and fastapi, develop a detailed file structure for the complete repo → it should be highly modular and should accommodate all the things in a sleek way such that any colleague and ai coding agents can easily navigate the codebase (even the further elements in the 13 day plan)  
- Create a comprehensive gitignore, create a readme which has THIS DOCUMENT ITSELF NOTHING ELSE, give steps of setup  
- create virtual env to work in  
- When you are developing the code make sure that you exhaust most of the cursor calls --> internally deivide the tasks into multiple subtasks and assign each subtask a set of objective delieverables --> only go ahead when YOU HAVE SIGNIFICANTLY TESTED and VERIFIED THE DELIEVERABLES YOURSELF 


**Day 2–3: Rule-Based Filtering (GuardianAI v1)**

* Implement database-backed profanity and severity lookup (SQLite) → SUGGEST ME BETTER STORAGE TYPES

* Deliverable: DatabaseFilter with regex, keywords, scoring

**Day 4: Basic Post Viewer UI**

* Create or mock a web form for post submission

* Render moderation action (Accept, Flag, Block)

* Deliverable: Lightweight UI \+ API integration

**Day 5: GuardianAI Core Pipeline**

* Combine DatabaseFilter \+ moderation router

* Return structured output (threat level, action, explanation)

* Deliverable: moderate\_content() entrypoint with result class

**Day 6: LLM Escalation Logic**

* Connect to Llama via API (borderline cases only)

* Add thresholds: block if LLM threat ≥ 3, else use feedback

* Deliverable: Hybrid decision logic tested across samples

**Day 7–8: AI Integration Testing**

* Run benchmark test posts (clean, spam, fraud, profane, nuanced)

* Measure average latency \+ escalation frequency

* Deliverable: Report logs with confidence heatmap

**Day 9: Phrase Suggestions (Optional)**

* LLM should suggest safer alternatives when posts are blocked

* Deliverable: Explanation \+ suggestion visible in UI response

**Day 10: Dictionary Expansion with Word Embeddings**

* Use FastText/Word2Vec to discover synonyms

* Build candidate list for moderator review

* Deliverable: Updated dictionary \+ tool for vetting terms

**Day 11: Feedback & Override System**

* Capture user/admin feedback on flagged posts

* Add manual override route in admin view

* Deliverable: Logs with correction indicators

**Day 12: QA Pass \+ Logging**

* Run full regression test on all flows

* Deliverable: logs, error handling, LLM fallback recovery

**Day 13: Documentation \+ Demo**

* Technical README, system diagrams, routes, thresholds

* Live walkthrough by dev team

* Deliverable: GitHub repo with clean commit history \+ presentation

**Success Criteria**

* Content is flagged accurately by rules, ML, and AI layers

* LLM is used for \~10–15% of posts (cost-aware usage)

* System returns action \+ human-readable explanation per post

* Interns demonstrate understanding of layered moderation strategy

