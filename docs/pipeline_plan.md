# _~PLAN:        hierarchical pattern matching & filtering pipeline _




## **Simplified Database Architecture**

### **Database: PostgreSQL (2 Tables Only)**

#### **Table 1: `keywords` (Master Table)**
- `id` (Primary Key)
- `word` (Original keyword from JSON)
- `tier` (1, 2, or 3 - assigned by ML model)
- `severity_score` (0-100 - assigned by ML model)
- `category` (optional: "violence", "hate", "sexual", "drugs", etc.)
- `is_active` (boolean - for easy enable/disable)

#### **Table 2: `patterns` (Generated Patterns)**
- `id` (Primary Key) 
- `keyword_id` (Foreign Key to keywords table)
- `regex_pattern` (The actual regex string)
- `pattern_type` ("exact", "leetspeak", "spaced", "contextual")
- `tier` (Inherited from parent keyword)
- `confidence` (0.0-1.0 - how reliable this pattern is)


## **ML Model Recommendations for Severity Classification**

### **Option 1: Pre-trained Toxicity Models --> We have chosen this**

#### **HuggingFace Toxicity Classifier:**
- **Model**: `unitary/toxic-bert` or `martin-ha/toxic-comment-model`
- **Process**: Run each word through the model, get toxicity probability
- **Implementation**: Use transformers library, batch process your JSON words
- **Output**: Direct severity scores that map to your 3 tiers

## **Complete Technical Flow**

### **Phase 1: System Setup**

#### **Step 1: JSON Processing & ML Classification**

**JSON File Processing:**
- Load your JSON file containing list of words
- Clean and normalize words (remove duplicates, handle case sensitivity)
- Prepare words for ML model input

**ML Classification Process:**
- **If using HuggingFace**: Load pre-trained model → process words in batches → extract severity scores

**Database Population:**
- Insert classified words into `keywords` table
- Each word gets: original text, assigned tier, severity score, category

#### **Step 2: Smart Pattern Generation**

**For each keyword in `keywords` table:**

**Tier 1 Pattern Generation (High Severity):**
- Exact match with word boundaries: `\bword\b`
- Case-insensitive: `(?i)\bword\b`
- Basic substitution: `w[o0]rd`, `w[@a]rd` for common replacements

**Tier 2 Pattern Generation (Medium Severity):**
- Leetspeak: `w[o0][r2][d]` - systematic character replacement
- Spaced: `w\s*o\s*r\s*d` - allows spaces between characters
- Punctuation: `w[.,!?]*o[.,!?]*r[.,!?]*d` - punctuation tolerance

**Tier 3 Pattern Generation (Context-Dependent):**
- Context patterns: `(?=.*context)word` - requires surrounding context
- Phrase patterns: combinations that need multiple words
- Negative lookaheads: `(?!.*educational)word` - avoid false positives

**Pattern Storage:**
- All generated patterns stored in `patterns` table
- Each pattern linked to parent keyword
- Pattern type and confidence score assigned

### **Phase 2: Things that are supposed to happen in real time pipeline**

#### **1. Content Processing Pipeline:**

###### **Step 1: Content Arrives**

- FastAPI receives post content
- Basic preprocessing: lowercase, whitespace cleanup
- Input validation and sanitization

###### **Step 2: Tiered Pattern Matching**

**Tier 1 Processing (Immediate Flags):**
- Query: `SELECT regex_pattern FROM patterns WHERE tier = 1`
- Execute all Tier 1 patterns against content
- If match found: immediate high-severity response
- Processing target: <30ms

**Tier 2 Processing (Pattern-Based):**
- <mark style="background: #FF00FD;">Only runs if Tier 1 doesn't block</mark>
- Query: `SELECT regex_pattern FROM patterns WHERE tier = 2`  
- Execute character substitution and spacing patterns
- Processing target: <100ms

**Tier 3 Processing (Contextual):**
- <mark style="background: #FF00FD;">Only runs if Tier 1 & 2 pass</mark>
- Query: `SELECT regex_pattern FROM patterns WHERE tier = 3`
- Execute context-aware patterns
- Processing target: <200ms

###### **Step 3: Scoring & Decision --> NEED TO REFINE THIS A LOT **

**Simple Scoring System:**
- Tier 1 match: 100 points (auto-block)
- Tier 2 match: 60 points (flag for review)
- Tier 3 match: 30 points (monitor)
- Multiple matches: additive scoring

**Decision Logic:**
- Score 0-29: Pass
- Score 30-59: Flag
- Score 60-89: Moderate  
- Score 90+: Block

### **Phase 3: Ambitious --> Database Optimization for Pilot**

#### **PostgreSQL Setup Strategy:**

**Connection Management:**
- Single database connection pool (5-10 connections for pilot)
- Simple connection reuse without complex pooling

**Pattern Caching Strategy:**
- Load all patterns into memory at startup (since it's pilot scale)
- Python dictionary: `{tier: [compiled_regex_patterns]}`
- No Redis needed for pilot - use in-memory caching

**Query Optimization:**
- Pre-compile all regex patterns at startup
- Use simple SELECT queries, no complex joins needed
- Index on `tier` column for fast filtering

### **Phase 4: FastAPI Integration**

#### **API Endpoint Structure:**

**Main Endpoint: `/moderate`**
- Input: `{"content": "post text"}`
- Processing: Run through 3-tier system
- Output: `{"action": "pass/flag/block", "score": 45, "tier": 2, "matches": ["pattern1"]}`

**Admin Endpoints:**
- `/admin/keywords` - View current keywords and tiers
- `/admin/regenerate` - Rebuild patterns from keywords
- `/health` - System status

#### **Application Startup Flow:**

**FastAPI Startup Process:**
1. Connect to PostgreSQL
2. Load all keywords and patterns from database
3. Compile all regex patterns and store in memory
4. Initialize pattern matching cache
5. Ready to process requests

**Per-Request Flow:**
1. Receive content via API
2. Run Tier 1 patterns (in-memory)
3. If needed, run Tier 2 patterns (in-memory)
4. If needed, run Tier 3 patterns (in-memory)
5. Calculate score and return decision

## **Implementation Timeline with DELIVERABLES**

### **SET 1: ML Classification Setup -------> DONE**
- Process JSON file
- Run ML model (Perspective API or HuggingFace)
- Populate `keywords` table with classified words

### **SET 2: Pattern Generation ------> UPCOMING**
- Build pattern generation script
- Generate patterns for all keywords
- Populate `patterns` table

### **SET 3: FastAPI Integration**
- Create 2-table database schema
- Build FastAPI endpoints
- Implement in-memory pattern caching
- Test end-to-end flow