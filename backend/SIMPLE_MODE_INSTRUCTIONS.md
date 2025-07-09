# 🎚️ Simple Mode Instructions for GuardianAI

## For Non-Technical Users

### What This Does
GuardianAI has **TWO modes** you can choose from by changing **ONE simple setting**:

- **PRODUCTION MODE** = Full protection (recommended for live use)
- **TESTING MODE** = Basic protection only (for testing/debugging)

---

## 🚀 How to Change the Mode

### Step 1: Open the File
Navigate to and open this file:
```
backend/app/core/moderation_safe.py
```

### Step 2: Find the Setting
Look for this section at the **very top** of the file:
```python
# =============================================================================
# 🎚️ SIMPLE TOGGLE FOR NON-TECHNICAL USERS
# Change this ONE setting to control the entire moderation pipeline:
# 
# PRODUCTION_MODE = True   → Full pipeline (all 4 stages: rules + LGBM + AI)
# PRODUCTION_MODE = False  → Testing mode (only basic rule-based filtering)
# =============================================================================
PRODUCTION_MODE = True
```

### Step 3: Change the Setting
**For Full Protection (Recommended):**
```python
PRODUCTION_MODE = True
```

**For Testing Only:**
```python
PRODUCTION_MODE = False
```

### Step 4: Save and Restart
1. Save the file
2. Restart the server by stopping it and running `python main.py` again

---

## 🔍 What Each Mode Does

### PRODUCTION_MODE = True (Recommended)
✅ **Full Protection Pipeline**
- Stage 1: Rule-based filtering (keywords & patterns)
- Stage 2: LGBM machine learning model (financial fraud detection)  
- Stage 3: Detoxify AI (toxicity detection)
- Stage 4: FinBERT AI (advanced financial fraud detection)

**Use this for:** Live websites, production environments, maximum protection

### PRODUCTION_MODE = False (Testing)
⚡ **Basic Protection Only**
- Stage 1: Rule-based filtering (keywords & patterns) ONLY
- Stages 2-4: Disabled

**Use this for:** Testing, debugging, when AI models aren't available

---

## 🌐 Check Current Mode

Visit this URL to see what mode you're currently in:
```
http://localhost:8000/simple-mode
```

This will show you:
- Current mode (PRODUCTION or TESTING)
- Which stages are active
- Step-by-step instructions

---

## ❓ Need Help?

If you need to change the mode but can't find the file:

1. Make sure you're in the correct project folder
2. The file path is: `backend/app/core/moderation_safe.py`
3. Look for the line that says `PRODUCTION_MODE = True` or `PRODUCTION_MODE = False`
4. Change it to what you want
5. Save and restart the server

**That's it!** One line change controls the entire moderation system. 