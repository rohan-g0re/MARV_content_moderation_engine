# Content Moderation Engine - Setup and Run Guide

## 🚀 Complete Step-by-Step Instructions

Follow these steps to set up and run both data generation scripts yourself.

---

## 📋 Prerequisites

- Python 3.11+ installed on your system
- Gemini API key (get from Google AI Studio)
- Windows PowerShell (already available)

---

## 🔧 Step 1: Create Environment File

1. **Create .env file using the provided script:**
   ```powershell
   python create_env.py
   ```

2. **Edit the .env file and add your Gemini API key:**
   ```bash
   # Content Moderation Engine - Environment Variables
   
   # API Keys - REQUIRED
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   
   # Database Configuration
   DATABASE_URL=sqlite:///./content_moderation.db
   POSTGRES_URL=postgresql://user:password@localhost:5432/content_moderation
   
   # Application Settings
   DEBUG=True
   LOG_LEVEL=INFO
   SECRET_KEY=your_secret_key_here
   
   # API Configuration
   API_HOST=0.0.0.0
   API_PORT=8000
   
   # AI Model Settings
   DEFAULT_MODEL=gemini-1.5-flash
   MODERATION_THRESHOLD=0.7
   LLM_ESCALATION_THRESHOLD=0.5
   
   # Rate Limiting
   GEMINI_RATE_LIMIT_DELAY=1.0
   MAX_REQUESTS_PER_MINUTE=60
   
   # Frontend Configuration
   FRONTEND_URL=http://localhost:3000
   CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
   
   # Data Generation Settings
   DATASET_SIZE_GEMINI=5000
   DATASET_SIZE_SYNTHETIC=5000
   OUTPUT_FORMAT=csv
   
   # Logging Configuration
   LOG_FILE=logs/content_moderation.log
   LOG_MAX_SIZE=10485760
   LOG_BACKUP_COUNT=5
   ```

3. **IMPORTANT:** Replace `your_actual_gemini_api_key_here` with your real Gemini API key

---

## 🔧 Step 2: Set Up Virtual Environment

1. **Open PowerShell in the project directory:**
   ```powershell
   cd D:\STUFF\MARV_Internship\MAIN\content_moderation_engine
   ```

2. **Create virtual environment:**
   ```powershell
   python -m venv content_moderation_env
   ```

3. **Activate virtual environment:**
   ```powershell
   content_moderation_env\Scripts\activate
   ```
   
   You should see `(content_moderation_env)` at the beginning of your prompt.

4. **Upgrade pip:**
   ```powershell
   pip install --upgrade pip
   ```

5. **Install essential dependencies:**
   ```powershell
   pip install google-generativeai python-dotenv pandas
   ```

---

## 🎯 Step 3: Run Synthetic Data Generation Script

**This script generates data without external APIs:**

1. **Ensure virtual environment is activated:**
   ```powershell
   content_moderation_env\Scripts\activate
   ```

2. **Run the synthetic script:**
   ```powershell
   python scripts/generate_data_synthetic.py
   ```

3. **Expected output:**
   - Script will generate 5,000 samples
   - Progress updates will show in console
   - Output file: `data/raw/synthetic_dataset.csv`
   - Log file: `logs/synthetic_generation.log`

4. **Verify the results:**
   ```powershell
   python -c "import pandas as pd; df = pd.read_csv('data/raw/synthetic_dataset.csv'); print(f'Generated {len(df)} samples'); print(df['label'].value_counts())"
   ```

---

## 🎯 Step 4: Run Gemini Data Generation Script

**This script uses Gemini API to generate diverse content:**

1. **Ensure your .env file has the correct Gemini API key**

2. **Run the Gemini script:**
   ```powershell
   python scripts/generate_data_gemini.py
   ```

3. **Expected behavior:**
   - Script will make API calls with 1-second delays
   - Progress updates will show batch generation
   - Output file: `data/raw/gemini_dataset.csv`
   - Log file: `logs/gemini_generation.log`

4. **Monitor progress:**
   - Watch console output for progress updates
   - Check log file for detailed information:
     ```powershell
     Get-Content logs/gemini_generation.log -Tail 10
     ```

5. **Verify the results:**
   ```powershell
   python -c "import pandas as pd; df = pd.read_csv('data/raw/gemini_dataset.csv'); print(f'Generated {len(df)} samples'); print(df['label'].value_counts())"
   ```

---

## 🔍 Step 5: Verify Combined Results

**Check both datasets:**

```powershell
python -c "
import pandas as pd
import os

# Check if both files exist
synthetic_exists = os.path.exists('data/raw/synthetic_dataset.csv')
gemini_exists = os.path.exists('data/raw/gemini_dataset.csv')

print('=== DATASET STATUS ===')
print(f'Synthetic Dataset: {\"✅ EXISTS\" if synthetic_exists else \"❌ MISSING\"}')
print(f'Gemini Dataset: {\"✅ EXISTS\" if gemini_exists else \"❌ MISSING\"}')

if synthetic_exists:
    df_synthetic = pd.read_csv('data/raw/synthetic_dataset.csv')
    print(f'\nSynthetic Dataset: {len(df_synthetic)} samples')
    print(df_synthetic['label'].value_counts())

if gemini_exists:
    df_gemini = pd.read_csv('data/raw/gemini_dataset.csv')
    print(f'\nGemini Dataset: {len(df_gemini)} samples')
    print(df_gemini['label'].value_counts())

if synthetic_exists and gemini_exists:
    total_samples = len(df_synthetic) + len(df_gemini)
    print(f'\n🎉 TOTAL SAMPLES: {total_samples}')
    print('✅ Day 1 Objective ACHIEVED!' if total_samples >= 10000 else '⚠️ Need more samples')
"
```

---

## 🛠️ Troubleshooting

### Common Issues and Solutions:

#### 1. **Virtual Environment Not Working**
```powershell
# Remove and recreate
Remove-Item -Recurse -Force content_moderation_env
python -m venv content_moderation_env
content_moderation_env\Scripts\activate
```

#### 2. **Gemini API Key Issues**
- Check your .env file has the correct key
- Verify API key is valid in Google AI Studio
- Ensure no extra spaces around the key

#### 3. **Permission Errors**
```powershell
# Run PowerShell as Administrator if needed
```

#### 4. **Package Installation Issues**
```powershell
# Update pip first
pip install --upgrade pip
# Install packages one by one
pip install google-generativeai
pip install python-dotenv
pip install pandas
```

#### 5. **Script Errors**
- Check log files in `logs/` directory
- Ensure virtual environment is activated
- Verify all dependencies are installed

---

## 📊 Expected Final Results

After running both scripts successfully:

- **Synthetic Dataset:** 5,000 samples in `data/raw/synthetic_dataset.csv`
- **Gemini Dataset:** 5,000 samples in `data/raw/gemini_dataset.csv`
- **Total:** 10,000+ samples with balanced label distribution
- **Labels:** SAFE, QUESTIONABLE, INAPPROPRIATE, HARMFUL, ILLEGAL

---

## 🎯 Success Criteria Checklist

- [ ] Virtual environment created and activated
- [ ] .env file created with valid Gemini API key
- [ ] Synthetic script runs without errors
- [ ] Gemini script runs without errors
- [ ] Both CSV files generated successfully
- [ ] Total samples >= 10,000
- [ ] Label distribution is balanced across all 5 categories

---

## 🚀 Next Steps

Once both scripts complete successfully:
1. ✅ Day 1 deliverables are complete
2. 📝 Update progress documentation
3. 🔄 Ready to begin Day 2 (Rule-based filtering)

---

**Need Help?** Check the log files in `logs/` directory for detailed error messages and troubleshooting information. 