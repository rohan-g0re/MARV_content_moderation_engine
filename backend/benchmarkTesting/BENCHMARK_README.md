# Day 7-8: AI Integration Testing Framework

This directory contains the comprehensive testing framework for the GuardianAI Content Moderation System, designed to achieve the Day 7-8 deliverables:

- ✅ Run benchmark test posts (clean, spam, fraud, profane, nuanced)
- ✅ Measure average latency + escalation frequency  
- ✅ Deliverable: Report logs with confidence heatmap

## Files Overview

### Core Testing Files
- `benchmark_testing.py` - Comprehensive testing framework with import verification and visualizations
- `content_analyzer.py` - Complete bulk content analysis from Excel/CSV files with all metrics and visualizations
- `run_benchmarks.py` - User-friendly runner script with formatted output
- `sample_content_template.csv` - Sample template for bulk analysis
- Dependencies are included in the main `requirements.txt` file

### Generated Reports
- `benchmark_report_YYYYMMDD_HHMMSS.json` - Detailed JSON report
- `confidence_heatmap_YYYYMMDD_HHMMSS.txt` - Text-based confidence heatmap
- `benchmark_testing.log` - Detailed test logs

## Quick Start

### Option 1: Direct Testing (Recommended)
```bash
cd backend/benchmarkTesting
python benchmark_testing.py
```

### Option 2: Using the Runner Script (User-Friendly)
```bash
cd backend/benchmarkTesting
python run_benchmarks.py
```

### Option 3: Bulk Content Analysis
```bash
cd backend/benchmarkTesting
python content_analyzer.py
```

## Bulk Content Analysis

### Overview
The bulk content analyzer allows you to import Excel/CSV files containing posts and analyze them through the complete moderation pipeline. It provides:

- **Detailed Analysis**: Each post is analyzed through all moderation stages
- **Performance Metrics**: Processing time, latency, and throughput analysis
- **Comprehensive Output**: Excel file with all analysis results
- **Visualizations**: 9 different charts analyzing the results
- **Escalation Tracking**: Identifies content requiring human review

### Input File Format
Your Excel/CSV file should be placed in the same folder as the script and have:
- A column containing the content to analyze (default: "content")
- Optional ID column for post identification
- Supported formats: CSV, XLSX, XLS

### Sample Template
Use `sample_content_template.csv` as a reference for your file structure.

### File Placement
Simply place your file in the `backend/benchmarkTesting/` folder and enter only the filename when prompted.

### Output Files
- **Excel Results**: Detailed analysis of each post with all metrics
- **JSON Report**: Summary statistics and performance data
- **Visualizations**: Comprehensive charts and heatmaps

## Test Categories

### 1. Clean Content
- **Purpose**: Baseline performance measurement
- **Examples**: "Hello, how are you today?", "I love this beautiful weather"
- **Expected**: High acceptance rate, low latency

### 2. Spam Content  
- **Purpose**: Test rule-based filtering
- **Examples**: "CLICK HERE TO WIN $1000 NOW!!!", "FREE VIAGRA"
- **Expected**: High rejection rate, fast processing

### 3. Fraud Content
- **Purpose**: Test financial fraud detection (FinBERT)
- **Examples**: "I'm a Nigerian prince", "Your account has been compromised"
- **Expected**: High rejection rate, medium latency

### 4. Profane Content
- **Purpose**: Test toxicity detection (Detoxify)
- **Examples**: "You're a complete idiot", "Go to hell"
- **Expected**: High rejection rate, medium latency

### 5. Nuanced Content
- **Purpose**: Test complex decision-making
- **Examples**: "The stock market crash could be seen as both disaster and opportunity"
- **Expected**: Variable results, potential escalations

## Metrics Measured

### Performance Metrics
- **Average Latency**: Mean processing time per request
- **Min/Max Latency**: Performance boundaries
- **Standard Deviation**: Consistency measurement
- **Total Requests**: Volume processed

### Accuracy Metrics
- **Category Accuracy**: Correct decisions per content type
- **False Positives**: Clean content incorrectly rejected
- **False Negatives**: Harmful content incorrectly accepted

### Escalation Metrics
- **Escalation Rate**: Percentage of content requiring human review
- **Escalation by Category**: Which content types trigger escalations
- **Escalation Criteria**: Threat level, confidence, stage, action

### Confidence Analysis
- **Confidence Distribution**: How confident the system is in decisions
- **Confidence by Category**: Average confidence per content type
- **Confidence by Stage**: How confidence varies across moderation stages

## Report Structure

### JSON Report (`benchmark_report_YYYYMMDD_HHMMSS.json`)
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "summary": {
    "total_tests": 50,
    "successful_tests": 48,
    "success_rate": 96.0
  },
  "performance": {
    "overall_latency": {
      "avg_latency": 0.245,
      "min_latency": 0.123,
      "max_latency": 0.567
    }
  },
  "escalation": {
    "total_escalations": 12,
    "escalation_rate": 25.0
  },
  "accuracy": {
    "clean": {"accuracy": 100.0, "total_tests": 10},
    "spam": {"accuracy": 95.0, "total_tests": 10}
  },
  "confidence": {
    "by_category": {...},
    "by_stage": {...}
  }
}
```

### Text Heatmap (`confidence_heatmap_YYYYMMDD_HHMMSS.txt`)
```
CONFIDENCE HEATMAP BY CATEGORY AND THREAT LEVEL
============================================================

Category        LOW            MEDIUM         HIGH           
------------------------------------------------------------
CLEAN           0.923          N/A            N/A           
SPAM            0.156          0.234          0.789         
FRAUD           0.123          0.456          0.912         
PROFANE         0.234          0.567          0.845         
NUANCED         0.678          0.345          N/A           
```

## Logging Integration

The framework integrates with the existing logging system:

- **Log File**: `benchmark_testing.log`
- **Format**: JSON-structured logs with timestamps
- **Level**: INFO and above
- **Content**: Test results, performance metrics, errors

Example log entry:
```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "INFO",
  "module": "benchmark",
  "message": "Test completed - Category: spam, Result: false, Time: 0.234s, Confidence: 0.789"
}
```

## Customization

### Adding New Test Categories
```python
self.test_categories["new_category"] = [
    "Test content 1",
    "Test content 2",
    "Test content 3"
]
```

### Modifying Escalation Criteria
```python
def _is_escalated(self, result) -> bool:
    escalation_indicators = [
        result.threat_level in ["medium", "high"],
        result.confidence < 0.7,  # Adjust threshold
        result.stage in ["detoxify", "finbert"],
        result.action in ["FLAG_LOW", "FLAG_MEDIUM", "FLAG_HIGH", "BLOCK"]
    ]
    return any(escalation_indicators)
```

### Adjusting Test Parameters
```python
# In main() function
report = benchmark.run_benchmark_suite(iterations=5)  # More iterations
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the benchmarkTesting directory
   cd backend/benchmarkTesting
   python benchmark_testing.py
   ```

2. **Model Loading Failures**
   - Check internet connection for model downloads
   - Verify sufficient disk space
   - Check Python environment compatibility

3. **Performance Issues**
   - Reduce iterations: `iterations=1`
   - Increase delay between tests: `time.sleep(0.5)`
   - Monitor system resources

### Debug Mode
Enable detailed logging by modifying the logger level:
```python
logger = get_logger("benchmark", "benchmark_testing.log")
# Add debug handler if needed
```

## Expected Results

### Performance Benchmarks
- **Average Latency**: 0.1-0.5 seconds per request
- **Throughput**: 10-50 requests per minute
- **Success Rate**: >95% successful processing

### Accuracy Benchmarks
- **Clean Content**: >95% acceptance rate
- **Spam Content**: >90% rejection rate
- **Fraud Content**: >85% rejection rate
- **Profane Content**: >90% rejection rate
- **Nuanced Content**: Variable (50-80% accuracy)

### Escalation Benchmarks
- **Overall Escalation Rate**: 15-35%
- **High Escalation Categories**: Fraud, Profane, Nuanced
- **Low Escalation Categories**: Clean, Spam

## Integration with Main System

The benchmark framework can be integrated with the main moderation system:

```python
# In main.py or API endpoints
from simple_benchmark_testing import SimpleBenchmarkTestSuite

# Run periodic benchmarks
def run_periodic_benchmarks():
    benchmark = SimpleBenchmarkTestSuite()
    report = benchmark.run_benchmark_suite(iterations=1)
    # Store results in database or send to monitoring system
```

## Next Steps

After running the benchmark tests:

1. **Analyze Results**: Review JSON reports and heatmaps
2. **Identify Bottlenecks**: Look for high latency categories
3. **Optimize Performance**: Adjust thresholds and parameters
4. **Monitor Trends**: Run regular benchmarks to track improvements
5. **Scale Testing**: Increase test volume for production validation

This framework provides comprehensive testing capabilities to validate the AI integration and ensure the content moderation system meets performance and accuracy requirements. 