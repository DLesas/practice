# Faculty AI Interview Practice Session Guide

## Pre-Practice Setup

### 1. Generate All Datasets
```bash
# Generate sample sales data (warm-up)
cd datasets
python generate_sample_sales.py

# Generate e-commerce data (main exercise)
python generate_ecommerce_data.py

# Generate sensor data (time series exercise)
python generate_sensor_data.py
```

### 2. Set Up Environment
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
```

## Practice Session Structure

### Session 1: Warm-up + Main Challenge (90 minutes)
**Simulates actual interview timing**

#### Warm-up (15 minutes)
- **Exercise**: `00_warmup_data_exploration.md`
- **Dataset**: `datasets/sample_sales.csv`
- **Focus**: Communication, systematic approach, basic pandas

**Practice script:**
```python
# Time yourself and practice thinking aloud
import time
start_time = time.time()

# Load and explore
df = pd.read_csv('datasets/sample_sales.csv')

# Your exploration here...
# Remember to verbalize your process!

print(f"Exploration took: {time.time() - start_time:.1f} seconds")
```

#### Main Challenge (60 minutes)
- **Exercise**: `01_data_pipeline_challenge.md`
- **Dataset**: E-commerce data (transactions, customers, products)
- **Focus**: Complex data manipulation, feature engineering

#### Wrap-up Discussion (15 minutes)
- Review your approach
- Identify areas for improvement
- Plan optimizations

### Session 2: Advanced Challenges (60 minutes)

#### Time Series Analysis (30 minutes)
- **Exercise**: `02_time_series_anomaly_detection.md`
- **Dataset**: `datasets/sensor_readings.csv`, `datasets/sensor_metadata.csv`

#### Performance Optimization (30 minutes)
- **Exercise**: `03_performance_optimization.md`
- **Focus**: Take your existing code and optimize it

## Practice Tips

### Before Starting Each Exercise:

1. **Read the full problem** (2-3 minutes)
2. **Ask clarifying questions** out loud (even to yourself)
3. **Plan your approach** before coding
4. **Set up your workspace** (imports, data loading)

### While Coding:

1. **Think aloud constantly**
   - "I'm checking for missing values because..."
   - "I notice this pattern, which suggests..."
   - "Let me validate this result by..."

2. **Manage your time**
   - Check the clock every 15 minutes
   - If stuck, move to a simpler approach
   - Leave time for validation and discussion

3. **Handle issues gracefully**
   - "This isn't working as expected, let me try..."
   - "I'm getting an error, let me debug this..."
   - "I need to look up the syntax for..."

### Self-Assessment Criteria:

Rate yourself (1-5) on:
- **Technical execution**: Did you solve the problem correctly?
- **Code quality**: Clean, readable, efficient code?
- **Communication**: Clear explanation of thought process?
- **Problem-solving**: Good approach to debugging and optimization?
- **Time management**: Completed in reasonable time?

## Mock Interview Simulation

### Solo Practice Version:
```python
def mock_interview_timer():
    """Use this to simulate interview pressure"""
    import time
    
    phases = [
        ("Problem Understanding", 10),
        ("Solution Planning", 10), 
        ("Core Implementation", 50),
        ("Testing & Validation", 15),
        ("Discussion", 5)
    ]
    
    start_time = time.time()
    
    for phase, duration in phases:
        print(f"\n=== {phase.upper()} ({duration} min) ===")
        print(f"Started at: {time.strftime('%H:%M:%S')}")
        
        # Your work here
        input(f"Press Enter when {phase} is complete...")
        
        elapsed = (time.time() - start_time) / 60
        print(f"Elapsed: {elapsed:.1f} min")
        
        if elapsed > sum(p[1] for p in phases[:phases.index((phase, duration))+1]):
            print("⚠️  Running behind schedule!")
        
    print(f"\nTotal time: {elapsed:.1f} minutes")

# Run this to practice with timing pressure
# mock_interview_timer()
```

### With a Practice Partner:
1. **Observer role**: Ask questions, provide feedback
2. **Interviewee role**: Solve problems while explaining
3. **Switch roles** for different exercises

## Common Mistakes to Avoid

### Technical Mistakes:
- Not checking data types after loading
- Forgetting to handle missing values before operations
- Not validating results (e.g., negative values where impossible)
- Using inefficient operations (loops instead of vectorization)
- Not considering memory usage with large datasets

### Communication Mistakes:
- Long silences without explanation
- Not asking clarifying questions
- Getting stuck on one approach without exploring alternatives
- Not explaining the business rationale for technical decisions
- Rushing through without validation

## Advanced Practice Ideas

### Week 1: Master the Basics
- Practice exercises 0 and 1 daily
- Focus on smooth execution and communication
- Time yourself consistently

### Week 2: Advanced Techniques
- Add complexity to existing exercises
- Practice exercises 2 and 3
- Work on performance optimization

### Week 3: Interview Simulation
- Full 90-minute sessions with mock timing
- Practice with different datasets
- Record yourself and review communication

### Week 4: Polish and Confidence
- Focus on weak areas identified
- Practice common pandas operations until automatic
- Review interview tips and strategies

## Quick Reference Cheat Sheet

### Essential Pandas Operations:
```python
# Data exploration
df.info(), df.describe(), df.head()
df.shape, df.dtypes, df.memory_usage()
df.isnull().sum(), df.duplicated().sum()

# Data cleaning
df.dropna(), df.fillna()
df.drop_duplicates()
pd.to_datetime(), pd.to_numeric()

# Data manipulation
df.groupby().agg()
df.merge(), pd.concat()
df.pivot_table(), df.melt()
df.sort_values(), df.reset_index()

# Feature engineering
pd.cut(), pd.qcut()
df.apply(), df.transform()
df.rolling(), df.resample()
```

### Performance Tips:
```python
# Memory optimization
df.astype('category')
pd.read_csv(dtype={'col': 'category'})

# Speed optimization
df.query() instead of boolean indexing
df.eval() for complex expressions
Vectorized operations instead of loops
```

Good luck with your practice! Remember: the interview is as much about demonstrating your thought process and communication skills as it is about technical ability. 🚀