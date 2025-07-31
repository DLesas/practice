# Warm-up Exercise: Quick Data Exploration
**Difficulty**: Intermediate  
**Time**: 10-15 minutes  
**Skills**: Data exploration, basic pandas operations

## Purpose
This warm-up exercise helps you get comfortable with the interview environment and demonstrates fundamental data analysis skills. Use this to build rapport and show your systematic approach to data exploration.

## The Scenario
You've been given a sales dataset and need to quickly understand its structure and identify potential data quality issues. This is typical of the first task in any data analysis project.

## Quick Setup
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the sample dataset
df = pd.read_csv('datasets/sample_sales.csv')
```

## Tasks (10-15 minutes total)

### 1. Initial Data Understanding (3-4 minutes)
```python
# Your approach here - think aloud about what you're checking:
# - Dataset shape and structure
# - Column types and names
# - Memory usage
# - Sample of the data
```

**What to verbalize:**
- "Let me start by understanding the basic structure of this dataset..."
- "I'll check the data types to see if anything needs conversion..."
- "Let me look at a sample to understand what each column represents..."

### 2. Data Quality Assessment (4-5 minutes)
```python
# Quick quality checks:
# - Missing values
# - Duplicates
# - Basic statistics
# - Outliers
```

**What to verbalize:**
- "Now I'll check for common data quality issues..."
- "Missing values can significantly impact analysis, so let me check for those..."
- "Let me look at the distribution of key variables..."

### 3. Quick Insights (3-4 minutes)
```python
# Generate a few quick insights:
# - Top categories/products
# - Time trends (if applicable)
# - Simple aggregations
```

**What to verbalize:**
- "Based on this initial exploration, I can see a few interesting patterns..."
- "I notice [specific observation] - that might be worth investigating further..."
- "For the main analysis, I'd want to focus on..."

### 4. Next Steps Discussion (2-3 minutes)
**Questions to ask:**
- "Based on this initial exploration, what aspects would you like me to focus on?"
- "Are there any specific data quality concerns you've encountered before?"
- "What business questions are most important to answer with this data?"

## Key Things to Demonstrate

### Technical Skills:
- Systematic approach to data exploration
- Knowledge of pandas exploration methods
- Quick identification of data quality issues
- Basic statistical understanding

### Communication Skills:
- Clear explanation of your process
- Asking relevant follow-up questions
- Connecting data observations to business implications
- Setting up for deeper analysis

## Sample Exploration Framework

```python
def explore_dataset(df, name="dataset"):
    """Systematic dataset exploration"""
    
    print(f"=== {name.upper()} EXPLORATION ===\n")
    
    # 1. Basic info
    print("1. BASIC INFORMATION")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nColumn info:")
    print(df.info())
    
    # 2. Missing values
    print("\n2. MISSING VALUES")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing': missing[missing > 0],
            'Percentage': missing_pct[missing > 0]
        })
        print(missing_df)
    else:
        print("No missing values found")
    
    # 3. Basic statistics
    print("\n3. BASIC STATISTICS")
    print(df.describe())
    
    # 4. Sample data
    print("\n4. SAMPLE DATA")
    print(df.head())
    
    return df

# Usage in interview:
# explored_df = explore_dataset(df, "Sales Data")
```

## Success Indicators
- **Structured approach**: Following a logical sequence
- **Clear communication**: Explaining what you're looking for and why
- **Business awareness**: Connecting data patterns to potential business implications
- **Question asking**: Seeking clarification and guidance
- **Time management**: Completing exploration efficiently

This warm-up sets the stage for more complex challenges while establishing your analytical approach and communication style.