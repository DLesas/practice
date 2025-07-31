# Exercise 5: A/B Testing for ML Models

**Difficulty**: Advanced  
**Time**: 30-45 minutes  
**Skills**: Experimental design, statistical testing, business metrics

## Business Context

Your new churn prediction model is ready for production testing. You need to design and analyze an A/B test to measure its impact on customer retention and business revenue.

## The Challenge

You have data from a 2-week A/B test where:

- **Control group**: Existing business-as-usual approach to customer retention
- **Treatment group**: Customers identified as high-churn-risk receive targeted interventions

Analyze the test results and provide business recommendations.

## A/B Test Setup

```python
# Test design
- Duration: 2 weeks
- Sample size: 10,000 customers (5,000 per group)
- Primary metric: Customer retention rate
- Secondary metrics: Revenue per customer, intervention cost
- Randomization: Stratified by customer segment
```

## Data Description

### Experiment Results Table

```python
# ab_test_results.csv
columns = ['customer_id', 'group', 'segment', 'predicted_churn_prob',
           'intervention_sent', 'intervention_cost', 'retained',
           'revenue_2weeks', 'revenue_4weeks']
```

### Intervention Details Table

```python
# interventions.csv
columns = ['intervention_type', 'cost_per_customer', 'expected_lift',
           'target_churn_threshold']
```

## Requirements

### Part 1: Experimental Validation (10 min)

1. **Randomization check**:

   - Verify balanced groups across customer segments
   - Check for any systematic bias in group assignment
   - Validate sample size and power calculations

2. **Data quality assessment**:
   - Check for missing data patterns
   - Identify any data collection issues
   - Validate metric definitions

### Part 2: Statistical Analysis (15 min)

1. **Primary metric analysis**:

   - Calculate retention rates by group
   - Perform appropriate statistical test (Chi-square or Fisher's exact test)
   - Calculate confidence intervals
   - Determine statistical significance

2. **Secondary metrics analysis**:

   - Revenue impact analysis
   - Cost-benefit analysis of interventions
   - Segment-level performance analysis

3. **Effect size and practical significance**:
   - Calculate lift in retention rate
   - Estimate business impact (revenue, cost savings)
   - Statistical vs practical significance

### Part 3: Advanced Analysis (15 min)

1. **Subgroup analysis**:

   - Performance by customer segment
   - Intervention effectiveness by churn probability
   - Time-based analysis (week 1 vs week 2)

2. **Model performance evaluation**:

   - Precision/recall of churn predictions
   - ROI of interventions by prediction confidence
   - False positive/negative cost analysis

3. **Business recommendations**:
   - Should we launch the model to all customers?
   - Optimal intervention strategy
   - Recommendations for model improvements

## Key Statistical Concepts

- **Hypothesis testing**: Null/alternative hypotheses, p-values, Type I/II errors
- **Effect size**: Cohen's d, relative risk, odds ratio
- **Multiple testing**: Bonferroni correction for multiple metrics
- **Power analysis**: Sample size calculations, statistical power
- **Confidence intervals**: Interpretation and business communication

## Sample Analysis Framework

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_ab_test(results_df):
    """
    Comprehensive A/B test analysis
    """

    # 1. Randomization check
    print("=== RANDOMIZATION CHECK ===")
    # Your analysis here

    # 2. Primary metric analysis
    print("=== PRIMARY METRIC ANALYSIS ===")
    # Your analysis here

    # 3. Statistical tests
    print("=== STATISTICAL TESTS ===")
    # Your analysis here

    # 4. Business impact
    print("=== BUSINESS IMPACT ===")
    # Your analysis here

    return results_summary

def calculate_confidence_interval(successes, trials, confidence=0.95):
    """Calculate confidence interval for proportion"""
    # Your implementation here
    pass

def power_analysis(p1, p2, alpha=0.05, power=0.8):
    """Calculate required sample size for given effect size"""
    # Your implementation here
    pass
```

## Discussion Points

- How do you balance statistical significance with business deadlines?
- What would you do if results are statistically significant but effect size is small?
- How would you handle multiple testing when analyzing many subgroups?
- What are the ethical considerations of A/B testing customer treatments?
- How would you design a follow-up experiment based on these results?

## Success Criteria

- Proper experimental analysis methodology
- Clear communication of statistical vs practical significance
- Business-focused recommendations with uncertainty quantification
- Understanding of A/B testing limitations and assumptions
- Thoughtful discussion of next steps and improvements

## Real-world Considerations

- **Seasonality**: How might time of year affect results?
- **Network effects**: Could treatment customers influence control customers?
- **Long-term impact**: How to measure effects beyond the test period?
- **Ethical concerns**: Fairness of withholding potentially beneficial treatments
- **Implementation challenges**: Technical requirements for deployment
