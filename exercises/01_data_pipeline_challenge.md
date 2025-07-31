# Exercise 1: E-commerce Data Pipeline Challenge

**Difficulty**: Advanced  
**Time**: 45-60 minutes  
**Skills**: Complex merging, aggregations, feature engineering

## Business Context

You're working on an ML model to predict customer lifetime value for an e-commerce platform. You need to create features from transactional data spanning multiple tables.

## The Challenge

Given three datasets:

1. **transactions**: Customer purchases over time
2. **customers**: Customer demographic information
3. **products**: Product catalog with categories

Create a customer-level feature matrix for ML training.

## Data Description

### Transactions Table

```python
# transactions.csv - 100k+ rows
columns = ['transaction_id', 'customer_id', 'product_id', 'quantity',
           'price', 'discount', 'timestamp', 'payment_method']
```

### Customers Table

```python
# customers.csv
columns = ['customer_id', 'registration_date', 'age', 'country',
           'email_domain', 'acquisition_channel']
```

### Products Table

```python
# products.csv
columns = ['product_id', 'category', 'subcategory', 'brand',
           'base_price', 'margin']
```

## Requirements

### Part 1: Data Quality Assessment (10 min)

1. Identify and handle missing values across all tables
2. Detect and resolve duplicate transactions
3. Find outliers in transaction amounts
4. Validate data consistency (e.g., negative quantities)

### Part 2: Feature Engineering (25 min)

Create these customer-level features:

1. **Behavioral features**:

   - Total transactions, revenue, items purchased
   - Average order value, frequency
   - Days since last purchase
   - Purchase velocity (transactions per week)

2. **Product affinity features**:

   - Number of unique categories purchased
   - Favorite category (by spend)
   - Brand loyalty score

3. **Time-based features**:
   - Seasonality patterns (quarterly spend)
   - Recency, Frequency, Monetary (RFM) scores
   - Purchase acceleration/deceleration

### Part 3: Advanced Aggregations (15 min)

1. Calculate rolling 30-day metrics
2. Create customer segments based on purchase patterns
3. Compute cross-category purchase probabilities

### Part 4: ML Pipeline Extension (Optional - 15 min)

If time permits, extend your feature matrix for machine learning:

1. **Target variable creation**:

   - Define a business metric to predict (e.g., customer lifetime value bucket, churn risk)
   - Handle class imbalance if present

2. **ML-ready features**:

   - Create interaction features (e.g., AOV × frequency)
   - Ratio features (profit margin, items per transaction)
   - Behavioral indicators (high-value customer flags)

3. **Quick model validation**:
   - Train/test split with proper stratification
   - Fit a simple baseline model (logistic regression or random forest)
   - Evaluate with appropriate metrics

## Key Discussion Points

- How would you handle memory constraints with larger datasets?
- What data quality issues concern you most?
- How would you make this pipeline production-ready?
- What additional features might improve the model?
- **ML Extension**: How would you validate this model in production?
- **ML Extension**: What metrics would you monitor for model drift?

## Success Criteria

- Clean, readable code with proper error handling
- Efficient pandas operations (vectorized when possible)
- Thoughtful feature engineering with business rationale
- Discussion of scalability considerations
- **ML Extension**: Proper train/test methodology without data leakage
