"""
Solution: Warm-up Data Exploration Exercise
Demonstrates systematic approach to data exploration and communication
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_dataset(df, name="dataset"):
    """
    Systematic dataset exploration framework
    This demonstrates the methodical approach expected in interviews
    """
    
    print(f"=== {name.upper()} EXPLORATION ===\n")
    
    # 1. Basic Information
    print("1. BASIC INFORMATION")
    print("-" * 30)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nColumn information:")
    print(df.info())
    
    # 2. Missing Values Analysis
    print("\n2. MISSING VALUES ANALYSIS")
    print("-" * 30)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Count': missing[missing > 0],
            'Percentage': missing_pct[missing > 0]
        }).sort_values('Count', ascending=False)
        print(missing_df)
        
        # Insight
        print(f"\nInsight: {missing_df.shape[0]} columns have missing values")
        print(f"Most problematic: {missing_df.index[0]} ({missing_df.iloc[0]['Percentage']:.1f}% missing)")
    else:
        print("✓ No missing values found")
    
    # 3. Duplicate Analysis
    print("\n3. DUPLICATE ANALYSIS")
    print("-" * 30)
    duplicates = df.duplicated().sum()
    print(f"Exact duplicates: {duplicates:,}")
    
    # Check for business logic duplicates (if applicable)
    if 'customer_id' in df.columns and 'transaction_date' in df.columns:
        business_dups = df.duplicated(['customer_id', 'transaction_date']).sum()
        print(f"Business duplicates (same customer, same date): {business_dups:,}")
    
    # 4. Data Types and Unique Values
    print("\n4. DATA TYPES & CARDINALITY")
    print("-" * 30)
    dtype_summary = pd.DataFrame({
        'dtype': df.dtypes,
        'unique_values': df.nunique(),
        'unique_ratio': (df.nunique() / len(df)).round(3)
    })
    print(dtype_summary)
    
    # Identify potential categorical columns
    potential_categories = dtype_summary[
        (dtype_summary['dtype'] == 'object') & 
        (dtype_summary['unique_ratio'] < 0.5)
    ].index.tolist()
    
    if potential_categories:
        print(f"\nPotential categorical columns: {potential_categories}")
    
    # 5. Numerical Summary Statistics
    print("\n5. NUMERICAL SUMMARY")
    print("-" * 30)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
        
        # Identify potential outliers
        print(f"\nOutlier Detection (using IQR method):")
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
            if outliers > 0:
                print(f"  {col}: {outliers:,} potential outliers ({outliers/len(df)*100:.1f}%)")
    else:
        print("No numerical columns found")
    
    # 6. Categorical Analysis
    print("\n6. CATEGORICAL ANALYSIS")
    print("-" * 30)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols[:5]:  # Limit to first 5 to avoid clutter
        print(f"\n{col}:")
        value_counts = df[col].value_counts().head()
        print(value_counts)
        
        if len(df[col].unique()) > 10:
            print(f"  ... and {len(df[col].unique()) - 5} more unique values")
    
    # 7. Sample Data
    print("\n7. SAMPLE DATA")
    print("-" * 30)
    print("First 5 rows:")
    print(df.head())
    
    print(f"\nRandom 3 rows:")
    if len(df) >= 3:
        print(df.sample(3))
    
    return df

def identify_data_quality_issues(df):
    """
    Identify specific data quality issues that need attention
    """
    
    print("\n" + "="*50)
    print("DATA QUALITY ISSUES IDENTIFIED")
    print("="*50)
    
    issues = []
    
    # 1. Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        critical_missing = missing[missing > len(df) * 0.1]  # >10% missing
        if len(critical_missing) > 0:
            issues.append(f"HIGH: {len(critical_missing)} columns have >10% missing values")
            for col in critical_missing.index:
                issues.append(f"  - {col}: {missing[col]:,} missing ({missing[col]/len(df)*100:.1f}%)")
    
    # 2. Potential data entry errors
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any() and 'id' not in col.lower():
            negative_count = (df[col] < 0).sum()
            issues.append(f"MEDIUM: {col} has {negative_count} negative values (might be data entry errors)")
    
    # 3. Inconsistent data
    if 'total_amount' in df.columns and 'unit_price' in df.columns and 'quantity' in df.columns:
        expected_total = df['unit_price'] * df['quantity']
        inconsistent = abs(df['total_amount'] - expected_total) > 0.01
        if inconsistent.any():
            issues.append(f"HIGH: {inconsistent.sum()} rows have inconsistent calculated totals")
    
    # 4. Extreme outliers
    for col in numeric_cols:
        if len(df[col].dropna()) > 0:
            mean_val = df[col].mean()
            std_val = df[col].std()
            extreme_outliers = (abs(df[col] - mean_val) > 4 * std_val).sum()
            if extreme_outliers > 0:
                issues.append(f"LOW: {col} has {extreme_outliers} extreme outliers (>4 std devs)")
    
    # 5. Duplicate transactions
    if 'transaction_id' in df.columns:
        if df['transaction_id'].duplicated().any():
            dup_count = df['transaction_id'].duplicated().sum()
            issues.append(f"HIGH: {dup_count} duplicate transaction IDs found")
    
    # Print issues
    if issues:
        for issue in issues:
            print(f"• {issue}")
    else:
        print("✓ No major data quality issues detected")
    
    return issues

def generate_quick_insights(df):
    """
    Generate business-relevant insights from the data
    """
    
    print("\n" + "="*50)
    print("QUICK BUSINESS INSIGHTS")
    print("="*50)
    
    insights = []
    
    # Revenue insights
    if 'total_amount' in df.columns:
        total_revenue = df['total_amount'].sum()
        avg_transaction = df['total_amount'].mean()
        insights.append(f"Total revenue: ${total_revenue:,.2f}")
        insights.append(f"Average transaction value: ${avg_transaction:.2f}")
        
        # Revenue distribution
        high_value_threshold = df['total_amount'].quantile(0.9)
        high_value_transactions = (df['total_amount'] >= high_value_threshold).sum()
        high_value_revenue = df[df['total_amount'] >= high_value_threshold]['total_amount'].sum()
        insights.append(f"Top 10% transactions (≥${high_value_threshold:.2f}) represent {high_value_revenue/total_revenue*100:.1f}% of revenue")
    
    # Customer insights
    if 'customer_id' in df.columns:
        unique_customers = df['customer_id'].nunique()
        total_transactions = len(df)
        avg_transactions_per_customer = total_transactions / unique_customers
        insights.append(f"Unique customers: {unique_customers:,}")
        insights.append(f"Average transactions per customer: {avg_transactions_per_customer:.1f}")
        
        # Most active customers
        top_customers = df['customer_id'].value_counts().head()
        insights.append(f"Most active customer has {top_customers.iloc[0]} transactions")
    
    # Product insights
    if 'product_category' in df.columns:
        category_performance = df.groupby('product_category').agg({
            'total_amount': ['sum', 'count'] if 'total_amount' in df.columns else 'count'
        }).round(2)
        
        if 'total_amount' in df.columns:
            top_category = category_performance['total_amount']['sum'].idxmax()
            top_revenue = category_performance['total_amount']['sum'].max()
            insights.append(f"Top category by revenue: {top_category} (${top_revenue:,.2f})")
    
    # Time insights
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        date_range = df['transaction_date'].max() - df['transaction_date'].min()
        insights.append(f"Data spans: {date_range.days} days")
        
        # Daily patterns
        df['day_of_week'] = df['transaction_date'].dt.day_name()
        busiest_day = df['day_of_week'].value_counts().idxmax()
        insights.append(f"Busiest day: {busiest_day}")
    
    # Print insights
    for insight in insights:
        print(f"• {insight}")
    
    return insights

def main():
    """
    Complete warm-up exercise demonstrating systematic data exploration
    """
    
    print("FACULTY AI INTERVIEW - WARM-UP EXERCISE")
    print("Systematic Data Exploration")
    print("=" * 50)
    
    # Load the dataset
    try:
        df = pd.read_csv('datasets/exercise_00/sample_sales.csv')
        print(f"✓ Successfully loaded sample_sales.csv")
    except FileNotFoundError:
        print("✗ sample_sales.csv not found. Please run the data generation script first.")
        return
    
    # Main exploration
    explored_df = explore_dataset(df, "Sample Sales Data")
    
    # Identify data quality issues
    issues = identify_data_quality_issues(df)
    
    # Generate insights
    insights = generate_quick_insights(df)
    
    # Summary and next steps
    print("\n" + "="*50)
    print("EXPLORATION SUMMARY & NEXT STEPS")
    print("="*50)
    
    print(f"✓ Dataset successfully explored: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"✓ Identified {len(issues)} data quality issues")
    print(f"✓ Generated {len(insights)} business insights")
    
    print(f"\nRecommended next steps:")
    print(f"1. Address high-priority data quality issues")
    print(f"2. Implement data cleaning pipeline")
    print(f"3. Create feature engineering strategy")
    print(f"4. Set up data validation rules")
    
    # Questions to ask in interview
    print(f"\nQuestions I would ask in an interview:")
    print(f"• What specific business questions should this analysis answer?")
    print(f"• Are there known data quality issues I should be aware of?")
    print(f"• What's the expected refresh frequency for this data?")
    print(f"• Are there any regulatory or privacy constraints?")
    
    return df, issues, insights

if __name__ == "__main__":
    # Run the complete warm-up exercise
    df, issues, insights = main()