"""
Solution: E-commerce Data Pipeline Challenge
Demonstrates advanced pandas techniques for ML feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EcommerceFeatureEngine:
    """
    A comprehensive feature engineering pipeline for e-commerce data
    Demonstrates production-ready data processing patterns
    """
    
    def __init__(self):
        self.reference_date = datetime.now()
        self.quality_report = {}
    
    def load_and_validate_data(self, transactions_path, customers_path, products_path):
        """Load data with comprehensive quality checks"""
        
        print("Loading datasets...")
        
        # Load with appropriate dtypes for memory efficiency
        transactions = pd.read_csv(transactions_path, 
                                 dtype={'customer_id': 'category',
                                        'product_id': 'category',
                                        'payment_method': 'category'},
                                 parse_dates=['timestamp'])
        
        customers = pd.read_csv(customers_path,
                              dtype={'customer_id': 'category',
                                     'country': 'category',
                                     'email_domain': 'category',
                                     'acquisition_channel': 'category'},
                              parse_dates=['registration_date'])
        
        products = pd.read_csv(products_path,
                             dtype={'product_id': 'category',
                                    'category': 'category',
                                    'subcategory': 'category',
                                    'brand': 'category'})
        
        print(f"Loaded: {len(transactions):,} transactions, {len(customers):,} customers, {len(products):,} products")
        
        # Store for later use
        self.transactions_raw = transactions.copy()
        self.customers = customers
        self.products = products
        
        return self.assess_data_quality(transactions, customers, products)
    
    def assess_data_quality(self, transactions, customers, products):
        """Comprehensive data quality assessment"""
        
        print("\n=== DATA QUALITY ASSESSMENT ===")
        
        quality_issues = {
            'transactions': {},
            'customers': {},
            'products': {}
        }
        
        # 1. Missing values analysis
        print("\n1. Missing Values:")
        for name, df in [('transactions', transactions), ('customers', customers), ('products', products)]:
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(2)
            
            if missing.sum() > 0:
                print(f"\n{name.title()}:")
                for col in missing[missing > 0].index:
                    print(f"  {col}: {missing[col]:,} ({missing_pct[col]}%)")
                    quality_issues[name][f'missing_{col}'] = missing[col]
        
        # 2. Duplicate analysis
        print("\n2. Duplicate Analysis:")
        
        # Check for exact duplicates
        txn_exact_dups = transactions.duplicated().sum()
        print(f"Exact duplicate transactions: {txn_exact_dups:,}")
        
        # Check for business logic duplicates (same customer, product, timestamp)
        txn_business_dups = transactions.duplicated(['customer_id', 'product_id', 'timestamp']).sum()
        print(f"Business duplicate transactions: {txn_business_dups:,}")
        
        quality_issues['transactions']['exact_duplicates'] = txn_exact_dups
        quality_issues['transactions']['business_duplicates'] = txn_business_dups
        
        # 3. Outlier detection
        print("\n3. Outlier Analysis:")
        
        # Negative quantities
        negative_qty = (transactions['quantity'] < 0).sum()
        print(f"Negative quantities: {negative_qty:,}")
        quality_issues['transactions']['negative_quantities'] = negative_qty
        
        # Extreme prices (using IQR method)
        Q1 = transactions['price'].quantile(0.25)
        Q3 = transactions['price'].quantile(0.75)
        IQR = Q3 - Q1
        price_outliers = ((transactions['price'] < Q1 - 1.5*IQR) | 
                         (transactions['price'] > Q3 + 1.5*IQR)).sum()
        print(f"Price outliers (IQR method): {price_outliers:,}")
        quality_issues['transactions']['price_outliers'] = price_outliers
        
        # 4. Referential integrity
        print("\n4. Referential Integrity:")
        
        # Check if all customer_ids in transactions exist in customers
        orphan_customers = ~transactions['customer_id'].isin(customers['customer_id'])
        print(f"Transactions with missing customers: {orphan_customers.sum():,}")
        
        # Check if all product_ids in transactions exist in products
        orphan_products = ~transactions['product_id'].isin(products['product_id'])
        print(f"Transactions with missing products: {orphan_products.sum():,}")
        
        quality_issues['transactions']['orphan_customers'] = orphan_customers.sum()
        quality_issues['transactions']['orphan_products'] = orphan_products.sum()
        
        self.quality_report = quality_issues
        return transactions
    
    def clean_data(self, transactions):
        """Clean data based on quality assessment"""
        
        print("\n=== DATA CLEANING ===")
        
        original_count = len(transactions)
        
        # 1. Remove exact duplicates
        transactions = transactions.drop_duplicates()
        print(f"Removed {original_count - len(transactions):,} exact duplicates")
        
        # 2. Handle business duplicates (keep first occurrence)
        business_dup_mask = transactions.duplicated(['customer_id', 'product_id', 'timestamp'], keep='first')
        transactions = transactions[~business_dup_mask]
        print(f"Removed {business_dup_mask.sum():,} business duplicates")
        
        # 3. Fix negative quantities (assume data entry error - make positive)
        negative_mask = transactions['quantity'] < 0
        transactions.loc[negative_mask, 'quantity'] = abs(transactions.loc[negative_mask, 'quantity'])
        print(f"Fixed {negative_mask.sum():,} negative quantities")
        
        # 4. Handle extreme price outliers (cap at 99th percentile)
        price_99th = transactions['price'].quantile(0.99)
        extreme_price_mask = transactions['price'] > price_99th * 10  # 10x the 99th percentile
        transactions.loc[extreme_price_mask, 'price'] = price_99th
        print(f"Capped {extreme_price_mask.sum():,} extreme prices")
        
        # 5. Handle missing payment methods (impute with mode)
        missing_payment = transactions['payment_method'].isnull()
        if missing_payment.any():
            mode_payment = transactions['payment_method'].mode()[0]
            transactions.loc[missing_payment, 'payment_method'] = mode_payment
            print(f"Filled {missing_payment.sum():,} missing payment methods with '{mode_payment}'")
        
        print(f"Final transaction count: {len(transactions):,}")
        
        return transactions
    
    def create_enriched_transactions(self, transactions):
        """Add calculated fields to transactions"""
        
        print("\n=== ENRICHING TRANSACTIONS ===")
        
        # Create enriched copy
        enriched = transactions.copy()
        
        # Add revenue (price * quantity - discount)
        enriched['revenue'] = enriched['price'] * enriched['quantity'] - enriched['discount']
        
        # Add profit (using product margin)
        enriched = enriched.merge(self.products[['product_id', 'margin', 'category', 'brand']], 
                                on='product_id', how='left')
        enriched['profit'] = enriched['revenue'] * enriched['margin']
        
        # Add customer info
        enriched = enriched.merge(self.customers[['customer_id', 'registration_date', 'country']], 
                                on='customer_id', how='left')
        
        # Time-based features
        enriched['date'] = enriched['timestamp'].dt.date
        enriched['hour'] = enriched['timestamp'].dt.hour
        enriched['day_of_week'] = enriched['timestamp'].dt.day_name()
        enriched['month'] = enriched['timestamp'].dt.month
        enriched['quarter'] = enriched['timestamp'].dt.quarter
        
        # Customer tenure at time of purchase
        enriched['customer_tenure_days'] = (enriched['timestamp'] - enriched['registration_date']).dt.days
        
        print(f"Created enriched dataset with {len(enriched.columns)} columns")
        
        self.enriched_transactions = enriched
        return enriched
    
    def create_customer_features(self, enriched_transactions):
        """Create comprehensive customer-level features"""
        
        print("\n=== CREATING CUSTOMER FEATURES ===")
        
        # Use reference date for recency calculations
        ref_date = enriched_transactions['timestamp'].max()
        
        # 1. Basic aggregations
        basic_features = enriched_transactions.groupby('customer_id').agg({
            'transaction_id': 'count',
            'revenue': ['sum', 'mean', 'std'],
            'quantity': 'sum',
            'profit': 'sum',
            'timestamp': ['min', 'max'],
            'product_id': 'nunique',
            'category': 'nunique',
            'brand': 'nunique'
        }).round(2)
        
        # Flatten column names
        basic_features.columns = ['_'.join(col).strip() for col in basic_features.columns]
        basic_features = basic_features.rename(columns={
            'transaction_id_count': 'total_transactions',
            'revenue_sum': 'total_revenue',
            'revenue_mean': 'avg_order_value',
            'revenue_std': 'revenue_std',
            'quantity_sum': 'total_items',
            'profit_sum': 'total_profit',
            'timestamp_min': 'first_purchase',
            'timestamp_max': 'last_purchase',
            'product_id_nunique': 'unique_products',
            'category_nunique': 'unique_categories',
            'brand_nunique': 'unique_brands'
        })
        
        # 2. Derived features
        basic_features['days_active'] = (basic_features['last_purchase'] - basic_features['first_purchase']).dt.days + 1
        basic_features['avg_days_between_purchases'] = basic_features['days_active'] / basic_features['total_transactions']
        basic_features['days_since_last_purchase'] = (ref_date - basic_features['last_purchase']).dt.days
        basic_features['purchase_frequency'] = basic_features['total_transactions'] / basic_features['days_active'] * 7  # per week
        
        # 3. RFM Features
        basic_features['recency_score'] = pd.qcut(basic_features['days_since_last_purchase'], 5, labels=[5,4,3,2,1])
        basic_features['frequency_score'] = pd.qcut(basic_features['total_transactions'].rank(method='first'), 5, labels=[1,2,3,4,5])
        basic_features['monetary_score'] = pd.qcut(basic_features['total_revenue'], 5, labels=[1,2,3,4,5])
        
        # 4. Product affinity features
        category_features = self._create_category_features(enriched_transactions)
        brand_features = self._create_brand_features(enriched_transactions)
        
        # 5. Time-based features
        temporal_features = self._create_temporal_features(enriched_transactions)
        
        # 6. Advanced behavioral features
        behavioral_features = self._create_behavioral_features(enriched_transactions)
        
        # Combine all features
        customer_features = basic_features.join([
            category_features, 
            brand_features, 
            temporal_features, 
            behavioral_features
        ], how='outer')
        
        # Add customer demographics
        customer_demo = self.customers.set_index('customer_id')[['age', 'country', 'acquisition_channel']]
        customer_features = customer_features.join(customer_demo, how='left')
        
        print(f"Created {len(customer_features.columns)} features for {len(customer_features)} customers")
        
        return customer_features
    
    def _create_category_features(self, enriched_transactions):
        """Create category-based features"""
        
        # Revenue by category
        cat_revenue = enriched_transactions.groupby(['customer_id', 'category'])['revenue'].sum().unstack(fill_value=0)
        cat_revenue.columns = [f'revenue_{cat.lower()}' for cat in cat_revenue.columns]
        
        # Favorite category (by revenue)
        cat_totals = enriched_transactions.groupby(['customer_id', 'category'])['revenue'].sum()
        favorite_category = cat_totals.groupby('customer_id').idxmax().str[1]
        favorite_category.name = 'favorite_category'
        
        # Category diversity (Shannon entropy)
        cat_diversity = cat_totals.groupby('customer_id').apply(
            lambda x: -sum((x/x.sum()) * np.log(x/x.sum())) if len(x) > 1 else 0
        )
        cat_diversity.name = 'category_diversity'
        
        return pd.concat([cat_revenue, favorite_category, cat_diversity], axis=1)
    
    def _create_brand_features(self, enriched_transactions):
        """Create brand loyalty features"""
        
        # Brand concentration (Herfindahl index)
        brand_revenue = enriched_transactions.groupby(['customer_id', 'brand'])['revenue'].sum()
        brand_concentration = brand_revenue.groupby('customer_id').apply(
            lambda x: sum((x/x.sum())**2)
        )
        brand_concentration.name = 'brand_concentration'
        
        # Top brand share
        top_brand_share = brand_revenue.groupby('customer_id').apply(
            lambda x: x.max() / x.sum()
        )
        top_brand_share.name = 'top_brand_share'
        
        return pd.concat([brand_concentration, top_brand_share], axis=1)
    
    def _create_temporal_features(self, enriched_transactions):
        """Create time-based features"""
        
        # Quarterly spending
        quarterly = enriched_transactions.groupby(['customer_id', 'quarter'])['revenue'].sum().unstack(fill_value=0)
        quarterly.columns = [f'q{q}_revenue' for q in quarterly.columns]
        
        # Shopping patterns
        hourly_avg = enriched_transactions.groupby(['customer_id', 'hour'])['revenue'].mean()
        peak_hour = hourly_avg.groupby('customer_id').idxmax().str[1]
        peak_hour.name = 'peak_shopping_hour'
        
        # Weekend vs weekday
        enriched_transactions['is_weekend'] = enriched_transactions['day_of_week'].isin(['Saturday', 'Sunday'])
        weekend_pct = enriched_transactions.groupby('customer_id')['is_weekend'].mean()
        weekend_pct.name = 'weekend_purchase_pct'
        
        return pd.concat([quarterly, peak_hour, weekend_pct], axis=1)
    
    def _create_behavioral_features(self, enriched_transactions):
        """Create advanced behavioral features"""
        
        # Purchase acceleration (trend in frequency)
        def calculate_trend(group):
            if len(group) < 3:
                return 0
            
            group = group.sort_values('timestamp')
            days = (group['timestamp'] - group['timestamp'].iloc[0]).dt.days
            
            # Simple linear trend
            if days.max() == 0:
                return 0
            
            return np.corrcoef(days, range(len(days)))[0,1] if len(days) > 1 else 0
        
        purchase_trend = enriched_transactions.groupby('customer_id').apply(calculate_trend)
        purchase_trend.name = 'purchase_trend'
        
        # Discount sensitivity
        discount_usage = enriched_transactions.groupby('customer_id').apply(
            lambda x: (x['discount'] > 0).mean()
        )
        discount_usage.name = 'discount_usage_rate'
        
        # Average basket size variability
        basket_cv = enriched_transactions.groupby(['customer_id', 'date'])['quantity'].sum().groupby('customer_id').apply(
            lambda x: x.std() / x.mean() if x.mean() > 0 else 0
        )
        basket_cv.name = 'basket_size_cv'
        
        return pd.concat([purchase_trend, discount_usage, basket_cv], axis=1)
    
    def create_rolling_features(self, enriched_transactions, windows=[7, 30, 90]):
        """Create rolling window features"""
        
        print(f"\n=== CREATING ROLLING FEATURES ===")
        
        # Sort by customer and timestamp
        df = enriched_transactions.sort_values(['customer_id', 'timestamp']).copy()
        
        rolling_features = []
        
        for window in windows:
            print(f"Processing {window}-day windows...")
            
            # Daily aggregates first
            daily_agg = df.groupby(['customer_id', 'date']).agg({
                'revenue': 'sum',
                'quantity': 'sum',
                'transaction_id': 'count'
            }).reset_index()
            
            # Create rolling features
            daily_agg = daily_agg.sort_values(['customer_id', 'date'])
            
            rolling_cols = {}
            for metric in ['revenue', 'quantity', 'transaction_id']:
                rolling_series = daily_agg.groupby('customer_id')[metric].rolling(
                    window=window, min_periods=1
                ).mean()
                rolling_cols[f'{metric}_rolling_{window}d'] = rolling_series.values
            
            # Get the latest value for each customer
            latest_rolling = daily_agg.groupby('customer_id').last()[[]].copy()
            for col, values in rolling_cols.items():
                # Map rolling values back to the aggregated data
                temp_df = daily_agg.copy()
                temp_df[col] = values
                latest_values = temp_df.groupby('customer_id')[col].last()
                latest_rolling[col] = latest_values
            
            rolling_features.append(latest_rolling)
        
        # Combine all rolling features
        combined_rolling = rolling_features[0]
        for rf in rolling_features[1:]:
            combined_rolling = combined_rolling.join(rf, how='outer')
        
        print(f"Created {len(combined_rolling.columns)} rolling features")
        
        return combined_rolling

def run_complete_pipeline():
    """Run the complete feature engineering pipeline"""
    
    print("=== E-COMMERCE FEATURE ENGINEERING PIPELINE ===\n")
    
    # Initialize the feature engine
    engine = EcommerceFeatureEngine()
    
    # Load and validate data
    transactions = engine.load_and_validate_data(
        'datasets/exercise_01/transactions.csv',
        'datasets/exercise_01/customers.csv', 
        'datasets/exercise_01/products.csv'
    )
    
    # Clean the data
    clean_transactions = engine.clean_data(transactions)
    
    # Create enriched transactions
    enriched = engine.create_enriched_transactions(clean_transactions)
    
    # Create customer features
    customer_features = engine.create_customer_features(enriched)
    
    # Create rolling features
    rolling_features = engine.create_rolling_features(enriched)
    
    # Combine features
    final_features = customer_features.join(rolling_features, how='outer')
    
    print(f"\n=== FINAL FEATURE MATRIX ===")
    print(f"Shape: {final_features.shape}")
    print(f"Features: {list(final_features.columns)}")
    
    # Show sample
    print(f"\nSample features:")
    print(final_features.head())
    
    # Feature importance insights
    print(f"\n=== FEATURE INSIGHTS ===")
    
    # Missing value summary
    missing_summary = final_features.isnull().sum().sort_values(ascending=False)
    if missing_summary.sum() > 0:
        print(f"Features with missing values:")
        print(missing_summary[missing_summary > 0])
    else:
        print("No missing values in final feature matrix!")
    
    # Correlation with total revenue (proxy for importance)
    numeric_features = final_features.select_dtypes(include=[np.number])
    if 'total_revenue' in numeric_features.columns:
        correlations = numeric_features.corr()['total_revenue'].abs().sort_values(ascending=False)
        print(f"\nTop features correlated with total revenue:")
        print(correlations.head(10))
    
    return final_features, engine.quality_report

if __name__ == "__main__":
    # Generate sample data first
    print("Generating sample data...")
    import subprocess
    subprocess.run(["python", "datasets/generate_ecommerce_data.py"], cwd=".")
    
    # Run the complete pipeline
    features, quality_report = run_complete_pipeline()
    
    # Save results
    features.to_csv('customer_features.csv')
    print(f"\nSaved final features to 'customer_features.csv'")
    
    print(f"\n=== PIPELINE COMPLETE ===")