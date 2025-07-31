"""
Generate sample sales dataset for warm-up exercise
Includes various data types and some quality issues
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seeds
np.random.seed(42)
random.seed(42)

def generate_sample_sales(n_records=1000):
    """Generate realistic sales dataset with common issues"""
    
    # Product categories and products
    categories = {
        'Electronics': ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Camera'],
        'Clothing': ['Shirt', 'Pants', 'Shoes', 'Jacket', 'Hat'],
        'Books': ['Fiction', 'Non-fiction', 'Textbook', 'Magazine', 'Comic'],
        'Home': ['Furniture', 'Kitchen', 'Bedding', 'Decor', 'Tools'],
        'Sports': ['Equipment', 'Apparel', 'Shoes', 'Accessories', 'Supplements']
    }
    
    regions = ['North', 'South', 'East', 'West', 'Central']
    sales_reps = ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson', 'Eva Brown']
    
    records = []
    
    for i in range(n_records):
        # Basic transaction info
        transaction_id = f"TXN_{i+1:06d}"
        
        # Date (last 12 months)
        transaction_date = datetime.now() - timedelta(days=np.random.randint(0, 365))
        
        # Product selection
        category = np.random.choice(list(categories.keys()))
        product = np.random.choice(categories[category])
        
        # Quantity (mostly 1-3, occasionally more)
        quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.3, 0.12, 0.05, 0.03])
        
        # Price based on category
        if category == 'Electronics':
            unit_price = np.random.uniform(200, 2000)
        elif category == 'Books':
            unit_price = np.random.uniform(10, 80)
        elif category == 'Clothing':
            unit_price = np.random.uniform(25, 150)
        elif category == 'Home':
            unit_price = np.random.uniform(50, 500)
        else:  # Sports
            unit_price = np.random.uniform(30, 300)
        
        unit_price = round(unit_price, 2)
        total_amount = round(unit_price * quantity, 2)
        
        # Geographic info
        region = np.random.choice(regions)
        sales_rep = np.random.choice(sales_reps)
        
        # Customer info
        customer_id = f"CUST_{np.random.randint(1, 200):04d}"
        
        # Add some data quality issues
        
        # 1. Missing values (realistic patterns)
        if np.random.random() < 0.02:  # 2% missing sales rep
            sales_rep = None
        if np.random.random() < 0.01:  # 1% missing region
            region = None
        
        # 2. Duplicate transactions (data entry errors)
        if np.random.random() < 0.01 and len(records) > 0:  # 1% duplicates
            # Duplicate the last record but change transaction_id
            duplicate_record = records[-1].copy()
            duplicate_record['transaction_id'] = transaction_id
            records.append(duplicate_record)
            continue
        
        # 3. Negative quantities (data entry errors)
        if np.random.random() < 0.005:  # 0.5% negative quantities
            quantity = -quantity
            total_amount = unit_price * quantity
        
        # 4. Inconsistent data (total doesn't match unit_price * quantity)
        if np.random.random() < 0.01:  # 1% calculation errors
            total_amount = round(total_amount * np.random.uniform(0.8, 1.3), 2)
        
        # 5. Outlier prices
        if np.random.random() < 0.005:  # 0.5% price outliers
            unit_price = unit_price * 10  # Accidentally added a zero
            total_amount = unit_price * quantity
        
        record = {
            'transaction_id': transaction_id,
            'transaction_date': transaction_date.strftime('%Y-%m-%d'),
            'customer_id': customer_id,
            'product_category': category,
            'product_name': product,
            'quantity': quantity,
            'unit_price': unit_price,
            'total_amount': total_amount,
            'sales_rep': sales_rep,
            'region': region
        }
        
        records.append(record)
    
    return pd.DataFrame(records)

def main():
    """Generate and save sample sales data"""
    
    print("Generating sample sales data...")
    
    # Create exercise directory if it doesn't exist
    import os
    os.makedirs('exercise_00', exist_ok=True)
    
    # Generate the dataset
    sales_df = generate_sample_sales(1000)
    
    # Save to CSV
    sales_df.to_csv('exercise_00/sample_sales.csv', index=False)
    
    print(f"Generated {len(sales_df)} sales records")
    
    # Data quality summary
    print("\nData Quality Summary:")
    print(f"Missing sales_rep: {sales_df['sales_rep'].isna().sum()}")
    print(f"Missing region: {sales_df['region'].isna().sum()}")
    print(f"Negative quantities: {(sales_df['quantity'] < 0).sum()}")
    
    # Check for duplicates (excluding transaction_id)
    cols_to_check = ['customer_id', 'product_category', 'product_name', 'transaction_date']
    duplicates = sales_df.duplicated(subset=cols_to_check).sum()
    print(f"Potential duplicate transactions: {duplicates}")
    
    # Price outliers
    price_q99 = sales_df['unit_price'].quantile(0.99)
    outliers = (sales_df['unit_price'] > price_q99 * 2).sum()
    print(f"Price outliers: {outliers}")
    
    # Calculation inconsistencies
    expected_total = sales_df['unit_price'] * sales_df['quantity']
    inconsistent = abs(sales_df['total_amount'] - expected_total) > 0.01
    print(f"Calculation inconsistencies: {inconsistent.sum()}")
    
    print("\nSample data:")
    print(sales_df.head())
    
    # Basic statistics
    print("\nBasic statistics:")
    print(sales_df.describe())

if __name__ == "__main__":
    main()