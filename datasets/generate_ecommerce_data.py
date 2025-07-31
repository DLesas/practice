"""
Generate realistic e-commerce datasets for interview practice
Includes intentional data quality issues to test cleaning skills
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_customers(n_customers=5000):
    """Generate customer data with realistic patterns"""
    
    # Email domains with different acquisition channels
    email_domains = {
        'gmail.com': 'organic', 'yahoo.com': 'organic', 'hotmail.com': 'organic',
        'company.com': 'b2b', 'university.edu': 'education',
        'startup.io': 'partnerships', 'enterprise.net': 'sales'
    }
    
    countries = ['US', 'UK', 'DE', 'FR', 'CA', 'AU', 'JP', 'BR']
    country_weights = [0.4, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05, 0.1]
    
    customers = []
    
    for i in range(n_customers):
        customer_id = f"CUST_{i+1:06d}"
        
        # Registration date (last 3 years)
        reg_date = datetime.now() - timedelta(days=np.random.randint(0, 1095))
        
        # Age with realistic distribution
        age = np.random.normal(35, 12)
        age = max(18, min(80, int(age)))
        
        # Country selection
        country = np.random.choice(countries, p=country_weights)
        
        # Email domain and acquisition channel
        domain = np.random.choice(list(email_domains.keys()))
        channel = email_domains[domain]
        
        # Add some missing values (realistic pattern)
        if np.random.random() < 0.03:  # 3% missing ages
            age = None
        if np.random.random() < 0.01:  # 1% missing countries
            country = None
            
        customers.append({
            'customer_id': customer_id,
            'registration_date': reg_date.strftime('%Y-%m-%d'),
            'age': age,
            'country': country,
            'email_domain': domain,
            'acquisition_channel': channel
        })
    
    return pd.DataFrame(customers)

def generate_products(n_products=1000):
    """Generate product catalog"""
    
    categories = {
        'Electronics': ['Smartphones', 'Laptops', 'Headphones', 'Cameras'],
        'Clothing': ['Shirts', 'Pants', 'Shoes', 'Accessories'],
        'Home': ['Furniture', 'Kitchen', 'Bedding', 'Decor'],
        'Books': ['Fiction', 'Non-fiction', 'Textbooks', 'Children'],
        'Sports': ['Equipment', 'Apparel', 'Supplements', 'Outdoor']
    }
    
    brands = ['Apple', 'Samsung', 'Nike', 'Adidas', 'IKEA', 'Amazon', 'Generic']
    
    products = []
    
    for i in range(n_products):
        product_id = f"PROD_{i+1:06d}"
        
        category = np.random.choice(list(categories.keys()))
        subcategory = np.random.choice(categories[category])
        
        # Brand selection with category bias
        if category == 'Electronics':
            brand = np.random.choice(['Apple', 'Samsung', 'Generic'], p=[0.3, 0.3, 0.4])
        elif category == 'Sports':
            brand = np.random.choice(['Nike', 'Adidas', 'Generic'], p=[0.4, 0.4, 0.2])
        else:
            brand = np.random.choice(brands)
        
        # Price based on category
        if category == 'Electronics':
            base_price = np.random.lognormal(5, 1)  # Higher prices
        elif category == 'Books':
            base_price = np.random.uniform(10, 50)   # Lower prices
        else:
            base_price = np.random.lognormal(3.5, 0.8)
        
        base_price = round(base_price, 2)
        margin = np.random.uniform(0.2, 0.6)  # 20-60% margin
        
        products.append({
            'product_id': product_id,
            'category': category,
            'subcategory': subcategory,
            'brand': brand,
            'base_price': base_price,
            'margin': margin
        })
    
    return pd.DataFrame(products)

def generate_transactions(customers_df, products_df, n_transactions=50000):
    """Generate transactions with realistic patterns and data quality issues"""
    
    customer_ids = customers_df['customer_id'].tolist()
    product_ids = products_df['product_id'].tolist()
    payment_methods = ['credit_card', 'debit_card', 'paypal', 'bank_transfer']
    
    transactions = []
    
    # Create customer purchase patterns
    customer_profiles = {}
    for customer_id in customer_ids:
        # Some customers are heavy buyers
        activity_level = np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1])
        customer_profiles[customer_id] = activity_level
    
    for i in range(n_transactions):
        transaction_id = f"TXN_{i+1:08d}"
        
        # Customer selection based on activity level
        customer_weights = [0.1 if customer_profiles[c] == 'low' 
                          else 0.5 if customer_profiles[c] == 'medium' 
                          else 1.0 for c in customer_ids]
        customer_weights = np.array(customer_weights) / sum(customer_weights)
        customer_id = np.random.choice(customer_ids, p=customer_weights)
        
        # Product selection
        product_id = np.random.choice(product_ids)
        product_info = products_df[products_df['product_id'] == product_id].iloc[0]
        
        # Quantity (mostly 1, sometimes more)
        quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.7, 0.15, 0.08, 0.04, 0.03])
        
        # Price with some variation from base price
        base_price = product_info['base_price']
        price_variation = np.random.uniform(0.8, 1.2)
        price = round(base_price * price_variation, 2)
        
        # Discount (sometimes)
        discount = 0
        if np.random.random() < 0.2:  # 20% chance of discount
            discount = round(np.random.uniform(0.05, 0.3) * price * quantity, 2)
        
        # Timestamp (last 2 years, with some seasonality)
        days_back = np.random.randint(0, 730)
        base_date = datetime.now() - timedelta(days=days_back)
        
        # Add some seasonality (more purchases in Nov-Dec)
        if base_date.month in [11, 12]:
            if np.random.random() < 0.3:  # 30% chance to add another purchase
                pass  # Keep this transaction
        
        timestamp = base_date + timedelta(
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        
        payment_method = np.random.choice(payment_methods)
        
        # Add data quality issues
        # 1. Some duplicate transactions (same customer, product, timestamp)
        if np.random.random() < 0.005:  # 0.5% duplicates
            # Create a duplicate by using the last transaction's data
            if len(transactions) > 0:
                last_txn = transactions[-1].copy()
                last_txn['transaction_id'] = transaction_id
                transactions.append(last_txn)
                continue
        
        # 2. Some negative quantities (data entry errors)
        if np.random.random() < 0.001:  # 0.1% negative quantities
            quantity = -quantity
        
        # 3. Some missing payment methods
        if np.random.random() < 0.02:  # 2% missing payment methods
            payment_method = None
            
        # 4. Some unrealistic high prices (data entry errors)
        if np.random.random() < 0.001:  # 0.1% price errors
            price = price * 100  # Accidentally entered cents as dollars
        
        transactions.append({
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'product_id': product_id,
            'quantity': quantity,
            'price': price,
            'discount': discount,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'payment_method': payment_method
        })
    
    return pd.DataFrame(transactions)

def main():
    """Generate all datasets"""
    
    # Create exercise directory if it doesn't exist
    import os
    os.makedirs('exercise_01', exist_ok=True)
    
    print("Generating customer data...")
    customers = generate_customers(5000)
    customers.to_csv('exercise_01/customers.csv', index=False)
    print(f"Generated {len(customers)} customers")
    
    print("Generating product data...")
    products = generate_products(1000)
    products.to_csv('exercise_01/products.csv', index=False)
    print(f"Generated {len(products)} products")
    
    print("Generating transaction data...")
    transactions = generate_transactions(customers, products, 50000)
    transactions.to_csv('exercise_01/transactions.csv', index=False)
    print(f"Generated {len(transactions)} transactions")
    
    print("\nData quality summary:")
    print(f"Customers with missing age: {customers['age'].isna().sum()}")
    print(f"Customers with missing country: {customers['country'].isna().sum()}")
    print(f"Transactions with missing payment method: {transactions['payment_method'].isna().sum()}")
    print(f"Transactions with negative quantity: {(transactions['quantity'] < 0).sum()}")
    print(f"Potential duplicate transactions: {transactions.duplicated(['customer_id', 'product_id', 'timestamp']).sum()}")
    
    # Show sample data
    print("\nSample data:")
    print("\nCustomers:")
    print(customers.head())
    print("\nProducts:")
    print(products.head())
    print("\nTransactions:")
    print(transactions.head())

if __name__ == "__main__":
    main()