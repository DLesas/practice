"""
Quick test script to verify all datasets load correctly
Run this to confirm your environment is ready for practice
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Test ML packages
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):  # Catch XGBoostError and other issues
    XGBOOST_AVAILABLE = False
    xgb = None  # Set to None so we can check later

ML_PACKAGES_AVAILABLE = SKLEARN_AVAILABLE  # Core ML functionality only needs sklearn

def test_environment():
    """Test that all packages are working"""
    print("=== TESTING ENVIRONMENT ===")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    if SKLEARN_AVAILABLE:
        from sklearn import __version__ as sklearn_version
        print(f"Scikit-learn version: {sklearn_version}")
        
        if XGBOOST_AVAILABLE and xgb is not None:
            print(f"XGBoost version: {xgb.__version__}")
            print("✓ All ML packages imported successfully\n")
        else:
            print("⚠️  XGBoost not available (may need: brew install libomp)")
            print("✓ Core ML functionality available with scikit-learn\n")
    else:
        print("⚠️  ML packages not available - basic exercises only")
        print("Run 'uv add scikit-learn xgboost' to enable ML exercises\n")

def test_datasets():
    """Test that all datasets load correctly"""
    print("=== TESTING DATASETS ===")
    
    datasets = {
        'Sample Sales': 'datasets/sample_sales.csv',
        'Customers': 'datasets/customers.csv', 
        'Products': 'datasets/products.csv',
        'Transactions': 'datasets/transactions.csv',
        'Sensor Metadata': 'datasets/sensor_metadata.csv',
        'Sensor Readings': 'datasets/sensor_readings.csv',
        'A/B Test Results': 'datasets/ab_test_results.csv',
        'Interventions': 'datasets/interventions.csv'
    }
    
    for name, path in datasets.items():
        if os.path.exists(path):
            try:
                if name == 'Sensor Readings':
                    # Only read a sample of the large file
                    df = pd.read_csv(path, nrows=1000)
                    print(f"✓ {name}: {path} (sample of 1000 rows loaded)")
                else:
                    df = pd.read_csv(path)
                    print(f"✓ {name}: {path} ({len(df):,} rows)")
            except Exception as e:
                print(f"✗ {name}: Error loading {path} - {e}")
        else:
            print(f"✗ {name}: File not found - {path}")
    
    print()

def test_basic_operations():
    """Test basic pandas operations on sample data"""
    print("=== TESTING BASIC OPERATIONS ===")
    
    try:
        # Load sample sales data
        df = pd.read_csv('datasets/sample_sales.csv')
        
        # Test basic operations
        print(f"Sample sales shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Unique customers: {df['customer_id'].nunique()}")
        print(f"Total revenue: ${df['total_amount'].sum():,.2f}")
        print("✓ Basic pandas operations working\n")
        
        # Test time series operations with sensor data (small sample)
        sensor_df = pd.read_csv('datasets/sensor_readings.csv', nrows=1000)
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
        print(f"Sensor data time range: {sensor_df['timestamp'].min()} to {sensor_df['timestamp'].max()}")
        print("✓ Time series operations working\n")
        
    except Exception as e:
        print(f"✗ Error in basic operations: {e}")

def test_memory_usage():
    """Check memory usage of datasets"""
    print("=== MEMORY USAGE ANALYSIS ===")
    
    try:
        # Test different loading strategies
        print("Loading strategies comparison:")
        
        # Standard loading
        df1 = pd.read_csv('datasets/sample_sales.csv')
        mem1 = df1.memory_usage(deep=True).sum() / 1024**2
        print(f"Standard loading: {mem1:.2f} MB")
        
        # Optimized loading
        df2 = pd.read_csv('datasets/sample_sales.csv', 
                         dtype={'customer_id': 'category',
                                'product_category': 'category',
                                'sales_rep': 'category',
                                'region': 'category'})
        mem2 = df2.memory_usage(deep=True).sum() / 1024**2
        print(f"Optimized loading: {mem2:.2f} MB")
        print(f"Memory savings: {((mem1-mem2)/mem1*100):.1f}%")
        print("✓ Memory optimization working\n")
        
    except Exception as e:
        print(f"✗ Error in memory test: {e}")

def test_ml_functionality():
    """Test ML packages and basic functionality"""
    print("=== ML FUNCTIONALITY TEST ===")
    
    if not ML_PACKAGES_AVAILABLE:
        print("⚠️ ML packages not available - skipping ML tests")
        return
    
    try:
        # Test basic ML pipeline
        print("Testing basic ML functionality...")
        
        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Test train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Test RandomForest
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        rf_score = rf.score(X_test, y_test)
        
        print(f"✓ RandomForest test accuracy: {rf_score:.3f}")
        
        # Test XGBoost if available
        if XGBOOST_AVAILABLE and xgb is not None:
            xgb_model = xgb.XGBClassifier(n_estimators=10, random_state=42, eval_metric='logloss')
            xgb_model.fit(X_train, y_train)
            xgb_score = xgb_model.score(X_test, y_test)
            print(f"✓ XGBoost test accuracy: {xgb_score:.3f}")
        else:
            print("⚠️  XGBoost test skipped (not available)")
            
        print("✓ ML functionality working correctly\n")
        
    except Exception as e:
        print(f"✗ Error in ML test: {e}")

def main():
    """Run all tests"""
    print("Faculty AI Interview Preparation - Environment Test")
    print("=" * 50)
    
    test_environment()
    test_datasets()
    test_basic_operations()
    test_memory_usage()
    test_ml_functionality()
    
    print("=== READY FOR PRACTICE! ===")
    print("Your environment is set up correctly.")
    print("\nAvailable exercises:")
    print("1. exercises/00_warmup_data_exploration.md (15 min)")
    print("2. exercises/01_data_pipeline_challenge.md (60 min)")
    print("3. exercises/02_time_series_anomaly_detection.md (45 min)")
    print("4. exercises/03_performance_optimization.md (30 min)")
    
    if ML_PACKAGES_AVAILABLE:
        print("5. exercises/04_ml_model_pipeline.md (60 min)")
        print("6. exercises/05_ab_testing_analysis.md (45 min)")
    else:
        print("5-6. ML exercises (install scikit-learn & xgboost to enable)")
    
    print("\nRecommended practice order:")
    print("• Start with 00_warmup + 01_data_pipeline (90 min total)")
    print("• Read interview_tips.md for communication strategies")
    print("• Use practice_session_guide.md for structured practice")
    print("• Add ML exercises for comprehensive ML engineer prep")

if __name__ == "__main__":
    main()