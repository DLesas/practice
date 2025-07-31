"""
Solution: Pandas Performance Optimization Challenge
Demonstrates comprehensive performance optimization techniques
"""

import pandas as pd
import numpy as np
import time
import psutil
import os
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

def measure_performance(func):
    """Decorator to measure execution time and memory usage"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time execution
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_used = memory_after - memory_before
        
        print(f"  ⏱️  {func.__name__}: {execution_time:.4f}s, Memory: {memory_used:+.1f}MB")
        
        return result, execution_time, memory_used
    return wrapper

class PerformanceOptimizer:
    """
    Comprehensive pandas performance optimization demonstrations
    """
    
    def __init__(self):
        self.results = {}
    
    def create_test_data(self, n_rows=100000):
        """Create large test dataset for performance testing"""
        
        print("=== CREATING TEST DATA ===")
        
        np.random.seed(42)
        
        data = {
            'customer_id': np.random.choice([f'CUST_{i:06d}' for i in range(1, 10001)], n_rows),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_rows),
            'product_name': np.random.choice([f'Product_{i}' for i in range(1, 1001)], n_rows),
            'amount': np.random.lognormal(3, 1, n_rows),
            'quantity': np.random.randint(1, 6, n_rows),
            'timestamp': pd.date_range('2023-01-01', periods=n_rows, freq='1min'),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
            'sales_rep': np.random.choice([f'Rep_{i}' for i in range(1, 101)], n_rows),
            'discount_rate': np.random.uniform(0, 0.3, n_rows),
            'is_weekend': np.random.choice([True, False], n_rows, p=[0.3, 0.7])
        }
        
        df = pd.DataFrame(data)
        print(f"Created test dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df
    
    def memory_optimization_demo(self, df):
        """Demonstrate memory optimization techniques"""
        
        print("\n=== MEMORY OPTIMIZATION ===")
        
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        print(f"Original memory usage: {original_memory:.1f} MB")
        
        # 1. Categorical optimization
        print("\n1. Categorical Data Optimization:")
        df_optimized = df.copy()
        
        categorical_candidates = ['customer_id', 'product_category', 'product_name', 'region', 'sales_rep']
        
        for col in categorical_candidates:
            if col in df_optimized.columns:
                unique_ratio = df_optimized[col].nunique() / len(df_optimized)
                print(f"  {col}: {df_optimized[col].nunique():,} unique values ({unique_ratio:.3f} ratio)")
                
                if unique_ratio < 0.5:  # Good candidate for categorical
                    memory_before = df_optimized[col].memory_usage(deep=True) / 1024**2
                    df_optimized[col] = df_optimized[col].astype('category')
                    memory_after = df_optimized[col].memory_usage(deep=True) / 1024**2
                    savings = memory_before - memory_after
                    print(f"    Converted to category: {savings:.1f} MB saved ({savings/memory_before*100:.1f}%)")
        
        # 2. Numeric downcastingl
        print("\n2. Numeric Downcasting:")
        for col in df_optimized.select_dtypes(include=[np.number]).columns:
            if col in ['amount', 'discount_rate']:
                # Float downcasting
                memory_before = df_optimized[col].memory_usage(deep=True) / 1024**2
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
                memory_after = df_optimized[col].memory_usage(deep=True) / 1024**2
                savings = memory_before - memory_after
                print(f"  {col}: {savings:.1f} MB saved (float downcasting)")
            
            elif col == 'quantity':
                # Integer downcasting
                memory_before = df_optimized[col].memory_usage(deep=True) / 1024**2
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
                memory_after = df_optimized[col].memory_usage(deep=True) / 1024**2
                savings = memory_before - memory_after
                print(f"  {col}: {savings:.1f} MB saved (integer downcasting)")
        
        # 3. Boolean optimization
        print("\n3. Boolean Optimization:")
        if 'is_weekend' in df_optimized.columns:
            memory_before = df_optimized['is_weekend'].memory_usage(deep=True) / 1024**2
            # Boolean is already optimized in pandas, but showing the concept
            df_optimized['is_weekend'] = df_optimized['is_weekend'].astype('bool')
            memory_after = df_optimized['is_weekend'].memory_usage(deep=True) / 1024**2
            print(f"  is_weekend: Already optimized as boolean")
        
        final_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
        total_savings = original_memory - final_memory
        print(f"\nTotal memory savings: {total_savings:.1f} MB ({total_savings/original_memory*100:.1f}%)")
        
        return df_optimized
    
    def vectorization_demo(self, df):
        """Demonstrate vectorization vs loops"""
        
        print("\n=== VECTORIZATION OPTIMIZATION ===")
        
        # Use smaller dataset for timing comparisons
        df_test = df.head(10000).copy()
        
        print("1. Conditional Logic Optimization:")
        
        # SLOW: Using loops
        @measure_performance
        def slow_categorization(data):
            categories = []
            for amount in data['amount']:
                if amount < 50:
                    categories.append('Low')
                elif amount < 200:
                    categories.append('Medium')
                else:
                    categories.append('High')
            return categories
        
        # FAST: Using vectorized operations
        @measure_performance  
        def fast_categorization(data):
            return pd.cut(data['amount'], 
                         bins=[0, 50, 200, float('inf')], 
                         labels=['Low', 'Medium', 'High'])
        
        # FASTER: Using numpy.where
        @measure_performance
        def fastest_categorization(data):
            return np.where(data['amount'] < 50, 'Low',
                          np.where(data['amount'] < 200, 'Medium', 'High'))
        
        slow_result, slow_time, _ = slow_categorization(df_test)
        fast_result, fast_time, _ = fast_categorization(df_test)
        fastest_result, fastest_time, _ = fastest_categorization(df_test)
        
        print(f"  Speedup (pd.cut): {slow_time / fast_time:.1f}x faster")
        print(f"  Speedup (np.where): {slow_time / fastest_time:.1f}x faster")
        
        print("\n2. String Operations Optimization:")
        
        # SLOW: Using apply with lambda
        @measure_performance
        def slow_string_ops(data):
            return data['customer_id'].apply(lambda x: x.upper() if isinstance(x, str) else x)
        
        # FAST: Using vectorized string methods
        @measure_performance
        def fast_string_ops(data):
            return data['customer_id'].str.upper()
        
        slow_str_result, slow_str_time, _ = slow_string_ops(df_test)
        fast_str_result, fast_str_time, _ = fast_string_ops(df_test)
        
        print(f"  Speedup (vectorized strings): {slow_str_time / fast_str_time:.1f}x faster")
        
        return df_test
    
    def groupby_optimization_demo(self, df):
        """Demonstrate groupby optimization techniques"""
        
        print("\n=== GROUPBY OPTIMIZATION ===")
        
        df_test = df.head(50000).copy()
        
        print("1. Transform vs Apply:")
        
        # SLOW: Using apply for element-wise operations
        @measure_performance
        def slow_groupby_apply(data):
            return data.groupby('product_category')['amount'].apply(
                lambda x: (x - x.mean()) / x.std()
            )
        
        # FAST: Using transform for element-wise operations
        @measure_performance
        def fast_groupby_transform(data):
            return data.groupby('product_category')['amount'].transform(
                lambda x: (x - x.mean()) / x.std()
            )
        
        slow_gb_result, slow_gb_time, _ = slow_groupby_apply(df_test)
        fast_gb_result, fast_gb_time, _ = fast_groupby_transform(df_test)
        
        print(f"  Speedup (transform): {slow_gb_time / fast_gb_time:.1f}x faster")
        
        print("\n2. Multiple Aggregations Optimization:")
        
        # SLOW: Multiple separate groupby operations
        @measure_performance
        def slow_multiple_aggs(data):
            result1 = data.groupby('product_category')['amount'].sum()
            result2 = data.groupby('product_category')['amount'].mean()
            result3 = data.groupby('product_category')['quantity'].sum()
            return pd.DataFrame({
                'amount_sum': result1,
                'amount_mean': result2,
                'quantity_sum': result3
            })
        
        # FAST: Single groupby with multiple aggregations
        @measure_performance
        def fast_multiple_aggs(data):
            return data.groupby('product_category').agg({
                'amount': ['sum', 'mean'],
                'quantity': 'sum'
            })
        
        slow_agg_result, slow_agg_time, _ = slow_multiple_aggs(df_test)
        fast_agg_result, fast_agg_time, _ = fast_multiple_aggs(df_test)
        
        print(f"  Speedup (combined aggs): {slow_agg_time / fast_agg_time:.1f}x faster")
        
        print("\n3. Custom Aggregation Functions:")
        
        # FAST: Using built-in functions
        @measure_performance
        def fast_builtin_agg(data):
            return data.groupby('product_category')['amount'].agg(['mean', 'std', 'count'])
        
        # SLOWER: Custom aggregation function
        @measure_performance
        def slower_custom_agg(data):
            def custom_stats(x):
                return pd.Series({
                    'mean': x.mean(),
                    'std': x.std(),
                    'count': x.count()
                })
            return data.groupby('product_category')['amount'].apply(custom_stats)
        
        fast_builtin_result, fast_builtin_time, _ = fast_builtin_agg(df_test)
        slower_custom_result, slower_custom_time, _ = slower_custom_agg(df_test)
        
        print(f"  Builtin vs Custom: {slower_custom_time / fast_builtin_time:.1f}x difference")
        
        return df_test
    
    def join_optimization_demo(self, df):
        """Demonstrate join optimization techniques"""
        
        print("\n=== JOIN OPTIMIZATION ===")
        
        # Create lookup table
        customer_lookup = pd.DataFrame({
            'customer_id': df['customer_id'].unique(),
            'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], 
                                               size=df['customer_id'].nunique()),
            'customer_age': np.random.randint(18, 80, size=df['customer_id'].nunique())
        })
        
        df_test = df.head(30000).copy()
        
        print("1. Index-based Joins:")
        
        # SLOW: Merge without index
        @measure_performance
        def slow_merge(left_data, right_data):
            return left_data.merge(right_data, on='customer_id', how='left')
        
        # FAST: Merge with index
        @measure_performance
        def fast_merge_with_index(left_data, right_data):
            right_indexed = right_data.set_index('customer_id')
            return left_data.merge(right_indexed, left_on='customer_id', right_index=True, how='left')
        
        slow_merge_result, slow_merge_time, _ = slow_merge(df_test, customer_lookup)
        fast_merge_result, fast_merge_time, _ = fast_merge_with_index(df_test, customer_lookup)
        
        print(f"  Speedup (indexed join): {slow_merge_time / fast_merge_time:.1f}x faster")
        
        print("\n2. Sorted Merges:")
        
        # Sort both dataframes for optimal merge performance
        @measure_performance
        def optimized_sorted_merge(left_data, right_data):
            left_sorted = left_data.sort_values('customer_id')
            right_sorted = right_data.sort_values('customer_id')
            return left_sorted.merge(right_sorted, on='customer_id', how='left')
        
        sorted_merge_result, sorted_merge_time, _ = optimized_sorted_merge(df_test, customer_lookup)
        
        print(f"  Sorted merge time: {sorted_merge_time:.4f}s")
        
        return df_test
    
    def chunked_processing_demo(self, large_file_path=None):
        """Demonstrate chunked processing for large datasets"""
        
        print("\n=== CHUNKED PROCESSING ===")
        
        if large_file_path and os.path.exists(large_file_path):
            print("Processing large file in chunks:")
            
            @measure_performance
            def process_in_chunks(file_path, chunk_size=10000):
                results = []
                
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    # Process each chunk
                    chunk_result = chunk.groupby('product_category')['amount'].sum()
                    results.append(chunk_result)
                
                # Combine results
                final_result = pd.concat(results).groupby(level=0).sum()
                return final_result
            
            chunk_result, chunk_time, _ = process_in_chunks(large_file_path)
            print(f"  Processed large file in chunks")
            
        else:
            print("Simulating chunked processing:")
            
            # Create large dataset simulation
            large_df = self.create_test_data(200000)
            
            @measure_performance
            def simulate_chunked_processing(data, chunk_size=20000):
                results = []
                
                for i in range(0, len(data), chunk_size):
                    chunk = data.iloc[i:i+chunk_size]
                    chunk_result = chunk.groupby('product_category')['amount'].sum()
                    results.append(chunk_result)
                
                # Combine results
                final_result = pd.concat(results).groupby(level=0).sum()
                return final_result
            
            chunk_result, chunk_time, _ = simulate_chunked_processing(large_df)
            print(f"  Processed {len(large_df):,} rows in chunks")
    
    def query_optimization_demo(self, df):
        """Demonstrate query optimization techniques"""
        
        print("\n=== QUERY OPTIMIZATION ===")
        
        df_test = df.head(50000).copy()
        
        print("1. Boolean Indexing vs Query:")
        
        # Traditional boolean indexing
        @measure_performance
        def boolean_indexing(data):
            return data[(data['amount'] > 100) & 
                       (data['product_category'] == 'Electronics') & 
                       (data['quantity'] >= 2)]
        
        # Using query method
        @measure_performance
        def query_method(data):
            return data.query("amount > 100 and product_category == 'Electronics' and quantity >= 2")
        
        bool_result, bool_time, _ = boolean_indexing(df_test)
        query_result, query_time, _ = query_method(df_test)
        
        print(f"  Query method performance: {bool_time / query_time:.1f}x difference")
        
        print("\n2. Eval for Complex Expressions:")
        
        # Traditional calculation
        @measure_performance
        def traditional_calc(data):
            return data['amount'] * data['quantity'] - data['amount'] * data['discount_rate']
        
        # Using eval
        @measure_performance
        def eval_calc(data):
            return data.eval('amount * quantity - amount * discount_rate')
        
        trad_result, trad_time, _ = traditional_calc(df_test)
        eval_result, eval_time, _ = eval_calc(df_test)
        
        print(f"  Eval performance: {trad_time / eval_time:.1f}x difference")
        
        return df_test
    
    def generate_performance_report(self):
        """Generate comprehensive performance optimization report"""
        
        print("\n" + "="*60)
        print("PERFORMANCE OPTIMIZATION REPORT")
        print("="*60)
        
        print("\nKey Optimization Techniques Demonstrated:")
        print("1. ✅ Memory Optimization:")
        print("   • Categorical data conversion (up to 90% memory savings)")
        print("   • Numeric downcasting (10-50% savings)")
        print("   • Boolean optimization")
        
        print("\n2. ✅ Vectorization:")
        print("   • Replaced loops with pandas/numpy operations (10-100x speedup)")
        print("   • Used pd.cut, np.where for conditional logic")
        print("   • Leveraged vectorized string methods")
        
        print("\n3. ✅ GroupBy Optimization:")
        print("   • Transform vs apply for element-wise operations")
        print("   • Combined multiple aggregations in single operation")
        print("   • Used built-in functions over custom ones")
        
        print("\n4. ✅ Join Optimization:")
        print("   • Index-based merges for faster joins")
        print("   • Sorted data for optimal merge performance")
        print("   • Proper join strategy selection")
        
        print("\n5. ✅ Query Optimization:")
        print("   • df.query() for complex boolean logic")
        print("   • df.eval() for complex mathematical expressions")
        print("   • Chunked processing for large datasets")
        
        print("\nProduction Recommendations:")
        print("• Profile code regularly to identify bottlenecks")
        print("• Use appropriate data types from the start")
        print("• Consider Dask or Polars for very large datasets")
        print("• Implement caching for expensive operations")
        print("• Monitor memory usage in production")
        print("• Use vectorized operations wherever possible")

def run_complete_optimization_demo():
    """Run the complete performance optimization demonstration"""
    
    print("=== PANDAS PERFORMANCE OPTIMIZATION CHALLENGE ===\n")
    
    optimizer = PerformanceOptimizer()
    
    # Create test data
    print("Creating test dataset for optimization demos...")
    df = optimizer.create_test_data(100000)
    
    # Run optimization demonstrations
    df_optimized = optimizer.memory_optimization_demo(df)
    optimizer.vectorization_demo(df_optimized)
    optimizer.groupby_optimization_demo(df_optimized)
    optimizer.join_optimization_demo(df_optimized)
    optimizer.chunked_processing_demo()
    optimizer.query_optimization_demo(df_optimized)
    
    # Generate final report
    optimizer.generate_performance_report()
    
    print(f"\n=== SCALABILITY CONSIDERATIONS ===")
    print("For production systems processing large datasets:")
    print("• Use Dask for out-of-core processing (datasets > RAM)")
    print("• Consider Polars for even better performance")
    print("• Implement data pipeline with Apache Airflow")
    print("• Use columnar storage formats (Parquet, Arrow)")
    print("• Leverage distributed computing (Spark, Ray)")
    print("• Implement proper data partitioning strategies")
    
    return optimizer, df_optimized

if __name__ == "__main__":
    # Check if psutil is available for memory monitoring
    try:
        import psutil
        optimizer, df = run_complete_optimization_demo()
    except ImportError:
        print("Note: Install psutil for memory monitoring: pip install psutil")
        # Run without memory monitoring
        optimizer = PerformanceOptimizer()
        df = optimizer.create_test_data(50000)
        df_opt = optimizer.memory_optimization_demo(df)
        optimizer.vectorization_demo(df_opt)
        optimizer.generate_performance_report()