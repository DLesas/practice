# Exercise 3: Pandas Performance Optimization Challenge
**Difficulty**: Expert  
**Time**: 30-45 minutes  
**Skills**: Performance optimization, memory management, vectorization

## Business Context
Your ML training pipeline is processing 10GB+ datasets daily, but current pandas operations are too slow and causing memory issues. You need to optimize the data processing pipeline to handle larger datasets efficiently.

## The Challenge
Given a large dataset simulation, optimize these common operations:
1. **Memory-efficient data loading**
2. **Vectorized feature engineering**  
3. **Optimized groupby operations**
4. **Efficient joins on large datasets**

## Sample Performance Issues

### Slow Code Examples
```python
# SLOW: Iterating through rows
def slow_feature_engineering(df):
    results = []
    for _, row in df.iterrows():
        # Complex calculation per row
        result = some_calculation(row)
        results.append(result)
    return results

# SLOW: Non-vectorized operations
def slow_categorization(df):
    categories = []
    for value in df['amount']:
        if value < 100:
            categories.append('small')
        elif value < 1000:
            categories.append('medium')
        else:
            categories.append('large')
    return categories

# SLOW: Inefficient groupby
def slow_aggregation(df):
    result = df.groupby('category').apply(lambda x: complex_calculation(x))
    return result
```

## Requirements

### Part 1: Memory Optimization (10 min)
1. **Optimize data types**:
   - Convert appropriate columns to categorical
   - Use smallest possible numeric types
   - Implement chunked reading for large files

2. **Memory monitoring**:
   - Measure memory usage before/after optimizations
   - Identify memory bottlenecks
   - Implement memory-efficient alternatives

### Part 2: Vectorization (15 min)
1. **Replace loops with vectorized operations**:
   - Convert iterrows() to vectorized calculations
   - Use numpy operations where possible
   - Implement efficient conditional logic

2. **Optimize string operations**:
   - Vectorized string processing
   - Efficient regex operations
   - Category-based string handling

### Part 3: Advanced Optimizations (15 min)
1. **Groupby optimization**:
   - Use transform() vs apply() appropriately
   - Implement custom aggregation functions
   - Parallel processing considerations

2. **Join optimization**:
   - Index-based merges
   - Efficient handling of large joins
   - Memory-conscious merge strategies

## Performance Benchmarks to Beat

```python
# Target improvements:
# - 10x speed improvement on feature engineering
# - 50% memory reduction on data loading
# - 5x faster groupby operations
# - 3x faster large dataset joins
```

## Key Pandas Performance Techniques
- **Memory**: dtype optimization, categorical data, chunking
- **Speed**: vectorization, avoid loops, efficient indexing
- **Groupby**: transform vs apply, custom aggregations
- **Joins**: sorted merges, index optimization
- **Strings**: category type, vectorized operations

## Discussion Points
- When would you consider moving beyond pandas (Polars, Dask)?
- How do you profile pandas operations to find bottlenecks?
- What's your approach to memory management in production?
- How would you handle out-of-memory datasets?

## Success Criteria
- Demonstrate significant performance improvements
- Show memory usage reduction techniques
- Explain when to use different optimization strategies
- Discuss trade-offs between memory and speed
- Production-ready optimized code