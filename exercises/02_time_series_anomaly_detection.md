# Exercise 2: Time Series Anomaly Detection
**Difficulty**: Advanced  
**Time**: 45-60 minutes  
**Skills**: Time series analysis, statistical methods, performance optimization

## Business Context
You're building a monitoring system for an IoT sensor network. Sensors collect temperature, humidity, and pressure readings every minute. The system needs to detect anomalous sensor behavior for predictive maintenance.

## The Challenge
Given sensor data from 100+ devices over 3 months, identify:
1. **Sensor malfunctions** (readings outside normal ranges)
2. **Pattern anomalies** (unusual temporal patterns)
3. **Drift detection** (gradual sensor degradation)
4. **Correlation anomalies** (sensors that don't correlate as expected)

## Data Description

### Sensor Readings Table
```python
# sensor_readings.csv - 500k+ rows
columns = ['sensor_id', 'timestamp', 'temperature', 'humidity', 
           'pressure', 'battery_level', 'location_id']
```

### Sensor Metadata Table
```python
# sensor_metadata.csv
columns = ['sensor_id', 'sensor_type', 'installation_date', 
           'location_id', 'expected_temp_range', 'calibration_date']
```

## Requirements

### Part 1: Data Exploration & Cleaning (15 min)
1. **Time series validation**:
   - Check for missing timestamps
   - Identify irregular sampling intervals
   - Handle timezone issues

2. **Data quality assessment**:
   - Detect and interpolate missing sensor readings
   - Identify sensors with consistent failures
   - Validate reading ranges against sensor specifications

3. **Memory optimization**:
   - Optimize dtypes for large time series
   - Implement efficient resampling strategies

### Part 2: Anomaly Detection (25 min)
1. **Statistical anomalies**:
   - Z-score based outlier detection
   - IQR method for each sensor type
   - Seasonal decomposition anomalies

2. **Pattern-based detection**:
   - Detect sudden spikes/drops
   - Identify stuck sensor readings (same value repeated)
   - Find sensors with abnormal daily patterns

3. **Multi-sensor correlations**:
   - Detect sensors that don't correlate with neighbors
   - Identify location-based anomalies
   - Cross-validate readings between sensor types

### Part 3: Advanced Analysis (15 min)
1. **Drift detection**:
   - Implement change point detection
   - Calculate sensor degradation rates
   - Predict maintenance windows

2. **Performance optimization**:
   - Vectorized operations for large datasets
   - Efficient groupby operations
   - Memory-efficient sliding window calculations

## Key Techniques to Demonstrate
- **Rolling statistics**: Moving averages, standard deviations
- **Resampling**: Upsampling/downsampling time series
- **Groupby optimization**: Using transform() vs apply()
- **Memory management**: Chunked processing for large datasets
- **Statistical methods**: Z-scores, percentiles, correlation matrices

## Discussion Points
- How would you handle sensors with different sampling rates?
- What's your approach to real-time anomaly detection?
- How would you scale this to millions of sensors?
- What additional features would improve detection accuracy?

## Success Criteria
- Efficient handling of large time series data
- Multiple anomaly detection approaches
- Clear explanation of statistical methods
- Production-ready code with error handling
- Thoughtful discussion of scalability challenges