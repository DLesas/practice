"""
Solution: Time Series Anomaly Detection Exercise
Demonstrates comprehensive approach to IoT sensor anomaly detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SensorAnomalyDetector:
    """
    Comprehensive anomaly detection system for IoT sensor data
    """
    
    def __init__(self):
        self.anomalies = []
        self.sensor_stats = {}
        self.thresholds = {}
    
    def load_and_validate_data(self, readings_path, metadata_path):
        """Load and perform initial validation of sensor data"""
        
        print("=== LOADING AND VALIDATING SENSOR DATA ===")
        
        # Load with optimized dtypes
        readings = pd.read_csv(readings_path,
                             dtype={'sensor_id': 'category',
                                    'location_id': 'category'},
                             parse_dates=['timestamp'])
        
        metadata = pd.read_csv(metadata_path,
                             dtype={'sensor_id': 'category',
                                    'sensor_type': 'category',
                                    'location_id': 'category'},
                             parse_dates=['installation_date', 'calibration_date'])
        
        print(f"Loaded {len(readings):,} sensor readings from {readings['sensor_id'].nunique()} sensors")
        print(f"Time range: {readings['timestamp'].min()} to {readings['timestamp'].max()}")
        print(f"Memory usage: {readings.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Validate time series properties
        self._validate_time_series(readings)
        
        # Merge with metadata
        enriched = readings.merge(metadata[['sensor_id', 'sensor_type', 'expected_temp_min', 
                                          'expected_temp_max', 'expected_humidity_min', 
                                          'expected_humidity_max']], 
                                on='sensor_id', how='left')
        
        return enriched, metadata
    
    def _validate_time_series(self, df):
        """Validate time series data quality"""
        
        print("\n--- Time Series Validation ---")
        
        # Check for missing timestamps
        missing_timestamps = df['timestamp'].isnull().sum()
        print(f"Missing timestamps: {missing_timestamps:,}")
        
        # Check sampling intervals
        df_sorted = df.sort_values(['sensor_id', 'timestamp'])
        df_sorted['time_diff'] = df_sorted.groupby('sensor_id')['timestamp'].diff()
        
        # Expected interval is 1 minute
        expected_interval = pd.Timedelta(minutes=1)
        irregular_intervals = (df_sorted['time_diff'] != expected_interval).sum()
        print(f"Irregular sampling intervals: {irregular_intervals:,} ({irregular_intervals/len(df)*100:.2f}%)")
        
        # Check for large gaps
        large_gaps = (df_sorted['time_diff'] > pd.Timedelta(minutes=10)).sum()
        print(f"Large gaps (>10 min): {large_gaps:,}")
        
        # Sensors with data issues
        sensors_with_issues = df_sorted[df_sorted['time_diff'] > pd.Timedelta(minutes=5)]['sensor_id'].nunique()
        print(f"Sensors with timing issues: {sensors_with_issues}")
    
    def detect_statistical_anomalies(self, df, z_threshold=3, iqr_multiplier=1.5):
        """Detect anomalies using statistical methods"""
        
        print("\n=== STATISTICAL ANOMALY DETECTION ===")
        
        anomaly_results = []
        
        for sensor_type in df['sensor_type'].unique():
            sensor_data = df[df['sensor_type'] == sensor_type].copy()
            
            print(f"\nAnalyzing {sensor_type} sensors ({len(sensor_data):,} readings)...")
            
            # Z-score based detection
            for metric in ['temperature', 'humidity', 'pressure']:
                if metric in sensor_data.columns:
                    # Calculate z-scores
                    z_scores = np.abs(stats.zscore(sensor_data[metric].dropna()))
                    z_anomalies = sensor_data[z_scores > z_threshold].copy()
                    z_anomalies['anomaly_type'] = f'{metric}_zscore'
                    z_anomalies['anomaly_score'] = z_scores[z_scores > z_threshold]
                    
                    # IQR based detection
                    Q1 = sensor_data[metric].quantile(0.25)
                    Q3 = sensor_data[metric].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - iqr_multiplier * IQR
                    upper_bound = Q3 + iqr_multiplier * IQR
                    
                    iqr_anomalies = sensor_data[
                        (sensor_data[metric] < lower_bound) | 
                        (sensor_data[metric] > upper_bound)
                    ].copy()
                    iqr_anomalies['anomaly_type'] = f'{metric}_iqr'
                    iqr_anomalies['anomaly_score'] = np.where(
                        iqr_anomalies[metric] < lower_bound,
                        lower_bound - iqr_anomalies[metric],
                        iqr_anomalies[metric] - upper_bound
                    )
                    
                    anomaly_results.extend([z_anomalies, iqr_anomalies])
                    
                    print(f"  {metric}: {len(z_anomalies)} Z-score anomalies, {len(iqr_anomalies)} IQR anomalies")
        
        if anomaly_results:
            all_anomalies = pd.concat(anomaly_results, ignore_index=True)
            self.anomalies.append(all_anomalies)
            return all_anomalies
        else:
            return pd.DataFrame()
    
    def detect_pattern_anomalies(self, df):
        """Detect pattern-based anomalies"""
        
        print("\n=== PATTERN ANOMALY DETECTION ===")
        
        pattern_anomalies = []
        
        for sensor_id in df['sensor_id'].unique():
            sensor_data = df[df['sensor_id'] == sensor_id].sort_values('timestamp').copy()
            
            if len(sensor_data) < 10:  # Skip sensors with too little data
                continue
            
            # 1. Stuck sensor detection (same value repeated)
            for metric in ['temperature', 'humidity', 'pressure']:
                if metric in sensor_data.columns:
                    # Rolling window to find consecutive identical values
                    sensor_data[f'{metric}_stuck'] = (
                        sensor_data[metric].rolling(window=10, min_periods=5)
                        .apply(lambda x: len(x.unique()) == 1)
                    )
                    
                    stuck_readings = sensor_data[sensor_data[f'{metric}_stuck'] == 1].copy()
                    if len(stuck_readings) > 0:
                        stuck_readings['anomaly_type'] = f'{metric}_stuck'
                        stuck_readings['anomaly_score'] = 1.0
                        pattern_anomalies.append(stuck_readings)
            
            # 2. Sudden spikes/drops
            for metric in ['temperature', 'humidity', 'pressure']:
                if metric in sensor_data.columns:
                    # Calculate rate of change
                    sensor_data[f'{metric}_diff'] = sensor_data[metric].diff().abs()
                    
                    # Detect sudden changes (>3 standard deviations from mean change)
                    mean_change = sensor_data[f'{metric}_diff'].mean()
                    std_change = sensor_data[f'{metric}_diff'].std()
                    threshold = mean_change + 3 * std_change
                    
                    spike_readings = sensor_data[sensor_data[f'{metric}_diff'] > threshold].copy()
                    if len(spike_readings) > 0:
                        spike_readings['anomaly_type'] = f'{metric}_spike'
                        spike_readings['anomaly_score'] = spike_readings[f'{metric}_diff'] / threshold
                        pattern_anomalies.append(spike_readings)
            
            # 3. Out of expected range
            sensor_metadata = df[df['sensor_id'] == sensor_id].iloc[0]
            
            # Temperature range check
            if 'temperature' in sensor_data.columns:
                out_of_range = sensor_data[
                    (sensor_data['temperature'] < sensor_metadata['expected_temp_min']) |
                    (sensor_data['temperature'] > sensor_metadata['expected_temp_max'])
                ].copy()
                
                if len(out_of_range) > 0:
                    out_of_range['anomaly_type'] = 'temperature_out_of_range'
                    out_of_range['anomaly_score'] = 1.0
                    pattern_anomalies.append(out_of_range)
            
            # Humidity range check
            if 'humidity' in sensor_data.columns:
                out_of_range = sensor_data[
                    (sensor_data['humidity'] < sensor_metadata['expected_humidity_min']) |
                    (sensor_data['humidity'] > sensor_metadata['expected_humidity_max'])
                ].copy()
                
                if len(out_of_range) > 0:
                    out_of_range['anomaly_type'] = 'humidity_out_of_range'
                    out_of_range['anomaly_score'] = 1.0
                    pattern_anomalies.append(out_of_range)
        
        if pattern_anomalies:
            all_pattern_anomalies = pd.concat(pattern_anomalies, ignore_index=True)
            self.anomalies.append(all_pattern_anomalies)
            print(f"Detected {len(all_pattern_anomalies):,} pattern anomalies")
            return all_pattern_anomalies
        else:
            return pd.DataFrame()
    
    def detect_correlation_anomalies(self, df):
        """Detect correlation-based anomalies"""
        
        print("\n=== CORRELATION ANOMALY DETECTION ===")
        
        correlation_anomalies = []
        
        # Group by location to find sensors that should correlate
        for location in df['location_id'].unique():
            location_data = df[df['location_id'] == location]
            
            if location_data['sensor_id'].nunique() < 2:  # Need at least 2 sensors
                continue
            
            # Create pivot table for correlation analysis
            pivot_temp = location_data.pivot_table(
                index='timestamp', 
                columns='sensor_id', 
                values='temperature',
                aggfunc='mean'
            )
            
            if pivot_temp.shape[1] >= 2:  # At least 2 sensors
                # Calculate correlation matrix
                corr_matrix = pivot_temp.corr()
                
                # Find sensors with low correlation to others
                for sensor_id in corr_matrix.index:
                    other_correlations = corr_matrix.loc[sensor_id, corr_matrix.index != sensor_id]
                    avg_correlation = other_correlations.mean()
                    
                    # If average correlation is very low, this sensor might be anomalous
                    if avg_correlation < 0.3:  # Threshold for low correlation
                        anomalous_readings = location_data[location_data['sensor_id'] == sensor_id].copy()
                        anomalous_readings['anomaly_type'] = 'low_correlation'
                        anomalous_readings['anomaly_score'] = 1 - avg_correlation
                        correlation_anomalies.append(anomalous_readings)
        
        if correlation_anomalies:
            all_correlation_anomalies = pd.concat(correlation_anomalies, ignore_index=True)
            self.anomalies.append(all_correlation_anomalies)
            print(f"Detected {len(all_correlation_anomalies):,} correlation anomalies")
            return all_correlation_anomalies
        else:
            return pd.DataFrame()
    
    def detect_drift(self, df, window_size='7D'):
        """Detect sensor drift over time"""
        
        print("\n=== DRIFT DETECTION ===")
        
        drift_results = []
        
        for sensor_id in df['sensor_id'].unique():
            sensor_data = df[df['sensor_id'] == sensor_id].set_index('timestamp').sort_index()
            
            if len(sensor_data) < 100:  # Skip sensors with too little data
                continue
            
            for metric in ['temperature', 'humidity', 'pressure']:
                if metric in sensor_data.columns:
                    # Calculate rolling mean
                    rolling_mean = sensor_data[metric].rolling(window=window_size).mean()
                    
                    # Calculate trend (linear regression slope over time)
                    timestamps_numeric = pd.to_numeric(sensor_data.index)
                    
                    if len(rolling_mean.dropna()) > 10:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            timestamps_numeric, sensor_data[metric].fillna(method='ffill')
                        )
                        
                        # Significant drift if slope is large and correlation is strong
                        if abs(slope) > 0.001 and abs(r_value) > 0.3:  # Thresholds for drift
                            drift_info = {
                                'sensor_id': sensor_id,
                                'metric': metric,
                                'drift_rate': slope,
                                'correlation': r_value,
                                'p_value': p_value,
                                'drift_severity': 'High' if abs(slope) > 0.01 else 'Medium'
                            }
                            drift_results.append(drift_info)
        
        drift_df = pd.DataFrame(drift_results)
        
        if len(drift_df) > 0:
            print(f"Detected drift in {len(drift_df)} sensor-metric combinations")
            print(drift_df.groupby('drift_severity').size())
        else:
            print("No significant drift detected")
        
        return drift_df
    
    def optimize_processing(self, df):
        """Demonstrate performance optimization techniques"""
        
        print("\n=== PERFORMANCE OPTIMIZATION ===")
        
        import time
        
        # 1. Memory optimization
        print("1. Memory Optimization:")
        memory_before = df.memory_usage(deep=True).sum() / 1024**2
        
        # Optimize dtypes
        df_optimized = df.copy()
        
        # Convert to categorical
        for col in ['sensor_id', 'location_id', 'sensor_type']:
            if col in df_optimized.columns:
                df_optimized[col] = df_optimized[col].astype('category')
        
        # Downcast numeric types
        for col in ['temperature', 'humidity', 'pressure', 'battery_level']:
            if col in df_optimized.columns:
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        memory_after = df_optimized.memory_usage(deep=True).sum() / 1024**2
        print(f"  Memory usage: {memory_before:.1f} MB → {memory_after:.1f} MB")
        print(f"  Memory savings: {(memory_before - memory_after) / memory_before * 100:.1f}%")
        
        # 2. Vectorized operations vs loops
        print("\n2. Vectorization Performance:")
        
        # Slow way (loops)
        start_time = time.time()
        anomaly_flags_slow = []
        for idx, row in df.head(1000).iterrows():
            if row['temperature'] > 30:
                anomaly_flags_slow.append(1)
            else:
                anomaly_flags_slow.append(0)
        slow_time = time.time() - start_time
        
        # Fast way (vectorized)
        start_time = time.time()
        anomaly_flags_fast = (df.head(1000)['temperature'] > 30).astype(int)
        fast_time = time.time() - start_time
        
        print(f"  Loop method: {slow_time:.4f} seconds")
        print(f"  Vectorized method: {fast_time:.4f} seconds")
        print(f"  Speedup: {slow_time / fast_time:.1f}x faster")
        
        # 3. Efficient groupby operations
        print("\n3. GroupBy Optimization:")
        
        start_time = time.time()
        # Use transform for element-wise operations
        df_sample = df.head(10000)
        df_sample['temp_zscore'] = df_sample.groupby('sensor_id')['temperature'].transform(
            lambda x: np.abs(stats.zscore(x.dropna())) if len(x.dropna()) > 1 else 0
        )
        groupby_time = time.time() - start_time
        print(f"  GroupBy transform: {groupby_time:.4f} seconds")
        
        return df_optimized
    
    def generate_anomaly_report(self):
        """Generate comprehensive anomaly detection report"""
        
        print("\n" + "="*60)
        print("ANOMALY DETECTION REPORT")
        print("="*60)
        
        if not self.anomalies:
            print("No anomalies detected.")
            return
        
        # Combine all anomalies
        all_anomalies = pd.concat(self.anomalies, ignore_index=True)
        
        # Summary statistics
        print(f"Total anomalies detected: {len(all_anomalies):,}")
        print(f"Sensors affected: {all_anomalies['sensor_id'].nunique()}")
        print(f"Locations affected: {all_anomalies['location_id'].nunique()}")
        
        # Anomaly types
        print(f"\nAnomaly breakdown by type:")
        anomaly_counts = all_anomalies['anomaly_type'].value_counts()
        for anomaly_type, count in anomaly_counts.items():
            print(f"  {anomaly_type}: {count:,}")
        
        # Most problematic sensors
        print(f"\nMost problematic sensors:")
        problematic_sensors = all_anomalies['sensor_id'].value_counts().head()
        for sensor_id, count in problematic_sensors.items():
            print(f"  {sensor_id}: {count:,} anomalies")
        
        # Recommendations
        print(f"\nRecommendations:")
        if 'stuck' in ' '.join(anomaly_counts.index):
            print("• Several sensors showing stuck readings - check for mechanical issues")
        if 'out_of_range' in ' '.join(anomaly_counts.index):
            print("• Sensors reading outside expected ranges - verify calibration")
        if 'spike' in ' '.join(anomaly_counts.index):
            print("• Sudden spikes detected - investigate environmental factors")
        if len(problematic_sensors) > 0:
            print(f"• Priority maintenance needed for sensors: {', '.join(problematic_sensors.head(3).index)}")
        
        return all_anomalies

def run_complete_anomaly_detection():
    """Run the complete anomaly detection pipeline"""
    
    print("=== IoT SENSOR ANOMALY DETECTION PIPELINE ===\n")
    
    # Initialize detector
    detector = SensorAnomalyDetector()
    
    # Load and validate data
    try:
        df, metadata = detector.load_and_validate_data(
            'datasets/exercise_02/sensor_readings.csv',
            'datasets/exercise_02/sensor_metadata.csv'
        )
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please run the sensor data generation script first.")
        return
    
    # Sample data for faster processing in demo
    print(f"\nSampling data for demonstration (using 50,000 rows from {len(df):,} total)")
    df_sample = df.sample(n=min(50000, len(df)), random_state=42).copy()
    
    # Optimize for performance
    df_optimized = detector.optimize_processing(df_sample)
    
    # Run anomaly detection methods
    statistical_anomalies = detector.detect_statistical_anomalies(df_optimized)
    pattern_anomalies = detector.detect_pattern_anomalies(df_optimized)
    correlation_anomalies = detector.detect_correlation_anomalies(df_optimized)
    
    # Drift detection
    drift_results = detector.detect_drift(df_optimized)
    
    # Generate report
    final_report = detector.generate_anomaly_report()
    
    print(f"\n=== PRODUCTION CONSIDERATIONS ===")
    print("For production deployment:")
    print("• Implement real-time anomaly detection with streaming data")
    print("• Set up alerting thresholds based on anomaly severity")
    print("• Create automated maintenance workflows")
    print("• Monitor detector performance and retrain models")
    print("• Implement data quality checks before anomaly detection")
    
    return detector, final_report, drift_results

if __name__ == "__main__":
    # Run the complete pipeline
    detector, anomalies, drift = run_complete_anomaly_detection()