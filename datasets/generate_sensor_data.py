"""
Generate realistic IoT sensor time series data for anomaly detection exercise
Includes various types of sensor failures and anomalies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seeds
np.random.seed(42)
random.seed(42)

def generate_sensor_metadata(n_sensors=100):
    """Generate sensor metadata"""
    
    sensor_types = ['indoor', 'outdoor', 'industrial', 'greenhouse']
    locations = [f"LOC_{i:03d}" for i in range(1, 21)]  # 20 locations
    
    sensors = []
    
    for i in range(n_sensors):
        sensor_id = f"SENSOR_{i+1:03d}"
        sensor_type = np.random.choice(sensor_types)
        location_id = np.random.choice(locations)
        
        # Installation date (last 2 years)
        install_date = datetime.now() - timedelta(days=np.random.randint(30, 730))
        
        # Calibration date (within last 6 months)
        calibration_date = datetime.now() - timedelta(days=np.random.randint(0, 180))
        
        # Expected ranges based on sensor type
        if sensor_type == 'indoor':
            temp_range = (18, 28)
            humidity_range = (30, 70)
            pressure_range = (1010, 1020)
        elif sensor_type == 'outdoor':
            temp_range = (-10, 40)
            humidity_range = (20, 90)
            pressure_range = (990, 1030)
        elif sensor_type == 'industrial':
            temp_range = (15, 35)
            humidity_range = (20, 60)
            pressure_range = (1000, 1025)
        else:  # greenhouse
            temp_range = (20, 35)
            humidity_range = (60, 85)
            pressure_range = (1005, 1020)
        
        sensors.append({
            'sensor_id': sensor_id,
            'sensor_type': sensor_type,
            'installation_date': install_date.strftime('%Y-%m-%d'),
            'location_id': location_id,
            'expected_temp_min': temp_range[0],
            'expected_temp_max': temp_range[1],
            'expected_humidity_min': humidity_range[0],
            'expected_humidity_max': humidity_range[1],
            'expected_pressure_min': pressure_range[0],
            'expected_pressure_max': pressure_range[1],
            'calibration_date': calibration_date.strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(sensors)

def generate_normal_readings(sensor_info, start_date, end_date):
    """Generate normal sensor readings for a sensor"""
    
    readings = []
    current_time = start_date
    
    # Base values for this sensor
    base_temp = np.random.uniform(sensor_info['expected_temp_min'], sensor_info['expected_temp_max'])
    base_humidity = np.random.uniform(sensor_info['expected_humidity_min'], sensor_info['expected_humidity_max'])
    base_pressure = np.random.uniform(sensor_info['expected_pressure_min'], sensor_info['expected_pressure_max'])
    
    # Daily patterns (temperature varies more during day)
    while current_time <= end_date:
        
        # Add daily pattern (sine wave)
        hour_factor = np.sin(2 * np.pi * current_time.hour / 24)
        
        # Temperature: higher during day, lower at night
        temp_daily_var = hour_factor * 3 if sensor_info['sensor_type'] == 'outdoor' else hour_factor * 1
        temperature = base_temp + temp_daily_var + np.random.normal(0, 0.5)
        
        # Humidity: inversely related to temperature
        humidity_daily_var = -hour_factor * 5 if sensor_info['sensor_type'] == 'outdoor' else -hour_factor * 2
        humidity = base_humidity + humidity_daily_var + np.random.normal(0, 2)
        
        # Pressure: more stable, slight variations
        pressure = base_pressure + np.random.normal(0, 1)
        
        # Battery level (slowly decreasing)
        days_since_install = (current_time - datetime.strptime(sensor_info['installation_date'], '%Y-%m-%d')).days
        battery_level = max(10, 100 - (days_since_install * 0.05) + np.random.normal(0, 2))
        
        readings.append({
            'sensor_id': sensor_info['sensor_id'],
            'timestamp': current_time,
            'temperature': round(temperature, 2),
            'humidity': round(max(0, min(100, humidity)), 2),
            'pressure': round(pressure, 2),
            'battery_level': round(max(0, min(100, battery_level)), 2),
            'location_id': sensor_info['location_id']
        })
        
        current_time += timedelta(minutes=1)
    
    return readings

def inject_anomalies(readings, sensor_info):
    """Inject various types of anomalies into sensor readings"""
    
    readings_df = pd.DataFrame(readings)
    anomaly_types = []
    
    # 1. Sensor stuck (same value repeated)
    if np.random.random() < 0.1:  # 10% chance
        start_idx = np.random.randint(100, len(readings_df) - 100)
        duration = np.random.randint(10, 60)  # 10-60 minutes stuck
        
        stuck_value = readings_df.iloc[start_idx]['temperature']
        readings_df.loc[start_idx:start_idx+duration, 'temperature'] = stuck_value
        anomaly_types.append(f"stuck_temperature_{start_idx}_{duration}")
    
    # 2. Sudden spike/drop
    if np.random.random() < 0.15:  # 15% chance
        spike_idx = np.random.randint(0, len(readings_df))
        spike_magnitude = np.random.choice([-1, 1]) * np.random.uniform(10, 20)
        
        readings_df.loc[spike_idx, 'temperature'] += spike_magnitude
        anomaly_types.append(f"temperature_spike_{spike_idx}_{spike_magnitude:.1f}")
    
    # 3. Gradual drift (sensor degradation)
    if np.random.random() < 0.08:  # 8% chance
        start_idx = np.random.randint(0, len(readings_df) // 2)
        drift_rate = np.random.uniform(-0.001, 0.001)  # per minute
        
        for i in range(start_idx, len(readings_df)):
            readings_df.loc[i, 'temperature'] += drift_rate * (i - start_idx)
        
        anomaly_types.append(f"drift_{start_idx}_{drift_rate:.4f}")
    
    # 4. Missing data periods
    if np.random.random() < 0.12:  # 12% chance
        missing_start = np.random.randint(0, len(readings_df) - 30)
        missing_duration = np.random.randint(5, 30)
        
        readings_df.loc[missing_start:missing_start+missing_duration, ['temperature', 'humidity', 'pressure']] = np.nan
        anomaly_types.append(f"missing_data_{missing_start}_{missing_duration}")
    
    # 5. Out of range values (sensor malfunction)
    if np.random.random() < 0.06:  # 6% chance
        out_of_range_idx = np.random.randint(0, len(readings_df))
        
        if sensor_info['sensor_type'] == 'indoor':
            readings_df.loc[out_of_range_idx, 'temperature'] = np.random.choice([-5, 50])  # Impossible indoor temps
        else:
            readings_df.loc[out_of_range_idx, 'humidity'] = np.random.choice([-10, 110])  # Impossible humidity
        
        anomaly_types.append(f"out_of_range_{out_of_range_idx}")
    
    # 6. Battery failure (sudden drop to 0)
    if np.random.random() < 0.05:  # 5% chance
        battery_fail_idx = np.random.randint(len(readings_df) // 2, len(readings_df))
        readings_df.loc[battery_fail_idx:, 'battery_level'] = 0
        anomaly_types.append(f"battery_failure_{battery_fail_idx}")
    
    # 7. Irregular timestamp intervals
    if np.random.random() < 0.08:  # 8% chance
        irregular_start = np.random.randint(0, len(readings_df) - 100)
        # Skip some readings to create gaps
        skip_indices = np.random.choice(range(irregular_start, irregular_start + 100), 
                                       size=np.random.randint(5, 20), replace=False)
        readings_df = readings_df.drop(skip_indices).reset_index(drop=True)
        anomaly_types.append(f"irregular_intervals_{irregular_start}")
    
    return readings_df, anomaly_types

def generate_sensor_readings(metadata_df, days=90):
    """Generate readings for all sensors"""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    all_readings = []
    anomaly_log = []
    
    print(f"Generating sensor readings from {start_date} to {end_date}")
    
    for idx, sensor_info in metadata_df.iterrows():
        print(f"Processing sensor {sensor_info['sensor_id']} ({idx+1}/{len(metadata_df)})")
        
        # Generate normal readings
        readings = generate_normal_readings(sensor_info, start_date, end_date)
        
        # Inject anomalies
        readings_df, anomalies = inject_anomalies(readings, sensor_info)
        
        all_readings.append(readings_df)
        
        if anomalies:
            for anomaly in anomalies:
                anomaly_log.append({
                    'sensor_id': sensor_info['sensor_id'],
                    'anomaly_type': anomaly,
                    'sensor_type': sensor_info['sensor_type']
                })
    
    # Combine all readings
    combined_readings = pd.concat(all_readings, ignore_index=True)
    
    # Sort by timestamp
    combined_readings = combined_readings.sort_values(['sensor_id', 'timestamp']).reset_index(drop=True)
    
    return combined_readings, pd.DataFrame(anomaly_log)

def main():
    """Generate complete sensor dataset"""
    
    print("Generating IoT sensor dataset...")
    
    # Create exercise directory if it doesn't exist
    import os
    os.makedirs('exercise_02', exist_ok=True)
    
    # Generate metadata
    print("1. Generating sensor metadata...")
    metadata = generate_sensor_metadata(100)
    metadata.to_csv('exercise_02/sensor_metadata.csv', index=False)
    print(f"Generated metadata for {len(metadata)} sensors")
    
    # Generate readings
    print("2. Generating sensor readings...")
    readings, anomaly_log = generate_sensor_readings(metadata, days=90)
    
    # Save readings
    readings.to_csv('exercise_02/sensor_readings.csv', index=False)
    print(f"Generated {len(readings):,} sensor readings")
    
    # Save anomaly log (for reference - not given to candidates)
    anomaly_log.to_csv('exercise_02/anomaly_log.csv', index=False)
    print(f"Injected {len(anomaly_log)} anomalies")
    
    print("\nDataset summary:")
    print(f"Sensors: {readings['sensor_id'].nunique()}")
    print(f"Locations: {readings['location_id'].nunique()}")
    print(f"Time range: {readings['timestamp'].min()} to {readings['timestamp'].max()}")
    print(f"Total readings: {len(readings):,}")
    
    print("\nData quality issues:")
    print(f"Missing temperature: {readings['temperature'].isna().sum():,}")
    print(f"Missing humidity: {readings['humidity'].isna().sum():,}")
    print(f"Missing pressure: {readings['pressure'].isna().sum():,}")
    print(f"Zero battery levels: {(readings['battery_level'] == 0).sum():,}")
    
    print("\nAnomaly types injected:")
    print(anomaly_log['anomaly_type'].str.split('_').str[0].value_counts())
    
    print("\nSample readings:")
    print(readings.head())
    
    print("\nSample metadata:")
    print(metadata.head())

if __name__ == "__main__":
    main()