"""
Generate A/B test results data for ML model evaluation
Simulates realistic test results with treatment effects
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seeds
np.random.seed(42)
random.seed(42)

def generate_ab_test_data(n_customers=10000):
    """Generate A/B test results data"""
    
    print("Generating A/B test results data...")
    
    # Customer segments
    segments = ['high_value', 'medium_value', 'low_value', 'new_customer']
    segment_weights = [0.2, 0.3, 0.3, 0.2]
    
    customers = []
    
    for i in range(n_customers):
        customer_id = f"CUST_{i+1:06d}"
        
        # Stratified randomization by segment
        segment = np.random.choice(segments, p=segment_weights)
        
        # Balanced assignment to groups within each segment
        group = 'treatment' if i % 2 == 0 else 'control'
        
        # Generate churn probability based on segment
        if segment == 'high_value':
            base_churn_prob = np.random.beta(2, 8)  # Lower churn for high value
        elif segment == 'medium_value':
            base_churn_prob = np.random.beta(3, 7)
        elif segment == 'low_value':
            base_churn_prob = np.random.beta(5, 5)  # Higher churn for low value
        else:  # new_customer
            base_churn_prob = np.random.beta(4, 6)
        
        # ML model predicted churn probability (with some noise)
        predicted_churn_prob = base_churn_prob + np.random.normal(0, 0.1)
        predicted_churn_prob = np.clip(predicted_churn_prob, 0, 1)
        
        # Intervention logic for treatment group
        intervention_sent = False
        intervention_cost = 0
        intervention_type = None
        
        if group == 'treatment':
            # Send intervention if predicted churn prob > threshold
            if predicted_churn_prob > 0.6:  # High risk threshold
                intervention_sent = True
                intervention_type = 'high_value_offer'
                intervention_cost = 25
            elif predicted_churn_prob > 0.4:  # Medium risk threshold
                intervention_sent = True
                intervention_type = 'discount_coupon'
                intervention_cost = 10
            elif predicted_churn_prob > 0.2:  # Low risk threshold
                intervention_sent = True
                intervention_type = 'engagement_email'
                intervention_cost = 2
        
        # Calculate actual retention (influenced by intervention)
        if intervention_sent:
            # Interventions reduce churn probability
            if intervention_type == 'high_value_offer':
                churn_reduction = np.random.uniform(0.15, 0.25)  # 15-25% reduction
            elif intervention_type == 'discount_coupon':
                churn_reduction = np.random.uniform(0.08, 0.15)  # 8-15% reduction
            else:  # engagement_email
                churn_reduction = np.random.uniform(0.03, 0.08)  # 3-8% reduction
            
            actual_churn_prob = max(0, base_churn_prob - churn_reduction)
        else:
            actual_churn_prob = base_churn_prob
        
        # Determine if customer was retained (inverse of churn)
        retained = np.random.random() > actual_churn_prob
        
        # Generate revenue based on segment and retention
        if segment == 'high_value':
            base_revenue_2w = np.random.gamma(3, 50)  # Higher revenue
            base_revenue_4w = base_revenue_2w * np.random.uniform(1.8, 2.2)
        elif segment == 'medium_value':
            base_revenue_2w = np.random.gamma(2, 25)
            base_revenue_4w = base_revenue_2w * np.random.uniform(1.8, 2.2)
        elif segment == 'low_value':
            base_revenue_2w = np.random.gamma(1.5, 15)
            base_revenue_4w = base_revenue_2w * np.random.uniform(1.5, 2.0)
        else:  # new_customer
            base_revenue_2w = np.random.gamma(1, 20)
            base_revenue_4w = base_revenue_2w * np.random.uniform(1.3, 1.8)
        
        # Revenue is 0 if customer churned, reduced if not retained
        if not retained:
            revenue_2weeks = 0
            revenue_4weeks = 0
        else:
            revenue_2weeks = base_revenue_2w
            revenue_4weeks = base_revenue_4w
            
            # Intervention recipients might have slightly higher revenue due to engagement
            if intervention_sent:
                revenue_multiplier = np.random.uniform(1.05, 1.15)
                revenue_2weeks *= revenue_multiplier
                revenue_4weeks *= revenue_multiplier
        
        customers.append({
            'customer_id': customer_id,
            'group': group,
            'segment': segment,
            'predicted_churn_prob': round(predicted_churn_prob, 4),
            'intervention_sent': intervention_sent,
            'intervention_type': intervention_type,
            'intervention_cost': intervention_cost,
            'retained': retained,
            'revenue_2weeks': round(revenue_2weeks, 2),
            'revenue_4weeks': round(revenue_4weeks, 2)
        })
    
    return pd.DataFrame(customers)

def generate_intervention_metadata():
    """Generate intervention metadata"""
    
    interventions = [
        {
            'intervention_type': 'high_value_offer',
            'cost_per_customer': 25.00,
            'expected_lift': 0.20,
            'target_churn_threshold': 0.6,
            'description': 'Premium discount offer with free shipping'
        },
        {
            'intervention_type': 'discount_coupon',
            'cost_per_customer': 10.00,
            'expected_lift': 0.12,
            'target_churn_threshold': 0.4,
            'description': '15% discount coupon for next purchase'
        },
        {
            'intervention_type': 'engagement_email',
            'cost_per_customer': 2.00,
            'expected_lift': 0.05,
            'target_churn_threshold': 0.2,
            'description': 'Personalized product recommendation email'
        }
    ]
    
    return pd.DataFrame(interventions)

def main():
    """Generate A/B test datasets"""
    
    print("Generating A/B test datasets...")
    
    # Create exercise directory if it doesn't exist
    import os
    os.makedirs('exercise_05', exist_ok=True)
    
    # Generate main results
    ab_results = generate_ab_test_data(10000)
    ab_results.to_csv('exercise_05/ab_test_results.csv', index=False)
    print(f"Generated A/B test results: {len(ab_results)} customers")
    
    # Generate intervention metadata
    interventions = generate_intervention_metadata()
    interventions.to_csv('exercise_05/interventions.csv', index=False)
    print(f"Generated intervention metadata: {len(interventions)} intervention types")
    
    # Analysis summary
    print("\n=== A/B TEST SUMMARY ===")
    
    # Group balance
    print("\nGroup Balance:")
    print(ab_results.groupby(['group', 'segment']).size().unstack(fill_value=0))
    
    # Retention rates
    print("\nRetention Rates by Group:")
    retention_by_group = ab_results.groupby('group')['retained'].agg(['count', 'sum', 'mean'])
    retention_by_group['retention_rate'] = retention_by_group['mean']
    print(retention_by_group[['count', 'sum', 'retention_rate']])
    
    # Intervention stats
    print(f"\nIntervention Statistics:")
    intervention_stats = ab_results[ab_results['group'] == 'treatment']['intervention_sent'].value_counts()
    print(f"Interventions sent: {intervention_stats.get(True, 0):,}")
    print(f"No intervention: {intervention_stats.get(False, 0):,}")
    
    # Revenue analysis
    print(f"\nRevenue Analysis (2 weeks):")
    revenue_by_group = ab_results.groupby('group')['revenue_2weeks'].agg(['mean', 'sum'])
    print(revenue_by_group)
    
    # Cost analysis
    total_intervention_cost = ab_results['intervention_cost'].sum()
    print(f"\nTotal intervention cost: ${total_intervention_cost:,.2f}")
    
    print(f"\nSample data:")
    print(ab_results.head())

if __name__ == "__main__":
    main()