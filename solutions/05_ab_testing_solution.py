"""
Solution: A/B Testing for ML Models
Demonstrates comprehensive A/B test analysis and statistical evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, ttest_ind
import warnings
warnings.filterwarnings('ignore')

class ABTestAnalyzer:
    """
    Comprehensive A/B test analysis framework for ML model evaluation
    """
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha  # Significance level
        self.results = {}
        self.recommendations = []
    
    def load_and_validate_data(self, results_path, interventions_path):
        """Load and validate A/B test data"""
        
        print("=== LOADING A/B TEST DATA ===")
        
        # Load datasets
        ab_results = pd.read_csv(results_path, 
                               dtype={'customer_id': 'category',
                                      'group': 'category', 
                                      'segment': 'category',
                                      'intervention_type': 'category'})
        
        interventions = pd.read_csv(interventions_path)
        
        print(f"Loaded A/B test results: {len(ab_results):,} customers")
        print(f"Test duration: 2 weeks")
        print(f"Groups: {ab_results['group'].value_counts().to_dict()}")
        
        # Basic validation
        self._validate_experiment_design(ab_results)
        
        return ab_results, interventions
    
    def _validate_experiment_design(self, df):
        """Validate experimental design and randomization"""
        
        print("\n=== EXPERIMENTAL VALIDATION ===")
        
        print("1. Randomization Check:")
        
        # Check group balance overall
        group_balance = df['group'].value_counts()
        balance_ratio = group_balance.min() / group_balance.max()
        print(f"   Overall balance ratio: {balance_ratio:.3f} (should be close to 1.0)")
        
        if balance_ratio < 0.9:
            print("   ⚠️  Groups are somewhat imbalanced")
        else:
            print("   ✅ Groups are well balanced")
        
        # Check balance within segments
        segment_balance = df.groupby(['segment', 'group']).size().unstack(fill_value=0)
        print(f"\n   Balance by segment:")
        print(segment_balance)
        
        # Statistical test for randomization
        chi2, p_value, dof, expected = chi2_contingency(segment_balance)
        print(f"\n   Chi-square test for randomization:")
        print(f"   p-value: {p_value:.4f}")
        
        if p_value > 0.05:
            print("   ✅ Randomization appears successful (p > 0.05)")
        else:
            print("   ⚠️  Potential randomization issue (p < 0.05)")
        
        # Check for sample size adequacy
        min_group_size = group_balance.min()
        print(f"\n2. Sample Size Check:")
        print(f"   Minimum group size: {min_group_size:,}")
        
        if min_group_size >= 1000:
            print("   ✅ Adequate sample size for statistical power")
        else:
            print("   ⚠️  Small sample size may limit statistical power")
    
    def analyze_primary_metrics(self, df):
        """Analyze primary retention metric"""
        
        print("\n=== PRIMARY METRIC ANALYSIS ===")
        
        # Calculate retention rates by group
        retention_stats = df.groupby('group')['retained'].agg(['count', 'sum', 'mean', 'std'])
        retention_stats['retention_rate'] = retention_stats['mean']
        retention_stats['std_error'] = retention_stats['std'] / np.sqrt(retention_stats['count'])
        
        print("Retention Rates by Group:")
        print(retention_stats[['count', 'sum', 'retention_rate', 'std_error']].round(4))
        
        # Calculate effect size
        control_rate = retention_stats.loc['control', 'retention_rate']
        treatment_rate = retention_stats.loc['treatment', 'retention_rate']
        
        absolute_lift = treatment_rate - control_rate
        relative_lift = (treatment_rate - control_rate) / control_rate
        
        print(f"\nEffect Size:")
        print(f"   Absolute lift: {absolute_lift:.4f} ({absolute_lift*100:+.2f} percentage points)")
        print(f"   Relative lift: {relative_lift:.4f} ({relative_lift*100:+.1f}%)")
        
        # Statistical significance test
        control_data = df[df['group'] == 'control']['retained']
        treatment_data = df[df['group'] == 'treatment']['retained']
        
        # Chi-square test for proportions
        contingency_table = pd.crosstab(df['group'], df['retained'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\nStatistical Significance (Chi-square test):")
        print(f"   Chi-square statistic: {chi2:.4f}")
        print(f"   p-value: {p_value:.6f}")
        print(f"   Degrees of freedom: {dof}")
        
        if p_value < self.alpha:
            print(f"   ✅ Statistically significant (p < {self.alpha})")
        else:
            print(f"   ❌ Not statistically significant (p >= {self.alpha})")
        
        # Confidence interval for the difference
        control_successes = retention_stats.loc['control', 'sum']
        control_total = retention_stats.loc['control', 'count']
        treatment_successes = retention_stats.loc['treatment', 'sum']
        treatment_total = retention_stats.loc['treatment', 'count']
        
        ci_lower, ci_upper = self._proportion_difference_ci(
            control_successes, control_total, 
            treatment_successes, treatment_total
        )
        
        print(f"\n95% Confidence Interval for lift:")
        print(f"   [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        if ci_lower > 0:
            print("   ✅ Confident that treatment improves retention")
        elif ci_upper < 0:
            print("   ❌ Confident that treatment hurts retention")
        else:
            print("   ⚠️  Confidence interval includes zero (inconclusive)")
        
        # Store results
        self.results['primary_metric'] = {
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'absolute_lift': absolute_lift,
            'relative_lift': relative_lift,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'statistically_significant': p_value < self.alpha
        }
        
        return retention_stats
    
    def analyze_secondary_metrics(self, df):
        """Analyze secondary business metrics"""
        
        print("\n=== SECONDARY METRICS ANALYSIS ===")
        
        secondary_results = {}
        
        # Revenue analysis
        print("1. Revenue Analysis (2 weeks):")
        revenue_stats = df.groupby('group')['revenue_2weeks'].agg(['count', 'mean', 'std', 'sum'])
        
        control_revenue = revenue_stats.loc['control', 'mean']
        treatment_revenue = revenue_stats.loc['treatment', 'mean']
        revenue_lift = (treatment_revenue - control_revenue) / control_revenue
        
        print(f"   Control avg revenue: ${control_revenue:.2f}")
        print(f"   Treatment avg revenue: ${treatment_revenue:.2f}")
        print(f"   Revenue lift: {revenue_lift*100:+.1f}%")
        
        # T-test for revenue difference
        control_rev_data = df[df['group'] == 'control']['revenue_2weeks']
        treatment_rev_data = df[df['group'] == 'treatment']['revenue_2weeks']
        
        t_stat, p_value_revenue = ttest_ind(treatment_rev_data, control_rev_data)
        print(f"   Revenue difference p-value: {p_value_revenue:.4f}")
        
        secondary_results['revenue'] = {
            'control_mean': control_revenue,
            'treatment_mean': treatment_revenue,
            'lift': revenue_lift,
            'p_value': p_value_revenue
        }
        
        # Cost analysis
        print(f"\n2. Cost Analysis:")
        total_intervention_cost = df['intervention_cost'].sum()
        treatment_customers = (df['group'] == 'treatment').sum()
        avg_cost_per_treatment = total_intervention_cost / treatment_customers
        
        print(f"   Total intervention cost: ${total_intervention_cost:,.2f}")
        print(f"   Average cost per treatment customer: ${avg_cost_per_treatment:.2f}")
        
        # ROI calculation
        treatment_total_revenue = df[df['group'] == 'treatment']['revenue_2weeks'].sum()
        control_total_revenue = df[df['group'] == 'control']['revenue_2weeks'].sum()
        
        # Normalize by group size
        treatment_size = (df['group'] == 'treatment').sum()
        control_size = (df['group'] == 'control').sum()
        
        treatment_revenue_per_customer = treatment_total_revenue / treatment_size
        control_revenue_per_customer = control_total_revenue / control_size
        
        incremental_revenue = (treatment_revenue_per_customer - control_revenue_per_customer) * treatment_size
        roi = (incremental_revenue - total_intervention_cost) / total_intervention_cost
        
        print(f"   Incremental revenue: ${incremental_revenue:,.2f}")
        print(f"   ROI: {roi*100:.1f}%")
        
        secondary_results['cost'] = {
            'total_cost': total_intervention_cost,
            'avg_cost_per_customer': avg_cost_per_treatment,
            'incremental_revenue': incremental_revenue,
            'roi': roi
        }
        
        self.results['secondary_metrics'] = secondary_results
        
        return secondary_results
    
    def segment_analysis(self, df):
        """Perform subgroup analysis by customer segment"""
        
        print("\n=== SEGMENT ANALYSIS ===")
        
        segment_results = {}
        
        for segment in df['segment'].unique():
            segment_data = df[df['segment'] == segment]
            
            if len(segment_data) < 100:  # Skip segments with too little data
                continue
            
            print(f"\n{segment.title()} Segment:")
            
            # Retention analysis by segment
            segment_retention = segment_data.groupby('group')['retained'].agg(['count', 'mean'])
            
            if 'control' in segment_retention.index and 'treatment' in segment_retention.index:
                control_rate = segment_retention.loc['control', 'mean']
                treatment_rate = segment_retention.loc['treatment', 'mean']
                segment_lift = (treatment_rate - control_rate) / control_rate
                
                print(f"   Sample sizes: Control={segment_retention.loc['control', 'count']}, Treatment={segment_retention.loc['treatment', 'count']}")
                print(f"   Control retention: {control_rate:.3f}")
                print(f"   Treatment retention: {treatment_rate:.3f}")
                print(f"   Segment lift: {segment_lift*100:+.1f}%")
                
                # Statistical test for this segment
                segment_contingency = pd.crosstab(segment_data['group'], segment_data['retained'])
                if segment_contingency.shape == (2, 2):  # Ensure 2x2 table
                    try:
                        chi2_seg, p_value_seg, dof_seg, expected_seg = chi2_contingency(segment_contingency)
                        print(f"   Segment p-value: {p_value_seg:.4f}")
                        
                        segment_results[segment] = {
                            'control_rate': control_rate,
                            'treatment_rate': treatment_rate,
                            'lift': segment_lift,
                            'p_value': p_value_seg,
                            'sample_size': len(segment_data)
                        }
                    except:
                        print(f"   Statistical test failed for {segment}")
        
        # Identify best performing segments
        if segment_results:
            best_segment = max(segment_results.keys(), key=lambda x: segment_results[x]['lift'])
            worst_segment = min(segment_results.keys(), key=lambda x: segment_results[x]['lift'])
            
            print(f"\nSegment Performance Summary:")
            print(f"   Best performing: {best_segment} ({segment_results[best_segment]['lift']*100:+.1f}% lift)")
            print(f"   Worst performing: {worst_segment} ({segment_results[worst_segment]['lift']*100:+.1f}% lift)")
        
        self.results['segment_analysis'] = segment_results
        
        return segment_results
    
    def intervention_effectiveness_analysis(self, df):
        """Analyze effectiveness of different intervention types"""
        
        print("\n=== INTERVENTION EFFECTIVENESS ANALYSIS ===")
        
        treatment_data = df[df['group'] == 'treatment'].copy()
        
        # Overall intervention sending rate
        intervention_rate = treatment_data['intervention_sent'].mean()
        print(f"Intervention sending rate: {intervention_rate:.1%}")
        
        # Effectiveness by intervention type
        intervention_analysis = treatment_data[treatment_data['intervention_sent']].groupby('intervention_type').agg({
            'retained': ['count', 'mean'],
            'revenue_2weeks': 'mean',
            'intervention_cost': 'mean'
        }).round(3)
        
        print(f"\nIntervention Effectiveness by Type:")
        print(intervention_analysis)
        
        # Compare intervention recipients vs non-recipients in treatment group
        intervention_comparison = treatment_data.groupby('intervention_sent').agg({
            'retained': ['count', 'mean'],
            'revenue_2weeks': 'mean'
        }).round(3)
        
        print(f"\nIntervention Recipients vs Non-Recipients:")
        print(intervention_comparison)
        
        # ROI by intervention type
        if 'intervention_type' in treatment_data.columns:
            intervention_roi = []
            
            for intervention_type in treatment_data['intervention_type'].dropna().unique():
                type_data = treatment_data[treatment_data['intervention_type'] == intervention_type]
                
                avg_cost = type_data['intervention_cost'].mean()
                avg_revenue = type_data['revenue_2weeks'].mean()
                retention_rate = type_data['retained'].mean()
                
                # Simple ROI calculation
                roi = (avg_revenue - avg_cost) / avg_cost if avg_cost > 0 else 0
                
                intervention_roi.append({
                    'intervention_type': intervention_type,
                    'avg_cost': avg_cost,
                    'avg_revenue': avg_revenue,
                    'retention_rate': retention_rate,
                    'roi': roi
                })
            
            roi_df = pd.DataFrame(intervention_roi)
            print(f"\nROI by Intervention Type:")
            print(roi_df.round(3))
        
        return intervention_analysis
    
    def power_analysis(self, df):
        """Perform power analysis and sample size calculations"""
        
        print("\n=== POWER ANALYSIS ===")
        
        # Current effect size
        control_rate = df[df['group'] == 'control']['retained'].mean()
        treatment_rate = df[df['group'] == 'treatment']['retained'].mean()
        effect_size = treatment_rate - control_rate
        
        print(f"Current Effect Size: {effect_size:.4f}")
        
        # Sample sizes
        n_control = (df['group'] == 'control').sum()
        n_treatment = (df['group'] == 'treatment').sum()
        
        print(f"Sample Sizes: Control={n_control:,}, Treatment={n_treatment:,}")
        
        # Calculate achieved power
        pooled_p = (df['retained'].sum()) / len(df)
        pooled_variance = pooled_p * (1 - pooled_p)
        se_difference = np.sqrt(pooled_variance * (1/n_control + 1/n_treatment))
        
        z_score = abs(effect_size) / se_difference
        achieved_power = 1 - stats.norm.cdf(stats.norm.ppf(1 - self.alpha/2) - z_score)
        
        print(f"Achieved Power: {achieved_power:.3f}")
        
        # Minimum detectable effect
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = stats.norm.ppf(0.8)  # 80% power
        
        mde = (z_alpha + z_beta) * se_difference
        
        print(f"Minimum Detectable Effect (80% power): {mde:.4f}")
        print(f"Minimum Detectable Effect (%): {mde/control_rate*100:.1f}%")
        
        return achieved_power, mde
    
    def _proportion_difference_ci(self, x1, n1, x2, n2, confidence=0.95):
        """Calculate confidence interval for difference in proportions"""
        
        p1 = x1 / n1
        p2 = x2 / n2
        
        se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
        
        alpha = 1 - confidence
        z = stats.norm.ppf(1 - alpha/2)
        
        diff = p2 - p1
        margin_error = z * se
        
        return diff - margin_error, diff + margin_error
    
    def generate_business_recommendations(self):
        """Generate business recommendations based on analysis"""
        
        print("\n" + "="*60)
        print("BUSINESS RECOMMENDATIONS")
        print("="*60)
        
        recommendations = []
        
        # Primary metric recommendation
        primary = self.results.get('primary_metric', {})
        
        if primary.get('statistically_significant', False):
            if primary.get('relative_lift', 0) > 0:
                recommendations.append("✅ LAUNCH: The ML model shows statistically significant improvement in customer retention")
                recommendations.append(f"   Expected retention lift: {primary.get('relative_lift', 0)*100:+.1f}%")
            else:
                recommendations.append("❌ DO NOT LAUNCH: The model significantly hurts retention")
        else:
            recommendations.append("⚠️  INCONCLUSIVE: No statistically significant effect detected")
            recommendations.append("   Consider: Longer test duration, larger sample size, or model improvements")
        
        # ROI recommendation
        secondary = self.results.get('secondary_metrics', {})
        cost_info = secondary.get('cost', {})
        
        if cost_info.get('roi', 0) > 0:
            recommendations.append(f"💰 POSITIVE ROI: {cost_info.get('roi', 0)*100:.1f}% return on intervention investment")
        else:
            recommendations.append("💸 NEGATIVE ROI: Intervention costs exceed incremental revenue")
            recommendations.append("   Consider: Reducing intervention costs or improving targeting")
        
        # Segment recommendations
        segment_results = self.results.get('segment_analysis', {})
        if segment_results:
            best_segments = [seg for seg, data in segment_results.items() 
                           if data.get('lift', 0) > 0 and data.get('p_value', 1) < 0.1]
            
            if best_segments:
                recommendations.append(f"🎯 SEGMENT TARGETING: Focus on high-performing segments: {', '.join(best_segments)}")
            
            poor_segments = [seg for seg, data in segment_results.items() 
                           if data.get('lift', 0) < 0]
            
            if poor_segments:
                recommendations.append(f"⚠️  EXCLUDE SEGMENTS: Consider excluding: {', '.join(poor_segments)}")
        
        # Implementation recommendations
        recommendations.append("\n📋 IMPLEMENTATION RECOMMENDATIONS:")
        recommendations.append("• Set up automated monitoring for model performance drift")
        recommendations.append("• Implement gradual rollout (10% → 50% → 100%) to monitor for issues")
        recommendations.append("• Establish alerting for significant metric changes")
        recommendations.append("• Plan quarterly model retraining and A/B test validation")
        
        # Print all recommendations
        for rec in recommendations:
            print(rec)
        
        self.recommendations = recommendations
        
        return recommendations
    
    def create_visualization_dashboard(self, df):
        """Create comprehensive visualization dashboard"""
        
        print("\n=== CREATING VISUALIZATION DASHBOARD ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('A/B Test Results Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Retention rates by group
        retention_by_group = df.groupby('group')['retained'].mean()
        ax1 = axes[0, 0]
        bars = ax1.bar(retention_by_group.index, retention_by_group.values, 
                      color=['lightcoral', 'lightblue'])
        ax1.set_title('Retention Rate by Group')
        ax1.set_ylabel('Retention Rate')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, retention_by_group.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.1%}', ha='center', va='bottom')
        
        # 2. Revenue by group
        revenue_by_group = df.groupby('group')['revenue_2weeks'].mean()
        ax2 = axes[0, 1]
        bars = ax2.bar(revenue_by_group.index, revenue_by_group.values, 
                      color=['lightcoral', 'lightblue'])
        ax2.set_title('Average Revenue by Group')
        ax2.set_ylabel('Revenue ($)')
        
        for bar, value in zip(bars, revenue_by_group.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'${value:.0f}', ha='center', va='bottom')
        
        # 3. Retention by segment
        ax3 = axes[0, 2]
        segment_retention = df.groupby(['segment', 'group'])['retained'].mean().unstack()
        segment_retention.plot(kind='bar', ax=ax3, color=['lightcoral', 'lightblue'])
        ax3.set_title('Retention Rate by Segment')
        ax3.set_ylabel('Retention Rate')
        ax3.legend(title='Group')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Intervention effectiveness
        ax4 = axes[1, 0]
        treatment_data = df[df['group'] == 'treatment']
        intervention_effect = treatment_data.groupby('intervention_sent')['retained'].mean()
        
        if len(intervention_effect) > 1:
            bars = ax4.bar(['No Intervention', 'Intervention'], intervention_effect.values,
                          color=['lightgray', 'lightgreen'])
            ax4.set_title('Intervention Effect (Treatment Group)')
            ax4.set_ylabel('Retention Rate')
            
            for bar, value in zip(bars, intervention_effect.values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.1%}', ha='center', va='bottom')
        
        # 5. Revenue distribution
        ax5 = axes[1, 1]
        df.boxplot(column='revenue_2weeks', by='group', ax=ax5)
        ax5.set_title('Revenue Distribution by Group')
        ax5.set_ylabel('Revenue ($)')
        
        # 6. Statistical summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create summary text
        primary = self.results.get('primary_metric', {})
        summary_text = f"""
Statistical Summary:

Retention Lift: {primary.get('relative_lift', 0)*100:+.1f}%
P-value: {primary.get('p_value', 'N/A'):.4f}
Significance: {'Yes' if primary.get('statistically_significant', False) else 'No'}

Sample Size: {len(df):,} customers
Test Duration: 2 weeks
        """
        
        ax6.text(0.1, 0.8, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('ab_test_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Dashboard saved as 'ab_test_dashboard.png'")

def run_complete_ab_test_analysis():
    """Run the complete A/B test analysis pipeline"""
    
    print("=== A/B TEST ANALYSIS FOR ML MODEL EVALUATION ===\n")
    
    # Initialize analyzer
    analyzer = ABTestAnalyzer(alpha=0.05)
    
    # Load and validate data
    try:
        ab_results, interventions = analyzer.load_and_validate_data(
            'datasets/exercise_05/ab_test_results.csv',
            'datasets/exercise_05/interventions.csv'
        )
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please run the A/B test data generation script first.")
        return
    
    # Run comprehensive analysis
    retention_stats = analyzer.analyze_primary_metrics(ab_results)
    secondary_results = analyzer.analyze_secondary_metrics(ab_results)
    segment_results = analyzer.segment_analysis(ab_results)
    intervention_analysis = analyzer.intervention_effectiveness_analysis(ab_results)
    power, mde = analyzer.power_analysis(ab_results)
    
    # Generate business recommendations
    recommendations = analyzer.generate_business_recommendations()
    
    # Create visualizations
    analyzer.create_visualization_dashboard(ab_results)
    
    print(f"\n=== NEXT STEPS ===")
    print("Based on this analysis:")
    print("1. Review business recommendations above")
    print("2. Consider longer-term metrics (customer lifetime value)")
    print("3. Plan for model monitoring and performance tracking")
    print("4. Design follow-up experiments for model improvements")
    print("5. Implement gradual rollout strategy if launching")
    
    return analyzer, ab_results

if __name__ == "__main__":
    # Run the complete A/B test analysis
    analyzer, results = run_complete_ab_test_analysis()