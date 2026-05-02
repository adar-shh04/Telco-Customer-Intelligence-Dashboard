"""
A/B Test Simulation Engine

Simulates running experiments on retention strategies using real customer data.
Demonstrates statistical rigor in decision-making:

1. Splits customers into Treatment vs Control groups
2. Simulates strategy effects with realistic response curves
3. Performs statistical significance testing (chi-squared, t-test)
4. Calculates power analysis and confidence intervals
5. Provides go/no-go recommendations

Shows interviewers you understand experimentation-driven product decisions.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ABTestSimulator:
    """Simulates and analyzes A/B tests on retention strategies."""
    
    # Pre-defined experiment templates with realistic parameters
    EXPERIMENTS = {
        'discount_10pct': {
            'name': '10% Discount on Monthly Bill',
            'description': 'Offer 10% discount to at-risk customers for 3 months',
            'treatment_effect': 0.08,  # 8% churn reduction
            'cost_per_customer': 6.50,
            'icon': '🏷️'
        },
        'premium_support': {
            'name': 'Free Premium Support Upgrade',
            'description': 'Upgrade to dedicated support for 6 months',
            'treatment_effect': 0.05,
            'cost_per_customer': 4.00,
            'icon': '🛠️'
        },
        'data_boost': {
            'name': 'Free Data Boost (2x Data)',
            'description': 'Double data allowance for 3 months',
            'treatment_effect': 0.06,
            'cost_per_customer': 3.00,
            'icon': '📶'
        },
        'loyalty_points': {
            'name': 'Loyalty Points Program',
            'description': 'Enroll in loyalty program with monthly rewards',
            'treatment_effect': 0.04,
            'cost_per_customer': 2.50,
            'icon': '🎁'
        },
        'personal_call': {
            'name': 'Personal Retention Call',
            'description': 'Dedicated account manager calls at-risk customers',
            'treatment_effect': 0.10,
            'cost_per_customer': 8.00,
            'icon': '📞'
        },
        'contract_incentive': {
            'name': '12-Month Contract Incentive',
            'description': 'Offer 15% discount for signing an annual contract',
            'treatment_effect': 0.12,
            'cost_per_customer': 10.00,
            'icon': '📝'
        }
    }
    
    def __init__(self, df):
        self.df = df.copy()
        self.results = {}
    
    def run_experiment(self, experiment_key, sample_size_pct=50, 
                       treatment_split=0.5, random_seed=42):
        """
        Simulate an A/B test experiment.
        
        Args:
            experiment_key: Key from EXPERIMENTS dict
            sample_size_pct: % of total customers to include (0-100)
            treatment_split: Fraction assigned to treatment (0-1)
            random_seed: For reproducibility
            
        Returns:
            Dictionary with experiment results and statistical tests
        """
        if experiment_key not in self.EXPERIMENTS:
            return {'error': f'Unknown experiment: {experiment_key}'}
        
        exp = self.EXPERIMENTS[experiment_key]
        np.random.seed(random_seed)
        
        # Sample customers
        n_sample = int(len(self.df) * sample_size_pct / 100)
        sample = self.df.sample(n=n_sample, random_state=random_seed)
        
        # Split into treatment and control
        n_treatment = int(len(sample) * treatment_split)
        treatment = sample.iloc[:n_treatment].copy()
        control = sample.iloc[n_treatment:].copy()
        
        # Simulate outcomes
        base_churn_rate = control['churned'].mean()
        
        # Control group: actual churn
        control['group'] = 'Control'
        control['outcome'] = control['churned']
        
        # Treatment group: reduced churn based on effect
        treatment['group'] = 'Treatment'
        treatment_effect = exp['treatment_effect']
        
        # Effect varies by customer — more effective for at-risk customers
        individual_effect = np.zeros(len(treatment))
        
        # Higher effect on dissatisfied customers
        sat_factor = (10 - treatment['satisfaction_score'].values) / 10
        individual_effect += sat_factor * treatment_effect * 0.5
        
        # Higher effect on customers with complaints
        comp_factor = np.clip(treatment['num_complaints_12m'].values / 5, 0, 1)
        individual_effect += comp_factor * treatment_effect * 0.3
        
        # Base effect for everyone
        individual_effect += treatment_effect * 0.2
        
        # Randomize slightly
        individual_effect += np.random.normal(0, treatment_effect * 0.1, len(treatment))
        individual_effect = np.clip(individual_effect, 0, 0.3)
        
        # Apply effect: reduce churn probability
        treatment_churn_prob = np.clip(
            treatment['churned'].values.astype(float) - individual_effect, 0, 1
        )
        treatment['outcome'] = np.random.binomial(1, 
            np.clip(treatment_churn_prob, 0.01, 0.99))
        
        # Calculate results
        control_churn = control['outcome'].mean()
        treatment_churn = treatment['outcome'].mean()
        churn_reduction = control_churn - treatment_churn
        relative_reduction = churn_reduction / max(control_churn, 0.001)
        
        # Statistical tests
        # Chi-squared test for proportions
        contingency = pd.crosstab(
            pd.concat([treatment, control])['group'],
            pd.concat([treatment, control])['outcome']
        )
        chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency)
        
        # Two-proportion z-test
        n1, n2 = len(treatment), len(control)
        p1, p2 = treatment_churn, control_churn
        p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        z_stat = (p1 - p2) / max(se, 1e-10)
        p_value_z = 2 * stats.norm.sf(abs(z_stat))
        
        # Confidence interval for the difference
        se_diff = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
        ci_lower = (p2 - p1) - 1.96 * se_diff  # note: reversed for positive = good
        ci_upper = (p2 - p1) + 1.96 * se_diff
        
        # Revenue impact
        avg_arpu = sample['arpu'].mean()
        monthly_revenue_saved = churn_reduction * len(treatment) * avg_arpu
        annual_revenue_saved = monthly_revenue_saved * 12
        total_cost = len(treatment) * exp['cost_per_customer']
        roi = (annual_revenue_saved - total_cost) / max(total_cost, 1)
        
        # Power analysis
        effect_size = abs(p2 - p1) / max(np.sqrt(p_pooled * (1 - p_pooled)), 0.001)
        
        # Minimum detectable effect
        mde = 2.8 * np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        result = {
            'experiment': exp,
            'experiment_key': experiment_key,
            'sample_size': n_sample,
            'n_treatment': n_treatment,
            'n_control': len(control),
            
            # Churn rates
            'control_churn': control_churn,
            'treatment_churn': treatment_churn,
            'churn_reduction': churn_reduction,
            'relative_reduction': relative_reduction,
            
            # Statistical tests
            'chi2_statistic': chi2,
            'p_value_chi2': p_value_chi2,
            'z_statistic': z_stat,
            'p_value_z': p_value_z,
            'is_significant': p_value_z < 0.05,
            
            # Confidence interval
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': 0.95,
            
            # Effect size
            'effect_size_h': effect_size,
            'mde': mde,
            
            # Revenue impact
            'avg_arpu': avg_arpu,
            'monthly_revenue_saved': monthly_revenue_saved,
            'annual_revenue_saved': annual_revenue_saved,
            'total_cost': total_cost,
            'roi': roi,
            
            # Recommendation
            'recommendation': self._make_recommendation(p_value_z, roi, churn_reduction)
        }
        
        self.results[experiment_key] = result
        return result
    
    def _make_recommendation(self, p_value, roi, churn_reduction):
        """Generate a go/no-go recommendation."""
        if p_value >= 0.05:
            return {
                'decision': '⏸️ INCONCLUSIVE',
                'color': 'warning',
                'reason': f'Results not statistically significant (p={p_value:.3f}). '
                         f'Need more data or larger sample size to draw conclusions.'
            }
        elif churn_reduction <= 0:
            return {
                'decision': '🛑 DO NOT LAUNCH',
                'color': 'error',
                'reason': f'Treatment group shows higher churn. The intervention may be '
                         f'causing harm. Investigate and redesign the strategy.'
            }
        elif roi < 0:
            return {
                'decision': '⚠️ REVIEW COSTS',
                'color': 'warning',
                'reason': f'Statistically significant churn reduction ({churn_reduction:.1%}), '
                         f'but negative ROI ({roi:.1f}x). Consider reducing intervention costs.'
            }
        else:
            return {
                'decision': '✅ LAUNCH',
                'color': 'success',
                'reason': f'Significant churn reduction of {churn_reduction:.1%} (p={p_value:.4f}) '
                         f'with {roi:.1f}x ROI. Recommend full deployment.'
            }
    
    def calculate_required_sample_size(self, baseline_rate, mde, alpha=0.05, power=0.8):
        """Calculate minimum sample size for desired statistical power."""
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        p1 = baseline_rate
        p2 = baseline_rate - mde
        
        p_bar = (p1 + p2) / 2
        
        n = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + 
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p1 - p2)) ** 2
        
        return int(np.ceil(n))
    
    def generate_power_curve(self, baseline_rate=None, sample_sizes=None):
        """Generate statistical power curves for different sample sizes."""
        if baseline_rate is None:
            baseline_rate = self.df['churned'].mean()
        
        if sample_sizes is None:
            sample_sizes = [100, 250, 500, 1000, 2000, 3000, 5000]
        
        effect_sizes = [0.02, 0.05, 0.08, 0.10, 0.15]
        
        power_data = []
        for n in sample_sizes:
            for effect in effect_sizes:
                p1 = baseline_rate
                p2 = max(baseline_rate - effect, 0.01)
                
                p_bar = (p1 + p2) / 2
                se = np.sqrt(p_bar * (1 - p_bar) * 2 / n)
                
                z_alpha = stats.norm.ppf(0.975)
                z = abs(p1 - p2) / max(se, 1e-10)
                power = stats.norm.cdf(z - z_alpha)
                
                power_data.append({
                    'sample_size': n,
                    'effect_size': f'{effect:.0%} reduction',
                    'power': min(power, 1.0)
                })
        
        return pd.DataFrame(power_data)


def run_ab_simulation(data_path='data/raw/master_dataset.csv'):
    """Run all experiments and summarize."""
    print("=" * 60)
    print("A/B TEST SIMULATION ENGINE")
    print("Experiment-driven retention strategy validation")
    print("=" * 60)
    
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} customers")
    
    simulator = ABTestSimulator(df)
    
    for key in ABTestSimulator.EXPERIMENTS:
        result = simulator.run_experiment(key)
        exp = result['experiment']
        print(f"\n  {exp['icon']} {exp['name']}:")
        print(f"    Control churn: {result['control_churn']:.1%}")
        print(f"    Treatment churn: {result['treatment_churn']:.1%}")
        print(f"    Reduction: {result['churn_reduction']:.1%} (p={result['p_value_z']:.4f})")
        print(f"    Decision: {result['recommendation']['decision']}")
    
    print(f"\n✅ A/B Test Simulation Complete!")
    return simulator


if __name__ == "__main__":
    simulator = run_ab_simulation()
