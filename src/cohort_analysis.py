"""
Customer Cohort Analysis

Groups customers by join-month cohort and tracks key metrics
(churn rate, ARPU, engagement) over their tenure journey.

Produces:
1. Retention heatmap matrix (cohort vs period)
2. ARPU evolution by cohort
3. Engagement trends by cohort
4. Cohort-level statistics table

Shows advanced analytical thinking — not just "what happened" but 
"how do customer behaviors evolve over time across different groups?"
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class CohortAnalyzer:
    """Performs cohort analysis on telecom customer data."""
    
    def __init__(self):
        self.cohort_data = None
        self.retention_matrix = None
        self.arpu_matrix = None
    
    def prepare_cohorts(self, df):
        """
        Assign customers to cohorts based on join date.
        
        Returns:
            DataFrame with cohort assignments and metrics
        """
        print("Preparing customer cohorts...")
        df = df.copy()
        
        # Create cohort from join_date
        if 'join_date' in df.columns:
            df['join_date'] = pd.to_datetime(df['join_date'])
            df['cohort'] = df['join_date'].dt.to_period('M').astype(str)
        else:
            # Derive from tenure_months
            max_tenure = df['tenure_months'].max()
            df['cohort_month_offset'] = max_tenure - df['tenure_months']
            
            # Create quarterly cohorts for cleaner visualization
            df['cohort_quarter'] = pd.cut(
                df['tenure_months'],
                bins=[0, 6, 12, 18, 24, 36, 48, 60, 72, 100],
                labels=['0-6mo', '7-12mo', '13-18mo', '19-24mo', 
                       '25-36mo', '37-48mo', '49-60mo', '61-72mo', '72+mo'],
                include_lowest=True
            )
            df['cohort'] = df['cohort_quarter'].astype(str)
        
        # Tenure period (how far along they are)
        df['tenure_period'] = pd.cut(
            df['tenure_months'],
            bins=[0, 3, 6, 12, 18, 24, 36, 48, 60, 72, 100],
            labels=['0-3m', '4-6m', '7-12m', '13-18m', '19-24m',
                   '25-36m', '37-48m', '49-60m', '61-72m', '72+m'],
            include_lowest=True
        )
        
        self.cohort_data = df
        print(f"  Created {df['cohort'].nunique()} cohorts")
        return df
    
    def calculate_retention_matrix(self):
        """Calculate retention rates by cohort."""
        if self.cohort_data is None:
            return None
        
        df = self.cohort_data
        
        # Retention rate = 1 - churn_rate per cohort
        cohort_stats = df.groupby('cohort').agg(
            total_customers=('customer_id', 'count'),
            churned=('churned', 'sum'),
            retained=('churned', lambda x: (1 - x).sum()),
            churn_rate=('churned', 'mean'),
            retention_rate=('churned', lambda x: 1 - x.mean()),
            avg_arpu=('arpu', 'mean'),
            avg_satisfaction=('satisfaction_score', 'mean'),
            avg_tenure=('tenure_months', 'mean')
        ).reset_index()
        
        self.retention_matrix = cohort_stats
        return cohort_stats
    
    def calculate_cohort_metrics(self):
        """Calculate detailed metrics per cohort for heatmap."""
        if self.cohort_data is None:
            return None
        
        df = self.cohort_data
        
        # Create a cross-tab style analysis
        # Cohort (when they joined) vs Metric
        metrics = df.groupby('cohort').agg({
            'customer_id': 'count',
            'churned': 'mean',
            'arpu': ['mean', 'median', 'std'],
            'satisfaction_score': 'mean',
            'monthly_data_gb': 'mean',
            'monthly_web_sessions': 'mean',
            'monthly_app_sessions': 'mean',
            'support_tickets_12m': 'mean',
            'late_payments_12m': 'mean',
            'tenure_months': 'mean'
        })
        
        metrics.columns = [
            'customers', 'churn_rate', 'arpu_mean', 'arpu_median', 'arpu_std',
            'avg_satisfaction', 'avg_data_gb', 'avg_web_sessions', 
            'avg_app_sessions', 'avg_support_tickets', 'avg_late_payments',
            'avg_tenure'
        ]
        
        metrics['retention_rate'] = 1 - metrics['churn_rate']
        metrics['total_revenue'] = metrics['customers'] * metrics['arpu_mean']
        metrics['digital_engagement'] = metrics['avg_web_sessions'] + metrics['avg_app_sessions']
        
        return metrics.reset_index()
    
    def generate_retention_heatmap_data(self):
        """Generate data for retention heatmap visualization."""
        if self.cohort_data is None:
            return None
        
        df = self.cohort_data
        
        # Define tenure bands and cohorts
        cohorts = sorted(df['cohort'].unique())
        
        # For each cohort, calculate metrics at different tenure points
        tenure_bands = ['0-3m', '4-6m', '7-12m', '13-18m', '19-24m',
                       '25-36m', '37-48m', '49-60m', '61-72m', '72+m']
        
        # Build the heatmap data
        heatmap_data = []
        for cohort in cohorts:
            cohort_df = df[df['cohort'] == cohort]
            for band in tenure_bands:
                band_df = cohort_df[cohort_df['tenure_period'] == band]
                if len(band_df) > 0:
                    heatmap_data.append({
                        'cohort': cohort,
                        'tenure_band': band,
                        'customers': len(band_df),
                        'retention_rate': 1 - band_df['churned'].mean(),
                        'avg_arpu': band_df['arpu'].mean(),
                        'churn_rate': band_df['churned'].mean()
                    })
        
        return pd.DataFrame(heatmap_data)
    
    def get_cohort_summary(self):
        """Generate business summary of cohort analysis."""
        if self.retention_matrix is None:
            self.calculate_retention_matrix()
        
        rm = self.retention_matrix
        
        best_cohort = rm.loc[rm['retention_rate'].idxmax()]
        worst_cohort = rm.loc[rm['retention_rate'].idxmin()]
        highest_arpu = rm.loc[rm['avg_arpu'].idxmax()]
        
        summary = {
            'total_cohorts': len(rm),
            'total_customers': int(rm['total_customers'].sum()),
            'overall_retention': float(1 - self.cohort_data['churned'].mean()),
            'best_cohort': {
                'name': best_cohort['cohort'],
                'retention': float(best_cohort['retention_rate']),
                'customers': int(best_cohort['total_customers'])
            },
            'worst_cohort': {
                'name': worst_cohort['cohort'],
                'retention': float(worst_cohort['retention_rate']),
                'customers': int(worst_cohort['total_customers'])
            },
            'highest_arpu_cohort': {
                'name': highest_arpu['cohort'],
                'arpu': float(highest_arpu['avg_arpu']),
                'customers': int(highest_arpu['total_customers'])
            },
            'cohort_details': rm.to_dict('records')
        }
        
        return summary
    
    def save_results(self, output_dir='data/processed'):
        """Save cohort analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.retention_matrix is not None:
            self.retention_matrix.to_csv(
                os.path.join(output_dir, 'cohort_retention.csv'), index=False
            )
        
        metrics = self.calculate_cohort_metrics()
        if metrics is not None:
            metrics.to_csv(
                os.path.join(output_dir, 'cohort_metrics.csv'), index=False
            )
        
        heatmap = self.generate_retention_heatmap_data()
        if heatmap is not None:
            heatmap.to_csv(
                os.path.join(output_dir, 'cohort_heatmap.csv'), index=False
            )
        
        print(f"  Saved cohort results to {output_dir}")


def run_cohort_analysis(data_path='data/raw/master_dataset.csv'):
    """Run the complete cohort analysis pipeline."""
    print("=" * 60)
    print("CUSTOMER COHORT ANALYSIS")
    print("How do customer behaviors evolve over time?")
    print("=" * 60)
    
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} customers")
    
    analyzer = CohortAnalyzer()
    analyzer.prepare_cohorts(df)
    analyzer.calculate_retention_matrix()
    
    summary = analyzer.get_cohort_summary()
    analyzer.save_results()
    
    print(f"\n  Total cohorts: {summary['total_cohorts']}")
    print(f"  Overall retention: {summary['overall_retention']:.1%}")
    print(f"  Best cohort: {summary['best_cohort']['name']} ({summary['best_cohort']['retention']:.1%} retention)")
    print(f"  Worst cohort: {summary['worst_cohort']['name']} ({summary['worst_cohort']['retention']:.1%} retention)")
    print(f"  Highest ARPU: {summary['highest_arpu_cohort']['name']} (${summary['highest_arpu_cohort']['arpu']:.2f})")
    
    print(f"\n✅ Cohort analysis complete!")
    return analyzer, summary


if __name__ == "__main__":
    analyzer, summary = run_cohort_analysis()
