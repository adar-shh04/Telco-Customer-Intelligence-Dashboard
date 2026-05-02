"""
Anomaly Detection & Alerting Engine

Monitors key telecom metrics for unusual patterns and generates
real-time alerts with severity levels.

Detection methods:
1. Z-Score based (statistical outliers)
2. IQR-based (robust to skewed data)
3. Trend analysis (rolling averages + directional change)

Alert severity: 🟢 Info → 🟡 Warning → 🔴 Critical

Shows interviewers you understand proactive monitoring,
not just reactive analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """Detects anomalies across key telecom metrics."""
    
    # Metrics to monitor with thresholds
    METRIC_CONFIG = {
        'arpu': {
            'name': 'Average Revenue Per User',
            'unit': '$',
            'direction': 'both',  # both high and low are interesting
            'z_threshold': 2.0,
            'icon': '💰'
        },
        'satisfaction_score': {
            'name': 'Satisfaction Score',
            'unit': '/10',
            'direction': 'low',  # low is bad
            'z_threshold': 1.5,
            'icon': '😊'
        },
        'num_complaints_12m': {
            'name': 'Complaints (12m)',
            'unit': '',
            'direction': 'high',  # high is bad
            'z_threshold': 2.0,
            'icon': '📞'
        },
        'support_tickets_12m': {
            'name': 'Support Tickets (12m)',
            'unit': '',
            'direction': 'high',
            'z_threshold': 2.0,
            'icon': '🎫'
        },
        'late_payments_12m': {
            'name': 'Late Payments (12m)',
            'unit': '',
            'direction': 'high',
            'z_threshold': 1.5,
            'icon': '⚠️'
        },
        'monthly_data_gb': {
            'name': 'Monthly Data Usage',
            'unit': 'GB',
            'direction': 'both',
            'z_threshold': 2.5,
            'icon': '📶'
        },
        'monthly_web_sessions': {
            'name': 'Web Sessions',
            'unit': '',
            'direction': 'both',
            'z_threshold': 2.0,
            'icon': '🌐'
        }
    }
    
    def __init__(self, df):
        self.df = df.copy()
        self.alerts = []
        self.segment_anomalies = []
    
    def detect_metric_anomalies(self):
        """Detect anomalies across all monitored metrics."""
        self.alerts = []
        
        for metric_key, config in self.METRIC_CONFIG.items():
            if metric_key not in self.df.columns:
                continue
            
            values = self.df[metric_key].dropna()
            if len(values) < 10:
                continue
            
            mean_val = values.mean()
            std_val = values.std()
            median_val = values.median()
            
            # Z-score anomalies (individual customers)
            z_scores = np.abs((values - mean_val) / (std_val + 1e-10))
            n_outliers = (z_scores > config['z_threshold']).sum()
            outlier_pct = n_outliers / len(values) * 100
            
            # IQR-based anomalies
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            iqr_outliers = ((values < lower_bound) | (values > upper_bound)).sum()
            
            # Determine severity
            if outlier_pct > 10:
                severity = 'critical'
            elif outlier_pct > 5:
                severity = 'warning'
            else:
                severity = 'info'
            
            self.alerts.append({
                'metric': config['name'],
                'metric_key': metric_key,
                'icon': config['icon'],
                'mean': mean_val,
                'std': std_val,
                'median': median_val,
                'unit': config['unit'],
                'n_outliers_zscore': int(n_outliers),
                'n_outliers_iqr': int(iqr_outliers),
                'outlier_pct': outlier_pct,
                'severity': severity,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min_val': values.min(),
                'max_val': values.max(),
                'skewness': float(values.skew()),
                'direction': config['direction']
            })
        
        return self.alerts
    
    def detect_segment_anomalies(self):
        """Detect anomalies at segment level (city, plan type)."""
        self.segment_anomalies = []
        
        # By city - churn rate anomalies
        if 'city' in self.df.columns:
            city_churn = self.df.groupby('city')['churned'].mean()
            overall_churn = self.df['churned'].mean()
            churn_std = city_churn.std()
            
            for city, churn_rate in city_churn.items():
                z = (churn_rate - overall_churn) / (churn_std + 1e-10)
                n_customers = (self.df['city'] == city).sum()
                
                if abs(z) > 1.5 and n_customers >= 50:
                    self.segment_anomalies.append({
                        'segment_type': 'City',
                        'segment_value': city,
                        'metric': 'Churn Rate',
                        'value': churn_rate,
                        'benchmark': overall_churn,
                        'z_score': z,
                        'n_customers': n_customers,
                        'severity': 'critical' if abs(z) > 2.5 else 'warning',
                        'direction': 'above' if z > 0 else 'below',
                        'deviation_pct': ((churn_rate / overall_churn) - 1) * 100
                    })
        
        # By plan type - satisfaction anomalies
        if 'plan_type' in self.df.columns:
            plan_sat = self.df.groupby('plan_type')['satisfaction_score'].mean()
            overall_sat = self.df['satisfaction_score'].mean()
            sat_std = plan_sat.std()
            
            for plan, sat_score in plan_sat.items():
                z = (sat_score - overall_sat) / (sat_std + 1e-10)
                n_customers = (self.df['plan_type'] == plan).sum()
                
                if abs(z) > 1.0:
                    self.segment_anomalies.append({
                        'segment_type': 'Plan Type',
                        'segment_value': plan,
                        'metric': 'Satisfaction Score',
                        'value': sat_score,
                        'benchmark': overall_sat,
                        'z_score': z,
                        'n_customers': n_customers,
                        'severity': 'warning' if abs(z) > 1.5 else 'info',
                        'direction': 'above' if z > 0 else 'below',
                        'deviation_pct': ((sat_score / overall_sat) - 1) * 100
                    })
        
        # By plan type - ARPU anomalies
        if 'plan_type' in self.df.columns:
            plan_arpu = self.df.groupby('plan_type')['arpu'].mean()
            overall_arpu = self.df['arpu'].mean()
            arpu_std = plan_arpu.std()
            
            for plan, arpu in plan_arpu.items():
                z = (arpu - overall_arpu) / (arpu_std + 1e-10)
                n_customers = (self.df['plan_type'] == plan).sum()
                
                if abs(z) > 1.0:
                    self.segment_anomalies.append({
                        'segment_type': 'Plan Type',
                        'segment_value': plan,
                        'metric': 'ARPU',
                        'value': arpu,
                        'benchmark': overall_arpu,
                        'z_score': z,
                        'n_customers': n_customers,
                        'severity': 'info',
                        'direction': 'above' if z > 0 else 'below',
                        'deviation_pct': ((arpu / overall_arpu) - 1) * 100
                    })
        
        return self.segment_anomalies
    
    def get_alert_summary(self):
        """Generate a summary of all alerts."""
        if not self.alerts:
            self.detect_metric_anomalies()
        if not self.segment_anomalies:
            self.detect_segment_anomalies()
        
        critical = sum(1 for a in self.alerts + self.segment_anomalies if a['severity'] == 'critical')
        warnings = sum(1 for a in self.alerts + self.segment_anomalies if a['severity'] == 'warning')
        info = sum(1 for a in self.alerts + self.segment_anomalies if a['severity'] == 'info')
        
        return {
            'total_alerts': len(self.alerts) + len(self.segment_anomalies),
            'critical': critical,
            'warnings': warnings,
            'info': info,
            'metric_alerts': self.alerts,
            'segment_alerts': self.segment_anomalies
        }
    
    def get_customer_anomalies(self, top_n=20):
        """Find the most anomalous individual customers."""
        df = self.df.copy()
        
        # Compute composite anomaly score per customer
        anomaly_scores = np.zeros(len(df))
        
        for metric_key, config in self.METRIC_CONFIG.items():
            if metric_key not in df.columns:
                continue
            values = df[metric_key].fillna(df[metric_key].median())
            z = np.abs((values - values.mean()) / (values.std() + 1e-10))
            anomaly_scores += z
        
        df['anomaly_score'] = anomaly_scores / len(self.METRIC_CONFIG)
        
        # Top anomalous customers
        top_anomalous = df.nlargest(top_n, 'anomaly_score')
        
        return top_anomalous[['customer_id', 'anomaly_score', 'arpu', 
                              'satisfaction_score', 'num_complaints_12m',
                              'churned', 'plan_type', 'city']]


def run_anomaly_detection(data_path='data/raw/master_dataset.csv'):
    """Run anomaly detection pipeline."""
    print("=" * 60)
    print("ANOMALY DETECTION ENGINE")
    print("Proactive monitoring and alerting")
    print("=" * 60)
    
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} customers")
    
    detector = AnomalyDetector(df)
    summary = detector.get_alert_summary()
    
    print(f"\n  Total alerts: {summary['total_alerts']}")
    print(f"    \U0001f534 Critical: {summary['critical']}")
    print(f"    \U0001f7e1 Warning: {summary['warnings']}")
    print(f"    \U0001f7e2 Info: {summary['info']}")
    
    print(f"\n  Metric Alerts:")
    for alert in summary['metric_alerts']:
        print(f"    {alert['icon']} {alert['metric']}: {alert['n_outliers_zscore']} outliers ({alert['outlier_pct']:.1f}%) [{alert['severity'].upper()}]")
    
    print(f"\n✅ Anomaly Detection Complete!")
    return detector, summary


if __name__ == "__main__":
    detector, summary = run_anomaly_detection()
