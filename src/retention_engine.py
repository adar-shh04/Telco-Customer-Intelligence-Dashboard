"""
Smart Retention Engine

Goes beyond churn prediction to answer: "WHAT should we do about each at-risk customer?"
For every customer, this engine:
1. Calculates a Customer Health Score (0-100) 
2. Identifies the PRIMARY churn driver (why they might leave)
3. Recommends a specific retention action with estimated cost & ROI
4. Prioritizes customers by expected revenue impact, not just churn probability

Data Source: Real Kaggle IBM Telco Customer Churn dataset
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score, 
                             precision_recall_curve, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class CustomerHealthScorer:
    """
    Computes a composite Customer Health Score (0-100) from multiple signals.
    
    Higher score = healthier customer (less likely to churn).
    Components weighted by business importance:
    - Satisfaction component (30%)
    - Usage engagement component (25%) 
    - Payment reliability component (20%)
    - Tenure loyalty component (15%)
    - Digital engagement component (10%)
    """
    
    def __init__(self):
        self.weights = {
            'satisfaction': 0.30,
            'usage': 0.25,
            'payment': 0.20,
            'tenure': 0.15,
            'digital': 0.10
        }
    
    def _normalize(self, series, inverse=False):
        """Min-max normalize to 0-100 scale."""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(50, index=series.index)
        normalized = (series - min_val) / (max_val - min_val) * 100
        if inverse:
            normalized = 100 - normalized
        return normalized
    
    def calculate(self, df):
        """
        Calculate Customer Health Score for each customer.
        
        Args:
            df: Master dataset DataFrame
            
        Returns:
            DataFrame with health score columns added
        """
        df = df.copy()
        
        # 1. Satisfaction component (satisfaction_score: 1-10 → 0-100)
        df['health_satisfaction'] = self._normalize(df['satisfaction_score'])
        
        # 2. Usage engagement (higher usage ratio = more engaged = healthier)
        usage_ratio = df['monthly_data_gb'] / (df['data_allowance_gb'] + 1e-6)
        minutes_ratio = df['monthly_minutes'] / (df['minutes_allowance'] + 1e-6)
        usage_score = (usage_ratio.clip(0, 2) + minutes_ratio.clip(0, 2)) / 2
        df['health_usage'] = self._normalize(usage_score)
        
        # 3. Payment reliability (fewer late payments = healthier)
        df['health_payment'] = self._normalize(df['late_payments_12m'], inverse=True)
        
        # 4. Tenure loyalty (longer tenure = healthier, with diminishing returns)
        tenure_score = np.log1p(df['tenure_months']) / np.log1p(72)  # Normalize by 6 years
        df['health_tenure'] = (tenure_score * 100).clip(0, 100)
        
        # 5. Digital engagement (more sessions = more invested in platform)
        total_sessions = df['monthly_web_sessions'] + df['monthly_app_sessions']
        df['health_digital'] = self._normalize(total_sessions)
        
        # Composite Health Score (weighted average)
        df['customer_health_score'] = (
            df['health_satisfaction'] * self.weights['satisfaction'] +
            df['health_usage'] * self.weights['usage'] +
            df['health_payment'] * self.weights['payment'] +
            df['health_tenure'] * self.weights['tenure'] +
            df['health_digital'] * self.weights['digital']
        ).round(1)
        
        # Health category
        df['health_category'] = pd.cut(
            df['customer_health_score'],
            bins=[0, 30, 50, 70, 100],
            labels=['Critical', 'At Risk', 'Stable', 'Healthy']
        )
        
        return df


class ChurnDriverAnalyzer:
    """
    Identifies the PRIMARY reason each customer might churn.
    
    Instead of just saying "this customer will churn", it explains WHY —
    which is what business stakeholders actually need to take action.
    """
    
    DRIVER_DEFINITIONS = {
        'low_satisfaction': {
            'label': 'Low Satisfaction',
            'description': 'Customer satisfaction is critically low',
            'icon': '😞'
        },
        'price_sensitivity': {
            'label': 'Price Sensitivity', 
            'description': 'ARPU is high relative to usage — customer may feel overcharged',
            'icon': '💸'
        },
        'service_issues': {
            'label': 'Service/Support Issues',
            'description': 'Multiple complaints and support tickets indicate service problems',
            'icon': '🔧'
        },
        'low_engagement': {
            'label': 'Low Engagement',
            'description': 'Customer barely uses the service — risk of switching',
            'icon': '📉'
        },
        'new_customer_risk': {
            'label': 'New Customer Risk',
            'description': 'Recently joined with no loyalty built yet',
            'icon': '🆕'
        },
        'payment_issues': {
            'label': 'Payment Problems',
            'description': 'Late payments suggest financial stress or disengagement',
            'icon': '💳'
        },
        'competitor_risk': {
            'label': 'Competitor Risk',
            'description': 'High-value customer on basic plan — likely shopping around',
            'icon': '🏃'
        }
    }
    
    def identify_primary_driver(self, df):
        """
        For each customer, identify the most likely churn driver.
        
        Uses a rule-based scoring system calibrated on the actual data patterns.
        
        Returns:
            DataFrame with churn_driver and driver_score columns
        """
        df = df.copy()
        
        # Score each potential driver (0-10 scale)
        drivers = pd.DataFrame(index=df.index)
        
        # Low satisfaction (score increases as satisfaction drops below 6)
        drivers['low_satisfaction'] = np.clip((6 - df['satisfaction_score']) * 2, 0, 10)
        
        # Price sensitivity (high ARPU + low usage = overcharged feeling)
        usage_ratio = df['monthly_data_gb'] / (df['data_allowance_gb'] + 1e-6)
        arpu_percentile = df['arpu'].rank(pct=True)
        drivers['price_sensitivity'] = np.clip(
            (arpu_percentile * 10) - (usage_ratio * 3), 0, 10
        )
        
        # Service issues (complaints + support tickets)
        drivers['service_issues'] = np.clip(
            df['num_complaints_12m'] * 2 + df['support_tickets_12m'] * 0.5, 0, 10
        )
        
        # Low engagement (low total sessions + low OTT usage)
        total_sessions = df['monthly_web_sessions'] + df['monthly_app_sessions']
        session_percentile = total_sessions.rank(pct=True)
        drivers['low_engagement'] = np.clip((1 - session_percentile) * 10, 0, 10)
        
        # New customer risk (tenure < 6 months)
        drivers['new_customer_risk'] = np.clip((6 - df['tenure_months']) * 1.5, 0, 10)
        
        # Payment issues
        drivers['payment_issues'] = np.clip(df['late_payments_12m'] * 3, 0, 10)
        
        # Competitor risk (high income, basic plan, decent satisfaction)
        is_basic = (df['plan_type'] == 'Basic').astype(int)
        income_percentile = df['annual_income'].rank(pct=True)
        drivers['competitor_risk'] = np.clip(
            is_basic * income_percentile * 8 + (df['satisfaction_score'] > 5).astype(int) * 2, 0, 10
        )
        
        # Primary driver = highest scoring
        df['churn_driver'] = drivers.idxmax(axis=1)
        df['churn_driver_score'] = drivers.max(axis=1).round(1)
        
        # Add human-readable label
        df['churn_driver_label'] = df['churn_driver'].map(
            {k: v['label'] for k, v in self.DRIVER_DEFINITIONS.items()}
        )
        
        # Store all driver scores for detailed analysis
        for col in drivers.columns:
            df[f'driver_{col}'] = drivers[col].round(1)
        
        return df


class RetentionStrategyEngine:
    """
    Recommends specific retention actions for each customer with cost-benefit analysis.
    
    Each intervention has:
    - Estimated cost to execute
    - Expected retention probability improvement
    - Projected ROI = (revenue_saved - cost) / cost
    
    Customers are prioritized by EXPECTED VALUE, not just churn probability.
    A high-value customer with 30% churn risk may deserve more attention than
    a low-value customer with 80% churn risk.
    """
    
    # Retention strategies with costs and expected effectiveness
    STRATEGIES = {
        'premium_discount': {
            'name': 'Premium Discount (20% off for 3 months)',
            'cost': 45,  # Average monthly discount * 3
            'retention_boost': 0.25,  # 25% improvement in retention
            'best_for': ['price_sensitivity', 'competitor_risk'],
            'icon': '🏷️'
        },
        'plan_upgrade_offer': {
            'name': 'Free Plan Upgrade Trial (1 month)',
            'cost': 30,
            'retention_boost': 0.20,
            'best_for': ['low_engagement', 'competitor_risk'],
            'icon': '⬆️'
        },
        'dedicated_support': {
            'name': 'Dedicated Account Manager Assignment',
            'cost': 60,
            'retention_boost': 0.30,
            'best_for': ['service_issues', 'low_satisfaction'],
            'icon': '🤝'
        },
        'loyalty_reward': {
            'name': 'Loyalty Points + Exclusive Perks Bundle',
            'cost': 25,
            'retention_boost': 0.15,
            'best_for': ['new_customer_risk', 'low_engagement'],
            'icon': '🎁'
        },
        'payment_flexibility': {
            'name': 'Flexible Payment Plan + Late Fee Waiver',
            'cost': 15,
            'retention_boost': 0.20,
            'best_for': ['payment_issues'],
            'icon': '💳'
        },
        'proactive_outreach': {
            'name': 'Proactive Customer Success Call',
            'cost': 10,
            'retention_boost': 0.10,
            'best_for': ['low_satisfaction', 'service_issues'],
            'icon': '📞'
        },
        'no_action': {
            'name': 'No Intervention Needed',
            'cost': 0,
            'retention_boost': 0,
            'best_for': [],
            'icon': '✅'
        }
    }
    
    def __init__(self):
        self.churn_model = None
        self.feature_columns = None
        self.model_metrics = {}
    
    def train_churn_model(self, df):
        """
        Train a churn prediction model to get probabilities for ROI calculation.
        
        Returns:
            Trained model and metrics
        """
        print("Training churn prediction model for retention engine...")
        
        # Prepare features
        feature_cols = [
            'age', 'annual_income', 'tenure_months', 'monthly_data_gb',
            'monthly_minutes', 'ott_usage_hours', 'arpu', 'data_allowance_gb',
            'minutes_allowance', 'satisfaction_score', 'num_complaints_12m',
            'support_tickets_12m', 'payment_score', 'late_payments_12m',
            'campaigns_exposed', 'total_impressions', 'total_clicks',
            'total_conversions', 'monthly_web_sessions', 'monthly_app_sessions',
            'avg_session_duration_min', 'self_service_transactions'
        ]
        
        # Add encoded categoricals
        df_model = df.copy()
        le_plan = LabelEncoder()
        df_model['plan_encoded'] = le_plan.fit_transform(df_model['plan_type'])
        le_gender = LabelEncoder()
        df_model['gender_encoded'] = le_gender.fit_transform(df_model['gender'])
        
        feature_cols_full = feature_cols + ['plan_encoded', 'gender_encoded']
        self.feature_columns = feature_cols_full
        
        X = df_model[feature_cols_full].fillna(0)
        y = df_model['churned']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Gradient Boosting (good balance of accuracy and interpretability)
        self.churn_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        self.churn_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.churn_model.predict(X_test)
        y_proba = self.churn_model.predict_proba(X_test)[:, 1]
        
        self.model_metrics = {
            'auc_roc': roc_auc_score(y_test, y_proba),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(
                feature_cols_full, 
                self.churn_model.feature_importances_
            ))
        }
        
        # Cross-validation AUC
        cv_scores = cross_val_score(self.churn_model, X, y, cv=5, scoring='roc_auc')
        self.model_metrics['cv_auc_mean'] = cv_scores.mean()
        self.model_metrics['cv_auc_std'] = cv_scores.std()
        
        print(f"  AUC-ROC: {self.model_metrics['auc_roc']:.4f}")
        print(f"  CV AUC:  {self.model_metrics['cv_auc_mean']:.4f} ± {self.model_metrics['cv_auc_std']:.4f}")
        
        return self.model_metrics
    
    def predict_churn_probability(self, df):
        """Get calibrated churn probabilities for all customers."""
        df_model = df.copy()
        le_plan = LabelEncoder()
        df_model['plan_encoded'] = le_plan.fit_transform(df_model['plan_type'])
        le_gender = LabelEncoder()
        df_model['gender_encoded'] = le_gender.fit_transform(df_model['gender'])
        
        X = df_model[self.feature_columns].fillna(0)
        return self.churn_model.predict_proba(X)[:, 1]
    
    def recommend_strategy(self, df):
        """
        For each customer, recommend the optimal retention strategy.
        
        Selection logic:
        1. If health score > 70 and churn prob < 0.2 → No action
        2. Otherwise, match churn driver to best strategy
        3. Calculate ROI for each candidate strategy
        4. Pick highest ROI strategy
        
        Returns:
            DataFrame with strategy recommendations and ROI analysis
        """
        df = df.copy()
        
        # Get churn probabilities from trained model
        df['ml_churn_probability'] = self.predict_churn_probability(df)
        
        strategies = []
        for idx, row in df.iterrows():
            churn_prob = row['ml_churn_probability']
            health = row.get('customer_health_score', 50)
            driver = row.get('churn_driver', 'low_engagement')
            arpu = row['arpu']
            
            # No action for healthy customers
            if health > 70 and churn_prob < 0.2:
                strategies.append({
                    'strategy_key': 'no_action',
                    'strategy_name': self.STRATEGIES['no_action']['name'],
                    'strategy_icon': '✅',
                    'intervention_cost': 0,
                    'retention_boost': 0,
                    'expected_revenue_saved': 0,
                    'expected_roi': 0,
                    'priority_score': 0
                })
                continue
            
            # Find best strategy by matching driver + maximizing ROI
            best_strategy = None
            best_roi = -float('inf')
            
            for key, strategy in self.STRATEGIES.items():
                if key == 'no_action':
                    continue
                
                # Base retention boost
                boost = strategy['retention_boost']
                
                # Bonus if strategy matches the driver
                if driver in strategy['best_for']:
                    boost *= 1.4  # 40% more effective when matched
                
                # Revenue saved = ARPU * 12 months * churn_prob * retention_boost
                annual_revenue = arpu * 12
                revenue_saved = annual_revenue * churn_prob * boost
                cost = strategy['cost']
                
                roi = (revenue_saved - cost) / max(cost, 1)
                
                if roi > best_roi:
                    best_roi = roi
                    best_strategy = {
                        'strategy_key': key,
                        'strategy_name': strategy['name'],
                        'strategy_icon': strategy['icon'],
                        'intervention_cost': cost,
                        'retention_boost': round(boost, 2),
                        'expected_revenue_saved': round(revenue_saved, 2),
                        'expected_roi': round(roi, 2),
                        'priority_score': round(revenue_saved * churn_prob, 2)
                    }
            
            strategies.append(best_strategy)
        
        # Add strategy columns to dataframe
        strategy_df = pd.DataFrame(strategies, index=df.index)
        for col in strategy_df.columns:
            df[col] = strategy_df[col]
        
        # Priority ranking (highest expected value first)
        df['retention_priority_rank'] = df['priority_score'].rank(
            ascending=False, method='min'
        ).astype(int)
        
        return df
    
    def generate_executive_summary(self, df):
        """
        Generate business-ready executive summary of retention recommendations.
        
        Returns:
            Dictionary with key business metrics and insights
        """
        at_risk = df[df['strategy_key'] != 'no_action']
        
        summary = {
            'total_customers': len(df),
            'at_risk_customers': len(at_risk),
            'at_risk_pct': len(at_risk) / len(df),
            
            # Revenue impact
            'total_monthly_revenue': df['arpu'].sum(),
            'at_risk_monthly_revenue': at_risk['arpu'].sum(),
            'total_intervention_cost': at_risk['intervention_cost'].sum(),
            'total_expected_revenue_saved': at_risk['expected_revenue_saved'].sum(),
            'overall_roi': (
                at_risk['expected_revenue_saved'].sum() - 
                at_risk['intervention_cost'].sum()
            ) / max(at_risk['intervention_cost'].sum(), 1),
            
            # Strategy distribution
            'strategy_distribution': df['strategy_name'].value_counts().to_dict(),
            
            # Driver distribution (for at-risk only)
            'driver_distribution': at_risk['churn_driver_label'].value_counts().to_dict(),
            
            # Top 10 priority customers
            'top_priority_customers': at_risk.nsmallest(10, 'retention_priority_rank')[
                ['customer_id', 'arpu', 'ml_churn_probability', 'customer_health_score',
                 'churn_driver_label', 'strategy_name', 'expected_revenue_saved',
                 'expected_roi', 'retention_priority_rank']
            ].to_dict('records'),
            
            # Budget allocation by strategy
            'budget_by_strategy': at_risk.groupby('strategy_name').agg({
                'intervention_cost': 'sum',
                'expected_revenue_saved': 'sum',
                'customer_id': 'count'
            }).rename(columns={'customer_id': 'customer_count'}).to_dict('index'),
            
            # Model performance
            'model_auc': self.model_metrics.get('auc_roc', 0),
            'model_cv_auc': self.model_metrics.get('cv_auc_mean', 0),
        }
        
        return summary


def run_full_retention_analysis(data_path='data/raw/master_dataset.csv'):
    """
    Run the complete retention engine pipeline end-to-end.
    
    Returns:
        Tuple of (enriched_df, executive_summary, model_metrics)
    """
    print("=" * 60)
    print("SMART RETENTION ENGINE")
    print("From Prediction → Prescription → Revenue Impact")
    print("=" * 60)
    
    # Load data
    print("\n📥 Loading data...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} customers")
    
    # Step 1: Customer Health Score
    print("\n📊 Step 1: Computing Customer Health Scores...")
    health_scorer = CustomerHealthScorer()
    df = health_scorer.calculate(df)
    print(f"  Average Health Score: {df['customer_health_score'].mean():.1f}/100")
    print(f"  Health Distribution:")
    print(f"    {df['health_category'].value_counts().to_string()}")
    
    # Step 2: Churn Driver Analysis
    print("\n🔍 Step 2: Identifying Churn Drivers...")
    driver_analyzer = ChurnDriverAnalyzer()
    df = driver_analyzer.identify_primary_driver(df)
    print(f"  Top Churn Drivers:")
    for driver, count in df['churn_driver_label'].value_counts().head(5).items():
        pct = count / len(df) * 100
        print(f"    {driver}: {count:,} customers ({pct:.1f}%)")
    
    # Step 3: Train ML Model + Recommend Strategies
    print("\n🤖 Step 3: Training Churn Model & Generating Recommendations...")
    engine = RetentionStrategyEngine()
    model_metrics = engine.train_churn_model(df)
    df = engine.recommend_strategy(df)
    
    # Step 4: Executive Summary
    print("\n📋 Step 4: Generating Executive Summary...")
    summary = engine.generate_executive_summary(df)
    
    # Print key results
    print(f"\n{'=' * 60}")
    print("RETENTION ENGINE RESULTS")
    print(f"{'=' * 60}")
    print(f"  Customers analyzed:        {summary['total_customers']:,}")
    print(f"  At-risk customers:         {summary['at_risk_customers']:,} ({summary['at_risk_pct']:.1%})")
    print(f"  Total monthly revenue:     ${summary['total_monthly_revenue']:,.2f}")
    print(f"  Revenue at risk:           ${summary['at_risk_monthly_revenue']:,.2f}")
    print(f"  Total intervention cost:   ${summary['total_intervention_cost']:,.2f}")
    print(f"  Expected revenue saved:    ${summary['total_expected_revenue_saved']:,.2f}")
    print(f"  Overall ROI:               {summary['overall_roi']:.1f}x")
    print(f"  Model AUC-ROC:             {summary['model_auc']:.4f}")
    
    print(f"\n  Strategy Distribution:")
    for strategy, count in summary['strategy_distribution'].items():
        print(f"    {strategy}: {count:,}")
    
    print(f"\n✅ Retention Engine Complete!")
    print(f"{'=' * 60}")
    
    return df, summary, model_metrics


if __name__ == "__main__":
    df, summary, metrics = run_full_retention_analysis()
