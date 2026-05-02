"""
Model Explainability Engine (SHAP)

Provides transparent, interpretable explanations for churn predictions.
Answers the critical interview question: "How do you explain your model to business?"

Uses SHAP (SHapley Additive exPlanations) to:
1. Show which features drive predictions globally
2. Explain individual customer predictions
3. Reveal feature interactions and nonlinear effects
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class ExplainabilityEngine:
    """SHAP-based model explainability for churn predictions."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.shap_values = None
        self.X_test = None
        self.model_metrics = {}
    
    def train_model(self, df):
        """Train a churn model and prepare for SHAP analysis."""
        print("Training churn model for explainability analysis...")
        
        feature_cols = [
            'age', 'annual_income', 'tenure_months', 'monthly_data_gb',
            'monthly_minutes', 'ott_usage_hours', 'arpu', 'data_allowance_gb',
            'minutes_allowance', 'satisfaction_score', 'num_complaints_12m',
            'support_tickets_12m', 'payment_score', 'late_payments_12m',
            'campaigns_exposed', 'total_clicks', 'total_conversions',
            'monthly_web_sessions', 'monthly_app_sessions',
            'avg_session_duration_min', 'self_service_transactions'
        ]
        
        df_model = df.copy()
        le_plan = LabelEncoder()
        df_model['plan_encoded'] = le_plan.fit_transform(df_model['plan_type'].astype(str))
        le_gender = LabelEncoder()
        df_model['gender_encoded'] = le_gender.fit_transform(df_model['gender'].astype(str))
        
        feature_cols_full = feature_cols + ['plan_encoded', 'gender_encoded']
        self.feature_names = feature_cols_full
        
        # Readable names for display
        self.display_names = {
            'age': 'Age', 'annual_income': 'Annual Income', 
            'tenure_months': 'Tenure (Months)', 'monthly_data_gb': 'Monthly Data (GB)',
            'monthly_minutes': 'Monthly Minutes', 'ott_usage_hours': 'OTT Usage (Hours)',
            'arpu': 'ARPU ($)', 'data_allowance_gb': 'Data Allowance (GB)',
            'minutes_allowance': 'Minutes Allowance', 'satisfaction_score': 'Satisfaction Score',
            'num_complaints_12m': 'Complaints (12m)', 'support_tickets_12m': 'Support Tickets (12m)',
            'payment_score': 'Payment Score', 'late_payments_12m': 'Late Payments (12m)',
            'campaigns_exposed': 'Campaigns Exposed', 'total_clicks': 'Total Clicks',
            'total_conversions': 'Total Conversions', 'monthly_web_sessions': 'Web Sessions',
            'monthly_app_sessions': 'App Sessions', 'avg_session_duration_min': 'Avg Session Duration',
            'self_service_transactions': 'Self-Service Txns',
            'plan_encoded': 'Plan Type', 'gender_encoded': 'Gender'
        }
        
        X = df_model[feature_cols_full].fillna(0)
        y = df_model['churned']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        self.model.fit(X_train, y_train)
        
        y_proba = self.model.predict_proba(X_test)[:, 1]
        self.model_metrics['auc'] = roc_auc_score(y_test, y_proba)
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"  AUC-ROC: {self.model_metrics['auc']:.4f}")
        return self.model_metrics
    
    def compute_shap_values(self):
        """
        Compute SHAP values using tree-based exact computation.
        Falls back to permutation-based if SHAP library not available.
        """
        print("Computing feature importance values...")
        
        X_sample = self.X_test.head(500)  # Use subset for speed
        
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer.shap_values(X_sample)
            # For binary classification, shap_values might be a list
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]  # Class 1 (churn)
            self.X_explain = X_sample
            self.method = 'SHAP (TreeExplainer)'
            print(f"  Computed SHAP values for {len(X_sample)} customers using TreeExplainer")
        except ImportError:
            print("  SHAP library not available. Using permutation importance as fallback...")
            self._compute_permutation_importance(X_sample)
            self.method = 'Permutation Importance'
        
        return self.shap_values
    
    def _compute_permutation_importance(self, X_sample):
        """Fallback: compute permutation-based feature importance."""
        from sklearn.inspection import permutation_importance
        
        y_sample = self.y_test.head(len(X_sample))
        result = permutation_importance(
            self.model, X_sample, y_sample, 
            n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # Create pseudo-SHAP values from permutation importance
        importances = result.importances_mean
        importances = importances / (importances.sum() + 1e-10)
        
        # Distribute importance across samples proportionally to feature values
        X_normalized = (X_sample - X_sample.mean()) / (X_sample.std() + 1e-10)
        self.shap_values = X_normalized.values * importances[np.newaxis, :]
        self.X_explain = X_sample
        
        print(f"  Computed permutation importance for {len(X_sample)} customers")
    
    def get_global_importance(self):
        """Get global feature importance (mean absolute SHAP value)."""
        if self.shap_values is None:
            return pd.Series()
        
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        importance = pd.Series(
            mean_abs_shap, 
            index=[self.display_names.get(f, f) for f in self.feature_names]
        ).sort_values(ascending=False)
        
        return importance
    
    def get_feature_effects(self):
        """Get feature effect data for dependence-style analysis."""
        if self.shap_values is None:
            return pd.DataFrame()
        
        effects = pd.DataFrame(
            self.shap_values, 
            columns=[self.display_names.get(f, f) for f in self.feature_names]
        )
        
        feature_values = self.X_explain.copy()
        feature_values.columns = [self.display_names.get(f, f) for f in self.feature_names]
        
        return effects, feature_values
    
    def explain_customer(self, customer_idx=0):
        """Explain a single customer's prediction."""
        if self.shap_values is None:
            return {}
        
        customer_shap = self.shap_values[customer_idx]
        customer_features = self.X_explain.iloc[customer_idx]
        
        explanation = {}
        for i, feature in enumerate(self.feature_names):
            display = self.display_names.get(feature, feature)
            explanation[display] = {
                'value': customer_features.iloc[i],
                'shap_value': customer_shap[i],
                'direction': 'increases churn' if customer_shap[i] > 0 else 'decreases churn'
            }
        
        # Sort by absolute SHAP value
        explanation = dict(sorted(
            explanation.items(), 
            key=lambda x: abs(x[1]['shap_value']), 
            reverse=True
        ))
        
        return explanation
    
    def save_results(self, output_dir='data/processed'):
        """Save SHAP analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save global importance
        importance = self.get_global_importance()
        importance.to_csv(os.path.join(output_dir, 'shap_importance.csv'))
        
        # Save SHAP values matrix
        if self.shap_values is not None:
            shap_df = pd.DataFrame(
                self.shap_values,
                columns=[self.display_names.get(f, f) for f in self.feature_names]
            )
            shap_df.to_csv(os.path.join(output_dir, 'shap_values.csv'), index=False)
        
        print(f"  Saved SHAP results to {output_dir}")


def run_explainability_analysis(data_path='data/raw/master_dataset.csv'):
    """Run the complete explainability pipeline."""
    print("=" * 60)
    print("MODEL EXPLAINABILITY ENGINE")
    print("Understanding WHY the model makes its predictions")
    print("=" * 60)
    
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} customers")
    
    engine = ExplainabilityEngine()
    engine.train_model(df)
    engine.compute_shap_values()
    engine.save_results()
    
    # Print top features
    importance = engine.get_global_importance()
    print(f"\n  Top 10 Features Driving Churn:")
    for feat, val in importance.head(10).items():
        print(f"    {feat}: {val:.4f}")
    
    print(f"\n✅ Explainability analysis complete!")
    return engine


if __name__ == "__main__":
    engine = run_explainability_analysis()
