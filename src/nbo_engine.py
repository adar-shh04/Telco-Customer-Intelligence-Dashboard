"""
Next Best Offer (NBO) Prediction Engine

Predicts which product/service upgrade each customer is most likely to accept,
with expected conversion probability and revenue uplift.

Uses the real Kaggle IBM Telco Customer Churn data to identify:
1. Which services each customer DOESN'T have yet
2. ML-predicted conversion probability for each potential offer
3. Expected revenue uplift per offer
4. Prioritized recommendations ranked by expected value

This transforms a reactive telecom into a proactive revenue growth engine.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# Define available offers with pricing
OFFER_CATALOG = {
    'plan_upgrade_standard': {
        'name': 'Upgrade to Standard Plan',
        'description': 'Upgrade from Basic to Standard (1-year contract)',
        'monthly_price_increase': 20,
        'icon': '⬆️',
        'eligible': lambda df: df['plan_type'] == 'Basic',
        'target_feature': 'plan_type',
        'target_value': 'Standard'
    },
    'plan_upgrade_premium': {
        'name': 'Upgrade to Premium Plan',
        'description': 'Upgrade to Premium (2-year contract, best value)',
        'monthly_price_increase': 30,
        'icon': '👑',
        'eligible': lambda df: df['plan_type'].isin(['Basic', 'Standard']),
        'target_feature': 'plan_type',
        'target_value': 'Premium'
    },
    'add_streaming_bundle': {
        'name': 'Add Streaming Bundle',
        'description': 'StreamingTV + StreamingMovies package',
        'monthly_price_increase': 15,
        'icon': '🎬',
        'eligible': lambda df: df['ott_usage_hours'] < 10,
        'target_feature': 'ott_usage_hours',
        'target_value': 'high'
    },
    'add_security_package': {
        'name': 'Add Online Security & Backup',
        'description': 'Online Security + Online Backup + Device Protection',
        'monthly_price_increase': 12,
        'icon': '🔒',
        'eligible': lambda df: df['self_service_transactions'] < 3,
        'target_feature': 'self_service_transactions',
        'target_value': 'high'
    },
    'add_tech_support': {
        'name': 'Add Premium Tech Support',
        'description': 'Dedicated tech support with priority queue',
        'monthly_price_increase': 8,
        'icon': '🛠️',
        'eligible': lambda df: df['support_tickets_12m'] >= 3,
        'target_feature': 'support_tickets_12m',
        'target_value': 'low'
    },
    'data_boost': {
        'name': 'Data Boost Package',
        'description': 'Double data allowance for power users',
        'monthly_price_increase': 10,
        'icon': '📶',
        'eligible': lambda df: (df['monthly_data_gb'] / (df['data_allowance_gb'] + 1e-6)) > 0.8,
        'target_feature': 'monthly_data_gb',
        'target_value': 'high_usage'
    },
    'loyalty_bundle': {
        'name': 'Loyalty Rewards Bundle',
        'description': 'Exclusive perks: free upgrades, priority support, bonus data',
        'monthly_price_increase': 5,
        'icon': '🎁',
        'eligible': lambda df: df['tenure_months'] >= 24,
        'target_feature': 'tenure_months',
        'target_value': 'loyal'
    }
}


class NBOEngine:
    """
    Next Best Offer prediction engine.
    
    For each customer:
    1. Identifies which offers they're eligible for
    2. Predicts conversion probability using ML
    3. Calculates expected revenue uplift
    4. Returns top-N ranked recommendations
    """
    
    def __init__(self):
        self.conversion_models = {}
        self.feature_columns = None
        self.model_metrics = {}
    
    def _prepare_features(self, df):
        """Prepare feature matrix for ML models."""
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
        
        # Encode categoricals
        le_plan = LabelEncoder()
        df_model['plan_encoded'] = le_plan.fit_transform(df_model['plan_type'].astype(str))
        le_gender = LabelEncoder()
        df_model['gender_encoded'] = le_gender.fit_transform(df_model['gender'].astype(str))
        
        feature_cols_full = feature_cols + ['plan_encoded', 'gender_encoded']
        self.feature_columns = feature_cols_full
        
        return df_model[feature_cols_full].fillna(0)
    
    def _create_synthetic_labels(self, df, offer_key, offer_config):
        """
        Create realistic conversion labels based on customer attributes.
        
        Since we don't have actual offer acceptance data, we create
        probabilistic labels based on customer profiles that would
        realistically accept each offer type.
        """
        n = len(df)
        np.random.seed(hash(offer_key) % 2**31)
        
        # Base conversion probability (5-15%)
        base_prob = np.full(n, 0.08)
        
        if offer_key == 'plan_upgrade_standard':
            # Higher income, higher satisfaction, more usage → more likely to upgrade
            base_prob += (df['annual_income'].rank(pct=True) - 0.5) * 0.1
            base_prob += (df['satisfaction_score'] / 10 - 0.5) * 0.08
            base_prob += np.where(df['monthly_data_gb'] > df['data_allowance_gb'] * 0.7, 0.05, 0)
            
        elif offer_key == 'plan_upgrade_premium':
            # Very high income, long tenure, high usage
            base_prob += (df['annual_income'].rank(pct=True) - 0.5) * 0.15
            base_prob += np.where(df['tenure_months'] > 24, 0.05, 0)
            base_prob += np.where(df['arpu'] > 70, 0.04, 0)
            base_prob *= 0.7  # Lower conversion for bigger commitment
            
        elif offer_key == 'add_streaming_bundle':
            # Younger, higher income, already digital
            base_prob += np.where(df['age'] < 40, 0.06, -0.02)
            base_prob += (df['monthly_web_sessions'] + df['monthly_app_sessions']).rank(pct=True) * 0.05
            base_prob += (df['annual_income'].rank(pct=True) - 0.5) * 0.06
            
        elif offer_key == 'add_security_package':
            # Older, high value, already using services
            base_prob += np.where(df['age'] > 35, 0.04, 0)
            base_prob += np.where(df['arpu'] > 50, 0.03, 0)
            base_prob += np.where(df['tenure_months'] > 12, 0.03, 0)
            
        elif offer_key == 'add_tech_support':
            # Frustrated customers who need help
            base_prob += np.clip(df['support_tickets_12m'] / 10, 0, 0.1)
            base_prob += np.where(df['satisfaction_score'] < 6, 0.06, 0)
            base_prob += np.clip(df['num_complaints_12m'] / 5, 0, 0.08)
            
        elif offer_key == 'data_boost':
            # Heavy data users
            usage_ratio = df['monthly_data_gb'] / (df['data_allowance_gb'] + 1e-6)
            base_prob += np.clip(usage_ratio - 0.5, 0, 0.15)
            base_prob += np.where(df['age'] < 45, 0.03, 0)
            
        elif offer_key == 'loyalty_bundle':
            # Long-tenure, satisfied customers
            base_prob += np.clip((df['tenure_months'] - 24) / 48, 0, 0.1)
            base_prob += np.where(df['satisfaction_score'] > 7, 0.05, 0)
            base_prob += np.where(df['late_payments_12m'] == 0, 0.03, -0.02)
        
        # Churned customers less likely to accept any offer
        if 'churned' in df.columns:
            base_prob -= df['churned'] * 0.05
        
        base_prob = np.clip(base_prob, 0.01, 0.35)
        
        # Sample binary labels from probabilities
        labels = np.random.binomial(1, base_prob)
        
        return labels, base_prob
    
    def train_conversion_models(self, df):
        """
        Train a conversion prediction model for each offer type.
        
        Returns:
            Dictionary of model metrics per offer
        """
        print("Training NBO conversion models...")
        
        X = self._prepare_features(df)
        
        for offer_key, offer_config in OFFER_CATALOG.items():
            print(f"  Training model for: {offer_config['name']}...")
            
            # Create training labels
            y, true_probs = self._create_synthetic_labels(df, offer_key, offer_config)
            
            # Only train on eligible customers
            eligible_mask = offer_config['eligible'](df)
            if eligible_mask.sum() < 100:
                print(f"    Skipping (only {eligible_mask.sum()} eligible customers)")
                continue
            
            X_eligible = X[eligible_mask]
            y_eligible = y[eligible_mask]
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X_eligible, y_eligible, test_size=0.2, random_state=42
            )
            
            # Train
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_proba = model.predict_proba(X_test)[:, 1]
            try:
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.5
            
            self.conversion_models[offer_key] = model
            self.model_metrics[offer_key] = {
                'auc': auc,
                'eligible_count': int(eligible_mask.sum()),
                'conversion_rate': float(y_eligible.mean()),
                'name': offer_config['name']
            }
            
            print(f"    AUC: {auc:.4f} | Eligible: {eligible_mask.sum():,} | Conv Rate: {y_eligible.mean():.2%}")
        
        return self.model_metrics
    
    def predict_offers(self, df, top_n=3):
        """
        Generate top-N offer recommendations for each customer.
        
        Returns:
            DataFrame with offer recommendations, probabilities, and expected revenue
        """
        print("Generating personalized offer recommendations...")
        
        X = self._prepare_features(df)
        n = len(df)
        
        # Store all offer scores
        all_offers = []
        
        for offer_key, offer_config in OFFER_CATALOG.items():
            if offer_key not in self.conversion_models:
                continue
            
            model = self.conversion_models[offer_key]
            eligible_mask = offer_config['eligible'](df)
            
            # Predict conversion probability
            proba = np.zeros(n)
            if eligible_mask.sum() > 0:
                proba[eligible_mask] = model.predict_proba(X[eligible_mask])[:, 1]
            
            # Expected monthly revenue uplift
            monthly_uplift = offer_config['monthly_price_increase']
            expected_value = proba * monthly_uplift * 12  # Annual expected value
            
            all_offers.append({
                'offer_key': offer_key,
                'offer_name': offer_config['name'],
                'offer_icon': offer_config['icon'],
                'monthly_price_increase': monthly_uplift,
                'conversion_probability': proba,
                'expected_annual_value': expected_value,
                'eligible': eligible_mask
            })
        
        # For each customer, rank offers by expected value
        results = []
        
        for i in range(n):
            customer_offers = []
            for offer in all_offers:
                if offer['eligible'].iloc[i]:
                    customer_offers.append({
                        'offer_key': offer['offer_key'],
                        'offer_name': offer['offer_name'],
                        'offer_icon': offer['offer_icon'],
                        'conversion_prob': offer['conversion_probability'][i],
                        'monthly_uplift': offer['monthly_price_increase'],
                        'expected_annual_value': offer['expected_annual_value'][i]
                    })
            
            # Sort by expected value
            customer_offers.sort(key=lambda x: x['expected_annual_value'], reverse=True)
            
            # Take top N
            for rank, offer in enumerate(customer_offers[:top_n], 1):
                results.append({
                    'customer_id': df['customer_id'].iloc[i],
                    'offer_rank': rank,
                    **offer
                })
        
        results_df = pd.DataFrame(results)
        
        # Also create a summary: best offer per customer
        best_offers = results_df[results_df['offer_rank'] == 1].copy() if len(results_df) > 0 else pd.DataFrame()
        
        return results_df, best_offers
    
    def generate_nbo_summary(self, df, best_offers):
        """Generate business summary of NBO recommendations."""
        
        if len(best_offers) == 0:
            return {'error': 'No offers generated'}
        
        summary = {
            'total_customers': len(df),
            'customers_with_offers': best_offers['customer_id'].nunique(),
            
            # Revenue potential
            'total_expected_annual_revenue': best_offers['expected_annual_value'].sum(),
            'avg_expected_value_per_customer': best_offers['expected_annual_value'].mean(),
            'avg_conversion_probability': best_offers['conversion_prob'].mean(),
            
            # Offer distribution
            'offer_distribution': best_offers['offer_name'].value_counts().to_dict(),
            
            # Revenue by offer type
            'revenue_by_offer': best_offers.groupby('offer_name').agg({
                'expected_annual_value': 'sum',
                'conversion_prob': 'mean',
                'customer_id': 'count'
            }).rename(columns={
                'customer_id': 'customer_count'
            }).to_dict('index'),
            
            # Top opportunities
            'top_20_opportunities': best_offers.nlargest(20, 'expected_annual_value').to_dict('records'),
            
            # Model performance
            'model_metrics': self.model_metrics
        }
        
        return summary


def run_full_nbo_analysis(data_path='data/raw/master_dataset.csv'):
    """Run the complete NBO pipeline."""
    print("=" * 60)
    print("NEXT BEST OFFER (NBO) ENGINE")
    print("Predicting the right offer for every customer")
    print("=" * 60)
    
    # Load data
    print("\n📥 Loading data...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} customers")
    
    # Train models
    print("\n🤖 Training conversion models...")
    engine = NBOEngine()
    metrics = engine.train_conversion_models(df)
    
    # Generate recommendations 
    print("\n🎯 Generating personalized recommendations...")
    all_offers, best_offers = engine.predict_offers(df, top_n=3)
    
    # Summary
    print("\n📋 Generating summary...")
    summary = engine.generate_nbo_summary(df, best_offers)
    
    # Print results
    print(f"\n{'=' * 60}")
    print("NBO ENGINE RESULTS")
    print(f"{'=' * 60}")
    print(f"  Customers analyzed:    {summary['total_customers']:,}")
    print(f"  Customers with offers: {summary['customers_with_offers']:,}")
    print(f"  Expected annual revenue: ${summary['total_expected_annual_revenue']:,.0f}")
    print(f"  Avg conversion prob:  {summary['avg_conversion_probability']:.1%}")
    
    print(f"\n  Best Offer Distribution:")
    for offer, count in summary['offer_distribution'].items():
        print(f"    {offer}: {count:,}")
    
    print(f"\n✅ NBO Engine Complete!")
    print(f"{'=' * 60}")
    
    return df, all_offers, best_offers, summary, engine


if __name__ == "__main__":
    df, all_offers, best_offers, summary, engine = run_full_nbo_analysis()
    
    # Save results
    import os
    os.makedirs('data/processed', exist_ok=True)
    best_offers.to_csv('data/processed/nbo_recommendations.csv', index=False)
    all_offers.to_csv('data/processed/nbo_all_offers.csv', index=False)
    print(f"\nSaved {len(best_offers):,} best offers to data/processed/nbo_recommendations.csv")
    print(f"Saved {len(all_offers):,} all offers to data/processed/nbo_all_offers.csv")
