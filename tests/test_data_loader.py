import unittest
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import TelecomDataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for Kaggle data loading and transformation."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = TelecomDataLoader()

    def test_loader_initialization(self):
        """Test if data loader initializes correctly."""
        self.assertIsNotNone(self.loader)
        self.assertEqual(len(self.loader.cities), 10)
        self.assertEqual(len(self.loader.plan_mapping), 3)

    def test_plan_mapping(self):
        """Test contract to plan type mapping."""
        self.assertEqual(self.loader.plan_mapping['Month-to-month'], 'Basic')
        self.assertEqual(self.loader.plan_mapping['One year'], 'Standard')
        self.assertEqual(self.loader.plan_mapping['Two year'], 'Premium')

    def test_full_pipeline(self):
        """Test the complete loading pipeline if data is available."""
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
        kaggle_csv = os.path.join(data_dir, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

        if not os.path.exists(kaggle_csv):
            self.skipTest("Kaggle dataset not downloaded yet")

        datasets = self.loader.load_and_transform(data_dir)

        # Check all expected dataset components exist
        expected_keys = ['master_dataset', 'demographics', 'usage_billing',
                         'crm_data', 'campaign_data', 'engagement_data', 'churn_data']
        for key in expected_keys:
            self.assertIn(key, datasets, f"Missing dataset: {key}")

        master = datasets['master_dataset']

        # Check shape (~7043 customers)
        self.assertGreater(len(master), 7000)
        self.assertLess(len(master), 7100)

        # Check all required columns exist
        required_cols = [
            'customer_id', 'age', 'gender', 'city', 'annual_income',
            'tenure_months', 'join_date', 'plan_type', 'monthly_data_gb',
            'monthly_minutes', 'ott_usage_hours', 'arpu', 'data_allowance_gb',
            'minutes_allowance', 'satisfaction_score', 'num_complaints_12m',
            'support_tickets_12m', 'payment_score', 'late_payments_12m',
            'campaigns_exposed', 'total_impressions', 'total_clicks',
            'total_conversions', 'primary_channel', 'monthly_web_sessions',
            'monthly_app_sessions', 'avg_session_duration_min',
            'self_service_transactions', 'churned', 'churn_probability'
        ]
        for col in required_cols:
            self.assertIn(col, master.columns, f"Missing column: {col}")

        # Check value ranges
        self.assertTrue(master['age'].min() >= 18)
        self.assertTrue(master['age'].max() <= 80)
        self.assertTrue(master['satisfaction_score'].min() >= 1.0)
        self.assertTrue(master['satisfaction_score'].max() <= 10.0)
        self.assertTrue(master['churned'].isin([0, 1]).all())
        self.assertTrue(master['arpu'].min() >= 0)

    def test_demographics_columns(self):
        """Test demographics generation with mock data."""
        mock_df = pd.DataFrame({
            'customerID': ['CUST001', 'CUST002', 'CUST003'],
            'gender': ['Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0],
            'tenure': [12, 48, 1],
            'MonthlyCharges': [50.0, 80.0, 30.0],
            'TotalCharges': [600.0, 3840.0, 30.0],
            'Churn': ['No', 'No', 'Yes'],
            'Churn_binary': [0, 0, 1],
            'services_count': [3, 6, 1],
            'Contract': ['Month-to-month', 'Two year', 'Month-to-month'],
            'PhoneService': ['Yes', 'Yes', 'No'],
            'MultipleLines': ['No', 'Yes', 'No phone service'],
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'TechSupport': ['No', 'Yes', 'No'],
            'StreamingTV': ['No', 'Yes', 'No'],
            'StreamingMovies': ['No', 'Yes', 'No'],
            'PaperlessBilling': ['Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Bank transfer (automatic)', 'Mailed check'],
            'OnlineSecurity': ['No', 'Yes', 'No'],
            'OnlineBackup': ['No', 'Yes', 'No'],
            'DeviceProtection': ['No', 'Yes', 'No'],
            'Partner': ['Yes', 'No', 'No'],
            'Dependents': ['No', 'No', 'No']
        })

        demographics = self.loader.generate_customer_demographics(mock_df)
        self.assertEqual(len(demographics), 3)
        self.assertIn('customer_id', demographics.columns)
        self.assertIn('age', demographics.columns)
        self.assertTrue(demographics['age'].min() >= 18)


if __name__ == '__main__':
    unittest.main()
