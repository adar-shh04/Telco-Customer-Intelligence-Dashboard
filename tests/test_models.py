import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import ChurnPredictor
from churn_pipeline import FEATURE_COLUMNS, TARGET_COL, build_pipeline

class TestModels(unittest.TestCase):
    """Test cases for machine learning models."""
    
    def setUp(self):
        """Create synthetic data for testing models."""
        # Create minimal dummy data
        self.X = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'customer_id': range(100),
            'churn_date': [None]*100,
            'join_date': ['2023-01-01']*100
        })
        self.y = pd.Series(np.random.randint(0, 2, 100))
        
    def test_churn_predictor_init(self):
        """Test model initialization."""
        model = ChurnPredictor(model_type='lightgbm')
        self.assertIsNotNone(model)
        self.assertFalse(model.is_fitted)
        
    def test_training_flow(self):
        """Test if model can train without error."""
        # We mock the actual training to avoid dependency overhead in simple unit tests
        # or use a very simple dataset if we want integration testing.
        # Here we just check the class structure.
        model = ChurnPredictor()
        self.assertTrue(hasattr(model, 'train'))
        self.assertTrue(hasattr(model, 'predict'))

    def test_api_churn_pipeline_contract(self):
        """Check that the API-facing churn pipeline has a stable feature contract."""
        self.assertIn('plan_type', FEATURE_COLUMNS)
        self.assertIn('gender', FEATURE_COLUMNS)
        self.assertIn('payment_score', FEATURE_COLUMNS)
        self.assertEqual(TARGET_COL, 'churned')

        pipeline = build_pipeline()
        self.assertTrue(hasattr(pipeline, 'fit'))
        self.assertTrue(hasattr(pipeline, 'predict_proba'))

if __name__ == '__main__':
    unittest.main()
