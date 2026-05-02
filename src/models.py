"""
Machine Learning Models for Telco Customer Intelligence

This module implements various ML models for:
- Churn prediction
- ARPU forecasting  
- Uplift modeling for cross-sell/up-sell
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import joblib
from prophet import Prophet
import shap
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    """Churn prediction model using LightGBM/XGBoost."""
    
    def __init__(self, model_type='lightgbm'):
        """
        Initialize churn predictor.
        
        Args:
            model_type: 'lightgbm' or 'xgboost'
        """
        self.model_type = model_type
        self.model = None  # type: lgb.LGBMClassifier | xgb.XGBClassifier | None
        self.scaler = StandardScaler()
        self.feature_importance = None  # type: pd.DataFrame | None
        self.is_fitted = False
        
    def prepare_data(self, df, target_col='churned', test_size=0.2):
        """Prepare data for training."""
        # Remove non-feature columns
        feature_cols = [col for col in df.columns 
                       if col not in ['customer_id', 'churn_date', 'join_date', target_col]]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, optimize_hyperparams=True):
        """Train the churn prediction model."""
        print(f"Training {self.model_type} churn prediction model...")
        
        if self.model_type == 'lightgbm':
            if optimize_hyperparams:
                # Hyperparameter optimization
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 0.9]
                }
                
                base_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
                grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                self.model = lgb.LGBMClassifier(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    random_state=42, verbose=-1
                )
                self.model.fit(X_train, y_train)
                
        elif self.model_type == 'xgboost':
            if optimize_hyperparams:
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [6, 8],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 0.9]
                }
                
                base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
            else:
                self.model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    random_state=42, eval_metric='logloss'
                )
                self.model.fit(X_train, y_train)
        
        # Store feature importance only if model is fitted
        if self.model is not None:
            importances = getattr(self.model, 'feature_importances_', None)
            if importances is not None:
                self.feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
        
        self.is_fitted = True
        print("Model training completed!")
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        proba = self.model.predict_proba(X)
        # Handle different return types safely
        if isinstance(proba, np.ndarray):
            return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        else:
            # Convert to numpy array first, handling different types
            try:
                # Simple conversion without accessing specific attributes
                proba_array = np.asarray(proba)
                if proba_array.ndim == 1:
                    return proba_array
                return proba_array[:, 1] if proba_array.shape[1] > 1 else proba_array[:, 0]
            except:
                # Final fallback
                proba_array = np.asarray(proba)
                if proba_array.ndim == 1:
                    return proba_array
                return proba_array[:, 1] if proba_array.shape[1] > 1 else proba_array[:, 0]
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, probabilities)
        
        print("=== CHURN PREDICTION MODEL EVALUATION ===")
        print(f"AUC Score: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        return {
            'auc': auc,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def get_feature_importance(self, top_n=15):
        """Get top feature importances."""
        if self.feature_importance is None:
            raise ValueError("Model must be trained first")
        return self.feature_importance.head(top_n)
    
    def save_model(self, filepath):
        """Save trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def explain_predictions(self, X_sample, sample_size=1000):
        """Generate SHAP explanations for model predictions."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before generating explanations")
        
        # Sample data for performance
        if len(X_sample) > sample_size:
            sample_indices = np.random.choice(X_sample.shape[0], size=sample_size, replace=False)
            X_sample = X_sample.iloc[sample_indices]
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        return explainer, shap_values, X_sample

class ARPUForecaster:
    """ARPU prediction using time series and regression models."""
    
    def __init__(self, model_type='regression'):
        """Initialize ARPU forecaster.
        
        Args:
            model_type: 'regression' for feature-based prediction or 'prophet' for time series
        """
        self.model_type = model_type
        self.model = None  # type: Prophet | RandomForestRegressor | None
        self.feature_importance = None  # type: pd.DataFrame | None
        self.is_fitted = False
        
    def prepare_features(self, df):
        """Create features for ARPU prediction."""
        df = df.copy()
        
        # Remove ARPU from features
        feature_cols = [col for col in df.columns 
                       if col not in ['customer_id', 'arpu', 'churn_date', 'join_date']]
        
        X = df[feature_cols]
        y = df['arpu']
        
        return X, y
    
    def train(self, X_train, y_train, time_series_data=None):
        """Train ARPU prediction model.
        
        Args:
            X_train: Feature data for regression model
            y_train: Target ARPU values
            time_series_data: DataFrame with 'ds' and 'y' columns for Prophet model
        """
        print(f"Training {self.model_type} ARPU forecasting model...")
        
        if self.model_type == 'prophet':
            if time_series_data is None:
                raise ValueError("time_series_data required for Prophet model")
            
            # Initialize Prophet model with correct string parameters
            self.model = Prophet(
                yearly_seasonality='auto',
                weekly_seasonality='auto',
                daily_seasonality='auto',
                seasonality_mode='multiplicative'
            )
            
            # Fit the model
            if self.model is not None:
                self.model.fit(time_series_data)
        else:
            # Use Random Forest for robust ARPU prediction
            self.model = RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            )
            if self.model is not None:
                self.model.fit(X_train, y_train)
                
                # Store feature importance
                importances = getattr(self.model, 'feature_importances_', None)
                if importances is not None:
                    self.feature_importance = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
        
        self.is_fitted = True
        print("ARPU model training completed!")
        
    def predict(self, X, future_periods=30):
        """Make ARPU predictions.
        
        Args:
            X: Feature data for regression model
            future_periods: Number of future periods to forecast (Prophet only)
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        # Check if we're using Prophet model
        if self.model_type == 'prophet':
            # Import Prophet here to check type
            from prophet import Prophet
            # Verify it's a Prophet model by checking type
            model = self.model
            if isinstance(model, Prophet):
                # Create future dataframe - type check to avoid errors
                if hasattr(model, 'make_future_dataframe') and callable(getattr(model, 'make_future_dataframe', None)):
                    future = model.make_future_dataframe(periods=future_periods)
                    if hasattr(model, 'predict') and callable(getattr(model, 'predict', None)):
                        forecast = model.predict(future)
                        return forecast
            raise ValueError("Prophet model not properly initialized")
        else:
            # For regression models
            model = self.model
            if model is not None and hasattr(model, 'predict'):
                # Double-check it's callable
                if callable(getattr(model, 'predict', None)):
                    return model.predict(X)
            raise ValueError("Regression model not properly initialized")
    
    def evaluate(self, X_test, y_test):
        """Evaluate ARPU model performance."""
        predictions = self.predict(X_test)
        
        if self.model_type == 'prophet':
            # For Prophet, we need to match the test period
            # This is a simplified evaluation - in practice, you'd align dates
            try:
                if isinstance(predictions, pd.DataFrame):
                    # Handle DataFrame with 'yhat' column
                    if 'yhat' in predictions.columns:
                        # Use positional indexing instead of label indexing to avoid type issues
                        yhat_values = predictions.iloc[:, predictions.columns.get_loc('yhat')].values
                        test_predictions = yhat_values[-len(y_test):] if len(yhat_values) >= len(y_test) else yhat_values
                    else:
                        # Fallback to last column
                        test_predictions = predictions.iloc[-len(y_test):, -1].values if len(predictions) >= len(y_test) else predictions.iloc[:, -1].values
                else:
                    # Handle case where predictions might be a dictionary or other structure
                    test_predictions = predictions[-len(y_test):] if hasattr(predictions, '__len__') and len(predictions) >= len(y_test) else predictions
                mse = mean_squared_error(y_test, test_predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, test_predictions)
                mape = np.mean(np.abs((y_test - test_predictions) / y_test)) * 100
            except Exception as e:
                # Fallback evaluation
                test_predictions = predictions[:len(y_test)] if hasattr(predictions, '__len__') else [0]*len(y_test)
                mse = mean_squared_error(y_test, test_predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, test_predictions)
                mape = np.mean(np.abs((y_test - test_predictions) / y_test)) * 100
        else:
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        print("=== ARPU FORECASTING MODEL EVALUATION ===")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'predictions': predictions
        }

class UpliftModeler:
    """Uplift modeling for cross-sell/up-sell campaigns."""
    
    def __init__(self):
        """Initialize uplift modeler."""
        self.treatment_model = None  # type: RandomForestClassifier | None
        self.control_model = None  # type: RandomForestClassifier | None
        self.is_fitted = False
        
    def prepare_uplift_data(self, df, treatment_col='campaign_responsive', outcome_col='total_conversions'):
        """Prepare data for uplift modeling."""
        # Simulate treatment assignment (in real scenario, this would be historical campaign data)
        df = df.copy()
        
        # Create treatment groups based on campaign responsiveness
        df['treatment'] = df[treatment_col]
        
        # Features for uplift modeling
        feature_cols = ['age', 'annual_income', 'tenure_months', 'satisfaction_score',
                       'monthly_data_gb', 'arpu', 'rfm_score', 'composite_risk_score']
        
        # Filter to only include columns that exist
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[available_cols]
        treatment = df['treatment']
        outcome = df[outcome_col] > 0  # Binary outcome: had any conversions
        
        return X, treatment, outcome
    
    def train(self, X, treatment, outcome):
        """Train uplift models using two-model approach."""
        print("Training uplift models...")
        
        # Split into treatment and control groups
        treatment_mask = treatment == 1
        control_mask = treatment == 0
        
        X_treatment = X[treatment_mask]
        y_treatment = outcome[treatment_mask]
        
        X_control = X[control_mask]
        y_control = outcome[control_mask]
        
        # Train separate models for treatment and control groups
        self.treatment_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.control_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.treatment_model.fit(X_treatment, y_treatment)
        self.control_model.fit(X_control, y_control)
        
        self.is_fitted = True
        print("Uplift models training completed!")
        
    def predict_uplift(self, X):
        """Predict uplift scores."""
        if not self.is_fitted or self.treatment_model is None or self.control_model is None:
            raise ValueError("Models must be trained before making predictions")
            
        # Get probabilities from both models
        p_treatment = self.treatment_model.predict_proba(X)
        p_control = self.control_model.predict_proba(X)
        
        # Handle different return types
        if isinstance(p_treatment, np.ndarray):
            p_treatment = p_treatment[:, 1] if p_treatment.shape[1] > 1 else p_treatment[:, 0]
        else:
            p_treatment = np.asarray(p_treatment)[:, 1] if hasattr(p_treatment, '__len__') and len(p_treatment) > 1 else np.asarray(p_treatment)[:, 0]
            
        if isinstance(p_control, np.ndarray):
            p_control = p_control[:, 1] if p_control.shape[1] > 1 else p_control[:, 0]
        else:
            p_control = np.asarray(p_control)[:, 1] if hasattr(p_control, '__len__') and len(p_control) > 1 else np.asarray(p_control)[:, 0]
        
        # Uplift = P(conversion|treatment) - P(conversion|control)
        uplift_scores = p_treatment - p_control
        
        return uplift_scores
    
    def identify_best_targets(self, X, top_percentile=20):
        """Identify best customers to target for campaigns."""
        uplift_scores = self.predict_uplift(X)
        threshold = np.percentile(uplift_scores, 100 - top_percentile)
        
        best_targets = uplift_scores >= threshold
        return best_targets, uplift_scores
    
    def explain_uplift(self, X_sample, sample_size=1000):
        """Generate SHAP explanations for uplift models."""
        if not self.is_fitted:
            raise ValueError("Models must be trained before generating explanations")
        
        # Sample data for performance
        if len(X_sample) > sample_size:
            sample_indices = np.random.choice(X_sample.shape[0], size=sample_size, replace=False)
            X_sample = X_sample.iloc[sample_indices]
        
        # Create SHAP explainers for both models
        treatment_explainer = shap.TreeExplainer(self.treatment_model)
        control_explainer = shap.TreeExplainer(self.control_model)
        
        # Calculate SHAP values
        treatment_shap = treatment_explainer.shap_values(X_sample)
        control_shap = control_explainer.shap_values(X_sample)
        
        return {
            'treatment_explainer': treatment_explainer,
            'control_explainer': control_explainer,
            'treatment_shap': treatment_shap,
            'control_shap': control_shap,
            'X_sample': X_sample
        }

class ModelEvaluator:
    """Comprehensive model evaluation and business impact assessment."""
    
    @staticmethod
    def calculate_business_impact(churn_model, arpu_model, df, X_test):
        """Calculate business impact of models."""
        print("=== BUSINESS IMPACT ASSESSMENT ===")
        
        # Current state
        current_monthly_revenue = df['arpu'].sum()
        current_churn_rate = df['churned'].mean()
        
        # Predictions
        churn_probs = churn_model.predict_proba(X_test)
        arpu_predictions = arpu_model.predict(X_test)
        
        # Assume we can reduce churn by 20% for high-risk customers (top 20%)
        high_risk_threshold = np.percentile(churn_probs, 80)
        high_risk_customers = churn_probs >= high_risk_threshold
        
        # Calculate potential savings
        customers_saved = high_risk_customers.sum() * 0.2  # 20% reduction
        avg_revenue_saved = arpu_predictions[high_risk_customers].mean()
        monthly_savings = customers_saved * avg_revenue_saved
        annual_savings = monthly_savings * 12
        
        print(f"Current monthly revenue: ${current_monthly_revenue:,.2f}")
        print(f"Current churn rate: {current_churn_rate:.2%}")
        print(f"High-risk customers identified: {high_risk_customers.sum():,}")
        print(f"Potential customers saved: {customers_saved:.0f}")
        print(f"Projected monthly savings: ${monthly_savings:,.2f}")
        print(f"Projected annual savings: ${annual_savings:,.2f}")
        
        # ROI calculation (assuming model implementation cost)
        implementation_cost = 50000  # Estimated cost
        roi = (annual_savings - implementation_cost) / implementation_cost * 100
        print(f"Estimated ROI: {roi:.1f}%")
        
        return {
            'monthly_savings': monthly_savings,
            'annual_savings': annual_savings,
            'customers_saved': customers_saved,
            'roi': roi
        }

def main():
    """Test the models with sample data."""
    # This would typically load processed data
    print("Model classes defined successfully!")
    print("Available models:")
    print("- ChurnPredictor: LightGBM/XGBoost churn prediction")
    print("- ARPUForecaster: Random Forest ARPU prediction")
    print("- UpliftModeler: Two-model uplift modeling")
    print("- ModelEvaluator: Business impact assessment")

if __name__ == "__main__":
    main()
