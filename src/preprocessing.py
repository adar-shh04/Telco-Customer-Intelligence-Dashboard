"""
Data Preprocessing and Feature Engineering Module

This module contains functions for cleaning, preprocessing, and engineering features
for the telecom revenue optimization project.
Supports both real Kaggle data and synthetic data inputs.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TelecomPreprocessor:
    """Handles data preprocessing and feature engineering for telecom customer data."""
    
    def __init__(self):
        """Initialize preprocessor with scalers and encoders."""
        self.numeric_scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False
        
    def create_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) features.
        
        Args:
            df: Input dataframe with customer data
            
        Returns:
            DataFrame with added RFM features
        """
        df = df.copy()
        
        # Handle missing values in key columns
        for col in ['tenure_months', 'monthly_web_sessions', 'monthly_app_sessions',
                    'support_tickets_12m', 'arpu']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Recency: How recently the customer joined (inverse of tenure)
        df['recency_score'] = np.where(df['tenure_months'] <= 6, 5,
                              np.where(df['tenure_months'] <= 12, 4,
                              np.where(df['tenure_months'] <= 24, 3,
                              np.where(df['tenure_months'] <= 36, 2, 1))))
        
        # Frequency: Usage frequency based on sessions and support interactions
        total_sessions = df['monthly_web_sessions'] + df['monthly_app_sessions']
        try:
            frequency_scores = pd.qcut(total_sessions + df['support_tickets_12m'], 
                                     q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            df['frequency_score'] = pd.Series(frequency_scores).astype(int)
        except ValueError:
            df['frequency_score'] = pd.cut(total_sessions + df['support_tickets_12m'],
                                          bins=5, labels=[1, 2, 3, 4, 5]).astype(int).fillna(3)
        
        # Monetary: ARPU-based scoring
        try:
            monetary_scores = pd.qcut(df['arpu'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            df['monetary_score'] = pd.Series(monetary_scores).astype(int)
        except ValueError:
            df['monetary_score'] = pd.cut(df['arpu'], bins=5, labels=[1, 2, 3, 4, 5]).astype(int).fillna(3)
        
        # Combined RFM score
        df['rfm_score'] = df['recency_score'] + df['frequency_score'] + df['monetary_score']
        
        return df
    
    def create_usage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create usage-based features.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with usage features
        """
        df = df.copy()
        
        # Usage ratios
        df['data_usage_ratio'] = df['monthly_data_gb'] / (df['data_allowance_gb'] + 1e-6)
        df['minutes_usage_ratio'] = df['monthly_minutes'] / (df['minutes_allowance'] + 1e-6)
        
        # Overage indicators
        df['data_overage'] = (df['monthly_data_gb'] > df['data_allowance_gb']).astype(int)
        df['minutes_overage'] = (df['monthly_minutes'] > df['minutes_allowance']).astype(int)
        
        # Overall usage efficiency
        df['usage_efficiency'] = (df['data_usage_ratio'] + df['minutes_usage_ratio']) / 2
        
        # Digital engagement ratio
        total_digital = df['monthly_web_sessions'] + df['monthly_app_sessions'] + 1e-6
        df['app_preference_ratio'] = df['monthly_app_sessions'] / total_digital
        
        # OTT engagement level
        df['ott_engagement_level'] = pd.cut(df['ott_usage_hours'], 
                                           bins=[0, 5, 15, 30, float('inf')],
                                           labels=['Low', 'Medium', 'High', 'Very High'])
        
        return df
    
    def create_customer_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer segments based on value and behavior.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with customer segments
        """
        df = df.copy()
        
        # Value-based segmentation
        def assign_value_segment(row):
            if row['arpu'] >= 80 and row['satisfaction_score'] >= 8:
                return 'Champions'
            elif row['arpu'] >= 80 and row['satisfaction_score'] >= 6:
                return 'Loyal Customers'
            elif row['arpu'] >= 60 and row['satisfaction_score'] >= 7:
                return 'Potential Loyalists'
            elif row['arpu'] >= 40 and row['satisfaction_score'] >= 6:
                return 'New Customers'
            elif row['arpu'] >= 40 and row['satisfaction_score'] < 6:
                return 'At Risk'
            elif row['arpu'] < 40 and row['satisfaction_score'] >= 7:
                return 'Price Sensitive'
            else:
                return 'Need Attention'
        
        df['customer_segment'] = df.apply(assign_value_segment, axis=1)
        
        # Behavioral segmentation based on usage patterns
        def assign_usage_segment(row):
            if row['usage_efficiency'] > 0.8:
                return 'Heavy User'
            elif row['usage_efficiency'] > 0.4:
                return 'Moderate User'
            else:
                return 'Light User'
                
        df['usage_segment'] = df.apply(assign_usage_segment, axis=1)
        
        return df
    
    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk-based features for churn prediction.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with risk features
        """
        df = df.copy()
        
        # Satisfaction risk
        df['satisfaction_risk'] = np.where(df['satisfaction_score'] <= 5, 'High',
                                  np.where(df['satisfaction_score'] <= 7, 'Medium', 'Low'))
        
        # Payment risk
        df['payment_risk'] = np.where(df['late_payments_12m'] >= 3, 'High',
                             np.where(df['late_payments_12m'] >= 1, 'Medium', 'Low'))
        
        # Support burden
        df['support_burden'] = np.where(df['support_tickets_12m'] >= 5, 'High',
                               np.where(df['support_tickets_12m'] >= 2, 'Medium', 'Low'))
        
        # Tenure risk (newer customers more at risk)
        df['tenure_risk'] = np.where(df['tenure_months'] <= 6, 'High',
                            np.where(df['tenure_months'] <= 12, 'Medium', 'Low'))
        
        # Composite risk score
        risk_mapping = {'High': 3, 'Medium': 2, 'Low': 1}
        df['composite_risk_score'] = (df['satisfaction_risk'].replace(risk_mapping).astype(int) +
                                    df['payment_risk'].replace(risk_mapping).astype(int) +
                                    df['support_burden'].replace(risk_mapping).astype(int) +
                                    df['tenure_risk'].replace(risk_mapping).astype(int))
        
        return df
    
    def create_campaign_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create campaign effectiveness features.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with campaign features
        """
        df = df.copy()
        
        # Campaign engagement metrics
        df['campaign_ctr'] = df['total_clicks'] / (df['total_impressions'] + 1e-6)
        df['campaign_conversion_rate'] = df['total_conversions'] / (df['total_clicks'] + 1e-6)
        
        # Campaign responsiveness
        df['campaign_responsive'] = np.where(df['total_conversions'] > 0, 1, 0)
        
        # Channel preference
        digital_channels = ['Social_Media', 'Online_Display', 'Search_Ads', 'Email_Promotion']
        df['prefers_digital'] = df['primary_channel'].isin(digital_channels).astype(int)
        
        return df
    
    def create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age-based features.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with age features
        """
        df = df.copy()
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], 
                               bins=[0, 25, 35, 45, 55, 100],
                               labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        
        # Generation classification
        def assign_generation(age):
            if age <= 27:
                return 'Gen Z'
            elif age <= 42:
                return 'Millennial'
            elif age <= 57:
                return 'Gen X'
            else:
                return 'Boomer'
                
        df['generation'] = df['age'].apply(assign_generation)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, target_col: str = '') -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input dataframe
            target_col: Name of target column to exclude from encoding
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if target_col != '' and target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        # Remove ID columns
        id_cols = [col for col in categorical_cols if 'id' in col.lower()]
        categorical_cols = [col for col in categorical_cols if col not in id_cols]
        
        # One-hot encode categorical features
        for col in categorical_cols:
            if df[col].nunique() <= 10:  # One-hot encode if ≤ 10 categories
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)
            else:  # Label encode if > 10 categories
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_numeric_features(self, df: pd.DataFrame, target_col: str = '') -> pd.DataFrame:
        """
        Scale numeric features.
        
        Args:
            df: Input dataframe
            target_col: Name of target column to exclude from scaling
            
        Returns:
            DataFrame with scaled numeric features
        """
        df = df.copy()
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col != '' and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Remove ID and binary columns from scaling
        binary_cols = [col for col in numeric_cols if df[col].nunique() <= 2]
        scale_cols = [col for col in numeric_cols if col not in binary_cols]
        
        if scale_cols:
            if not self.is_fitted:
                df[scale_cols] = self.numeric_scaler.fit_transform(df[scale_cols])
                self.is_fitted = True
            else:
                df[scale_cols] = self.numeric_scaler.transform(df[scale_cols])
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Age-Income interaction
        if 'age' in df.columns and 'annual_income' in df.columns:
            df['age_income_interaction'] = df['age'] * df['annual_income'] / 1000
        
        # ARPU-Satisfaction interaction
        if 'arpu' in df.columns and 'satisfaction_score' in df.columns:
            df['arpu_satisfaction_interaction'] = df['arpu'] * df['satisfaction_score']
        
        # Usage-Tenure interaction
        if 'usage_efficiency' in df.columns and 'tenure_months' in df.columns:
            df['usage_tenure_interaction'] = df['usage_efficiency'] * df['tenure_months']
        
        return df
    
    def prepare_for_modeling(self, df: pd.DataFrame, target_col: str, 
                           test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Complete preprocessing pipeline for modeling.
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("Starting preprocessing pipeline...")
        
        # Apply all feature engineering steps
        df = self.create_rfm_features(df)
        df = self.create_usage_features(df)
        df = self.create_customer_segments(df)
        df = self.create_risk_features(df)
        df = self.create_campaign_features(df)
        df = self.create_age_features(df)
        df = self.create_interaction_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, target_col)
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove non-feature columns
        cols_to_remove = [col for col in X.columns if any(x in col.lower() 
                         for x in ['id', 'date', 'customer'])]
        X = X.drop(columns=cols_to_remove, errors='ignore')
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale numeric features
        X = self.scale_numeric_features(X, target_col)
        
        # Split data
        split_result = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y if y.dtype == 'object' or y.nunique() <= 10 else None
        )
        
        # Ensure proper types for return values
        X_train = pd.DataFrame(split_result[0])
        X_test = pd.DataFrame(split_result[1])
        y_train = pd.Series(split_result[2])
        y_test = pd.Series(split_result[3])
        
        print("Preprocessing complete!")
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Features: {len(self.feature_names)}")
        
        return X_train, X_test, y_train, y_test

def load_and_merge_data(data_dir: str = "../data/raw") -> pd.DataFrame:
    """
    Load and merge all dataset components.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        Merged dataframe
    """
    import os
    
    # Load master dataset if available
    master_file = os.path.join(data_dir, "master_dataset.csv")
    if os.path.exists(master_file):
        return pd.read_csv(master_file)
    
    # Otherwise, load and merge individual files
    print("Loading individual dataset files...")
    
    # Load all components
    demographics = pd.read_csv(os.path.join(data_dir, "demographics.csv"))
    usage_billing = pd.read_csv(os.path.join(data_dir, "usage_billing.csv"))
    crm_data = pd.read_csv(os.path.join(data_dir, "crm_data.csv"))
    campaign_data = pd.read_csv(os.path.join(data_dir, "campaign_data.csv"))
    engagement_data = pd.read_csv(os.path.join(data_dir, "engagement_data.csv"))
    churn_data = pd.read_csv(os.path.join(data_dir, "churn_data.csv"))
    
    # Merge all datasets
    df = (demographics
          .merge(usage_billing, on='customer_id')
          .merge(crm_data, on='customer_id')
          .merge(campaign_data, on='customer_id')
          .merge(engagement_data, on='customer_id')
          .merge(churn_data, on='customer_id'))
    
    print(f"Merged dataset shape: {df.shape}")
    return df

def create_time_series_features(df: pd.DataFrame, date_col: str = 'join_date') -> pd.DataFrame:
    """
    Create time-series features from date columns.
    
    Args:
        df: Input dataframe
        date_col: Name of date column
        
    Returns:
        DataFrame with time features
    """
    df = df.copy()
    
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Extract time components
        df[f'{date_col}_year'] = df[date_col].dt.year
        df[f'{date_col}_month'] = df[date_col].dt.month
        df[f'{date_col}_quarter'] = df[date_col].dt.quarter
        df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
        
        # Seasonality
        df[f'{date_col}_is_weekend'] = df[f'{date_col}_dayofweek'].isin([5, 6]).astype(int)
        
    return df

def main():
    """Test preprocessing pipeline."""
    # Load data
    df = load_and_merge_data()
    
    # Initialize preprocessor
    preprocessor = TelecomPreprocessor()
    
    # Prepare for churn prediction
    X_train, X_test, y_train, y_test = preprocessor.prepare_for_modeling(df, 'churned')
    
    print("\nPreprocessing pipeline test completed successfully!")
    print(f"Feature names: {preprocessor.feature_names[:10]}...")  # Show first 10

if __name__ == "__main__":
    main()