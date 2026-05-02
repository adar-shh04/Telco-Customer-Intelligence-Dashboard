"""
Telco Customer Intelligence Platform - Data Pipeline Entry Point

Downloads the real IBM Telco Customer Churn dataset from Kaggle,
enriches it with derived features, and saves all CSVs + SQLite database.

Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""

import sys
import os
sys.path.append('.')

from src.data_loader import TelecomDataLoader

def main():
    # Step 1: Load and transform Kaggle data
    print("=" * 60)
    print("TELCO CUSTOMER INTELLIGENCE - DATA PIPELINE")
    print("Source: Kaggle IBM Telco Customer Churn (Real Data)")
    print("=" * 60)

    loader = TelecomDataLoader()
    datasets = loader.load_and_transform()

    # Step 2: Save all CSVs
    data_dir = 'data/raw'
    os.makedirs(data_dir, exist_ok=True)

    for name, df in datasets.items():
        if name == 'kaggle_raw':
            continue
        filepath = os.path.join(data_dir, f"{name}.csv")
        df.to_csv(filepath, index=False)
        print(f"Saved {name} → {filepath} ({len(df)} rows)")

    # Step 3: Create SQLite database
    print("\nCreating SQLite database...")
    from src.create_db import create_database
    create_database()

    # Step 4: Summary
    master = datasets['master_dataset']
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Dataset shape: {master.shape}")
    print(f"Real customers: {len(master):,}")
    print(f"Churn rate: {master['churned'].mean():.2%}")
    print(f"Avg ARPU: ${master['arpu'].mean():.2f}")
    print(f"Avg Satisfaction: {master['satisfaction_score'].mean():.1f}/10")

if __name__ == '__main__':
    main()
