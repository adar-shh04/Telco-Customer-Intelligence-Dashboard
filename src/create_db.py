"""
Create SQLite Database from CSV files.

This script reads the CSV files produced by the Kaggle data loading pipeline
(src/data_loader.py) and loads them into a SQLite database for SQL-based analysis.

Data Source: IBM Telco Customer Churn dataset from Kaggle, enriched with
derived CRM, campaign, and engagement features.
"""

import pandas as pd
import sqlite3
import os
import sys

def create_database():
    """Load CSVs into SQLite database."""
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'raw')
    db_path = os.path.join(base_dir, 'data', 'telecom.db')
    
    print(f"Creating database at {db_path}...")
    
    # Connect to SQLite
    conn = sqlite3.connect(db_path)
    
    # List of files to load
    files_to_load = [
        'demographics.csv',
        'usage_billing.csv',
        'crm_data.csv',
        'campaign_data.csv',
        'engagement_data.csv',
        'churn_data.csv'
        # 'master_dataset.csv' is excluded to avoid duplication, as it's a join of the above
    ]
    
    total_rows = 0
    tables_created = []
    
    try:
        for filename in files_to_load:
            file_path = os.path.join(data_dir, filename)
            
            if not os.path.exists(file_path):
                print(f"Warning: {filename} not found in {data_dir}. Skipping.")
                continue
                
            # Table name is filename without extension
            table_name = os.path.splitext(filename)[0]
            
            # Read CSV
            print(f"Loading {filename} into table '{table_name}'...")
            df = pd.read_csv(file_path)
            
            # Write to SQLite
            df.to_sql(name=table_name, con=conn, if_exists='replace', index=False)
            
            # Verify
            count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table_name}", conn)['count'].iloc[0]
            print(f"  - Loaded {count:,} rows into '{table_name}'")
            
            total_rows += count
            tables_created.append(table_name)
            
        print("\nDatabase creation complete!")
        print(f"Total tables: {len(tables_created)}")
        print(f"Total rows loaded: {total_rows:,}")
        
    except Exception as e:
        print(f"Error creating database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    create_database()
