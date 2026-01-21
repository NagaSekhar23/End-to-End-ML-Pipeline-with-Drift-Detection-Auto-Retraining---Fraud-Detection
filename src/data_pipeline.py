# File 2/5: src/data_pipeline.py
"""
Reusable data pipeline for fraud detection.
Handles loading, preprocessing, splitting, imbalance.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")


def load_raw_data():
    """Load Kaggle credit card fraud dataset."""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def prepare_features_target(df):
    """Split into X, y."""
    TARGET_COL = 'Class'
    feature_cols = [col for col in df.columns if col != TARGET_COL]
    X = df[feature_cols].values
    y = df[TARGET_COL].values
    return X, y, feature_cols


def stratified_splits(X, y, test_size=0.2, val_size=0.25, random_state=42):
    """60/20/20 stratified split."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=random_state, stratify=y_trainval
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def scale_amount(X_train, X_val, X_test, feature_cols):
    """Scale Amount feature only."""
    amount_idx = feature_cols.index('Amount')
    scaler = StandardScaler()
    X_train[:, [amount_idx]] = scaler.fit_transform(X_train[:, [amount_idx]])
    X_val[:, [amount_idx]] = scaler.transform(X_val[:, [amount_idx]])
    X_test[:, [amount_idx]] = scaler.transform(X_test[:, [amount_idx]])
    return scaler


def handle_imbalance(X_train, y_train):
    """Undersample majority class."""
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    return X_res, y_res


def get_pipeline_data(random_state=42):
    """Full pipeline: load → split → scale → balance → return."""
    df = load_raw_data()
    X, y, feature_cols = prepare_features_target(df)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_splits(X, y, random_state=random_state)
    scaler = scale_amount(X_train, X_val, X_test, feature_cols)
    
    X_train_bal, y_train_bal = handle_imbalance(X_train, y_train)
    
    print(f"Pipeline complete: train_bal={X_train_bal.shape}")
    return {
        'X_train': X_train_bal, 'y_train': y_train_bal,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler,
        'feature_cols': feature_cols
    }


if __name__ == "__main__":
    data = get_pipeline_data()
    print("Pipeline test successful!")

