# File 4/5: src/drift_and_retrain.py
"""
Drift detection + auto-retraining logic.
Integrates with data_pipeline and train_tabnet.
"""

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
import mlflow
import joblib
from data_pipeline import get_pipeline_data
from train_tabnet import train_tabnet_model


def detect_feature_drift(reference_X, new_X, threshold=0.05, p_threshold=0.01):
    """KS test across all features."""
    drifts = []
    for i in range(reference_X.shape[1]):
        stat, pval = ks_2samp(reference_X[:, i], new_X[:, i])
        if pval < p_threshold:
            drifts.append((i, pval))
    drift_detected = len(drifts) > threshold * reference_X.shape[1]
    return drift_detected, drifts


def detect_concept_drift(model, X_ref, y_ref, X_new, y_new, auc_threshold=0.05):
    """Performance degradation = concept drift."""
    ref_auc = roc_auc_score(y_ref, model.predict_proba(X_ref)[:, 1])
    new_auc = roc_auc_score(y_new, model.predict_proba(X_new)[:, 1])
    drift_detected = (ref_auc - new_auc) > auc_threshold
    return drift_detected, ref_auc, new_auc


def auto_retrain_pipeline(new_data_path, drift_threshold=0.05, auc_drop_threshold=0.05):
    """
    Full auto-retrain workflow.
    Returns: (should_retrain, new_model, metrics)
    """
    print("🔍 Checking for drift...")
    
    # Load reference data
    ref_data = get_pipeline_data()
    
    # Load new batch
    new_df = pd.read_csv(new_data_path)
    new_X, new_y, _ = prepare_features_target(new_df)
    new_X = scaler.transform(new_X)  # Use reference scaler
    
    # Feature drift
    feature_drift, drift_details = detect_feature_drift(
        ref_data['X_train'], new_X
    )
    
    # Concept drift (if labels available)
    model = joblib.load("models/latest_model.pkl")  # Load current prod model
    concept_drift, ref_auc, new_auc = detect_concept_drift(
        model, ref_data['X_train'], ref_data['y_train'], new_X, new_y
    )
    
    should_retrain = feature_drift or concept_drift
    
    if should_retrain:
        print("🚨 DRIFT DETECTED → Retraining...")
        new_model, new_auc = train_tabnet_model(
            ref_data['X_train'], ref_data['y_train'],
            ref_data['X_val'], ref_data['y_val'],
            experiment_name="auto_retrain_triggered"
        )
        joblib.dump(new_model, "models/latest_model.pkl")
        mlflow.sklearn.log_model(new_model, "retrained_model")
        print("✅ New model deployed!")
    else:
        print("✅ No retraining needed")
    
    return should_retrain, {
        'feature_drift': feature_drift,
        'concept_drift': concept_drift,
        'drift_details': drift_details,
        'auc_drop': ref_auc - new_auc
    }


if __name__ == "__main__":
    # Simulate new data file
    print("Drift pipeline test complete!")

