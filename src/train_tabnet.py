# File 3/5: src/train_tabnet.py
"""
TabNet training with MLflow integration.
"""

import mlflow
import mlflow.sklearn
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np


def train_tabnet_model(X_train, y_train, X_val, y_val, experiment_name="fraud_tabnet"):
    """Train TabNet and log to MLflow."""
    
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="tabnet_auto"):
        
        # Production hyperparameters
        model = TabNetClassifier(
            n_d=24, n_a=24, n_steps=3,
            gamma=1.3, lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=0.02, weight_decay=1e-5),
            mask_type="sparsemax",
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            seed=42,
            verbose=1
        )
        
        # Train
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_name=['val'],
            eval_metric=['auc'],
            max_epochs=200,
            patience=50,
            batch_size=256,
            virtual_batch_size=128
        )
        
        # Evaluate on validation
        val_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        
        # Log everything
        mlflow.log_param("n_d", 24)
        mlflow.log_param("n_a", 24)
        mlflow.log_param("max_epochs", 200)
        mlflow.log_metric("val_auc", val_auc)
        
        # Save model
        mlflow.sklearn.log_model(model, "tabnet_production")
        
        print(f"✅ Training complete. Val AUC: {val_auc:.4f}")
        return model, val_auc


if __name__ == "__main__":
    from data_pipeline import get_pipeline_data
    data = get_pipeline_data()
    model, auc = train_tabnet_model(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )

