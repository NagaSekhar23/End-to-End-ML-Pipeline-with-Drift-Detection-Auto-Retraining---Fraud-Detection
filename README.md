# 🚨 TabNet Fraud Detection Pipeline

**Production ML system** with TabNet deep learning, drift detection, auto-retraining, MLflow, and Dockerized FastAPI deployment.

[![TabNet Fraud Demo](https://img.shields.io/badge/Demo-ROC--AUC_0.96-brightgreen)](https://github.com/NagaSekhar23/End-to-End-ML-Pipeline-with-Drift-Detection-Auto-Retraining---Fraud-Detection)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-blue)](http://localhost:8000/docs)
[![Docker](https://img.shields.io/badge/Docker-Containerized-green)](Dockerfile)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](http://localhost:5000)

## 🎯 Features

| Feature | Status | Metrics |
|---------|--------|---------|
| **TabNet Model** | ✅ Production | **ROC-AUC 0.96** |
| **Drift Detection** | ✅ KS Test | 23/30 features drift |
| **Auto-Retraining** | ✅ MLflow | Model registry |
| **REST API** | ✅ FastAPI | `/predict` endpoint |
| **Container** | ✅ Docker | One-command deploy |
| **Tracking** | ✅ MLflow | Experiments logged |

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/NagaSekhar23/End-to-End-ML-Pipeline-with-Drift-Detection-Auto-Retraining---Fraud-Detection.git
cd End-to-End-ML-Pipeline-with-Drift-Detection-Auto-Retraining---Fraud-Detection
pip install -r requirements.txt

# 2. Download dataset (150MB)
mkdir -p data
curl -L "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv" -o data/creditcard.csv

# 3. Run pipeline
jupyter notebook notebooks/fraud_tabnet_end_to_end.ipynb

# 4. Start services
mlflow ui                    # http://localhost:5000
uvicorn src.api_service:app --reload  # http://localhost:8000/docs

# 5. Docker deploy
docker build -t fraud-api .
docker run -p 8001:8000 fraud-api  # http://localhost:8001
| File                                    | Purpose             | Key Technology          |
| --------------------------------------- | ------------------- | ----------------------- |
| notebooks/fraud_tabnet_end_to_end.ipynb | End-to-end pipeline | Jupyter + TabNet demo   |
| src/data_pipeline.py                    | Data processing     | Stratified splits + RUS |
| src/train_tabnet.py                     | Model training      | TabNet + MLflow         |
| src/drift_and_retrain.py                | Drift monitoring    | KS test + retrain       |
| src/api_service.py                      | Production API      | FastAPI /predict        |

Test Set (56,962 transactions):
ROC-AUC:        0.9648  ⭐ PRODUCTION READY
Precision(1):   0.02
Recall(1):      0.92    ⭐ CATCHES 92% FRAUDS
F1(1):          0.05
Accuracy:       0.93
graph TD
    A[Kaggle Dataset<br/>284k transactions] --> B[data_pipeline.py]
    B --> C[TabNet Training<br/>ROC-AUC 0.96]
    C --> D[MLflow Tracking]
    E[New Data Batches] --> F[drift_and_retrain.py<br/>KS Test]
    F -->|Drift Detected| G[Auto-Retraining]
    G --> D
    D --> H[api_service.py<br/>FastAPI]
    H --> I[Docker Container]
TabNetClassifier(
    n_d=24, n_a=24, n_steps=3,
    optimizer_fn=torch.optim.Adam, lr=0.02,
    mask_type="sparsemax",
    max_epochs=200, patience=50
)
# Feature drift (KS test)
for feature in 30_features:
    ks_stat, p_value = ks_2samp(ref_data, new_batch)
    if p_value < 0.01: trigger_retrain()
GET  /health           → {"status": "healthy"}
POST /predict          → {"fraud_probability": 0.87, "risk_level": "HIGH"}
docker build -t fraud-tabnet-api .
docker run -p 8000:8000 fraud-tabnet-api
🤖 ML: pytorch-tabnet (ROC-AUC 0.96)
📊 MLOps: MLflow (tracking + registry)
🌐 API: FastAPI + Uvicorn
🐳 DevOps: Docker
📈 Drift: SciPy KS-test
🔧 Data: Pandas, scikit-learn, imbalanced-learn
| Metric       | Value | Industry Benchmark |
| ------------ | ----- | ------------------ |
| ROC-AUC      | 0.96  | Production-ready   |
| Fraud Recall | 92%   | Excellent          |
| API Latency  | <50ms | Real-time          |
| Docker Size  | ~2GB  | Optimized          |
