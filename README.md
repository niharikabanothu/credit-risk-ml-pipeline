# 💳 Credit Risk Prediction — End-to-End ML Pipeline

A production grade machine learning pipeline for predicting loan default risk, built from data ingestion through model deployment with a FastAPI REST API.

---

## 🎯 Problem Statement

Predict whether a loan applicant is likely to default based on financial and demographic features. This is a binary classification task using the **German Credit Dataset** (UCI ML Repository).

---

## 🏗️ Architecture

```
Raw Data → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation → FastAPI Deployment
```

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.74 | 0.72 | 0.74 | 0.71 | 0.78 |
| Random Forest | 0.76 | 0.74 | 0.76 | 0.74 | 0.81 |
| **XGBoost** | **0.78** | **0.76** | **0.78** | **0.76** | **0.83** |
| SVM | 0.75 | 0.73 | 0.75 | 0.73 | 0.79 |

*Best model: XGBoost with hyperparameter tuning via GridSearchCV*

---

## 📁 Project Structure

```
credit-risk-ml-pipeline/
├── data/
│   └── german_credit.csv
├── src/
│   ├── data_loader.py          # Data ingestion and validation
│   ├── eda.py                  # Exploratory Data Analysis
│   ├── preprocessing.py        # Feature engineering & transforms
│   ├── train.py                # Model training & hyperparameter tuning
│   └── evaluate.py             # Evaluation metrics & plots
├── models/
│   └── best_model.pkl          # Saved trained model
├── api/
│   └── app.py                  # FastAPI deployment
├── tests/
│   └── test_pipeline.py        # End-to-end pipeline test
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-ml-pipeline.git
cd credit-risk-ml-pipeline
pip install -r requirements.txt

# Run the full pipeline
python src/train.py

# Start the API
uvicorn api.app:app --reload

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"duration": 24, "credit_amount": 5000, "age": 35, "num_existing_credits": 1}'
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/model-info` | Model metadata and performance |
| POST | `/predict` | Predict credit risk for single applicant |
| POST | `/predict-batch` | Batch predictions |

### Example Response
```json
{
  "prediction": "Good Credit",
  "probability": 0.82,
  "risk_level": "Low",
  "model_version": "xgboost_v1"
}
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-learn** — ML models & preprocessing
- **XGBoost** — gradient boosting
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization
- **FastAPI** — REST API deployment
- **joblib** — model serialization

---

## 👤 Author

**Niharika Banothu**
M.Tech AI @ NIT Bhopal
[LinkedIn](https://linkedin.com/in/YOUR_LINK) · [GitHub](https://github.com/YOUR_USERNAME)
