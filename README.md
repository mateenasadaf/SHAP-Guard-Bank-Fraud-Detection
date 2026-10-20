# Bank Fraud Detection System Using XGBoost and SHAP

## Overview

The Bank Fraud Detection System is a machine-learning-based application that identifies potentially fraudulent financial transactions. It uses an XGBoost classifier trained on the PaySim dataset to estimate fraud probability and SHAP (SHapley Additive exPlanations) to explain the features influencing individual predictions.

The application provides a banking simulation where users can create accounts, preview transactions, check fraud risk, confirm or cancel transactions, and review transaction history.

## Objectives

- Detect potentially fraudulent financial transactions using machine learning.
- Estimate fraud probability and classify transactions using a configurable threshold.
- Explain individual predictions using SHAP.
- Simulate banking operations with account balance validation.
- Store transaction records and prediction results in SQLite.
- Provide an interactive dashboard using Streamlit and a FastAPI backend.

## Tech Stack

- **Programming:** Python
- **Machine Learning:** XGBoost, scikit-learn
- **Explainable AI:** SHAP
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Database:** SQLite
- **Data Processing:** Pandas, NumPy
- **Dataset:** PaySim synthetic financial transaction dataset

## Project Structure

```text
real-time-fraud-detection/
├── app.py
├── src/
│   ├── api.py
│   └── database.py
├── models/
│   ├── xgboost_fraud_model.json
│   ├── metadata.json
│   └── evaluation_results.json
├── notebooks/
│   └── fraud_detection_model_training.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Model Performance

The model was trained and evaluated using a chronological train-validation-test split.

| Metric | Test Result |
|---|---:|
| Average Precision | 95.10% |
| ROC-AUC | 99.91% |
| Precision at selected threshold | 74.84% |
| Recall at selected threshold | 94.57% |
| F1-score at selected threshold | 83.56% |

The selected decision threshold is approximately **0.97706**. These results are measured on the held-out PaySim test set and do not guarantee equivalent performance on real-world banking transactions.

## Key Features

### 1. Fraud Prediction
The XGBoost model estimates the probability that a transaction is fraudulent and compares it with the threshold stored in the model metadata.

### 2. SHAP Explanations
SHAP identifies the transaction features that push a prediction toward or away from the fraud class.

### 3. Banking Simulation
Users can create demo accounts, select sender and receiver accounts, enter transaction amounts, and preview projected balances.

### 4. Transaction Confirmation
Users can review, confirm, or cancel pending transactions. Flagged transactions require an explicit override in this demonstration application.

### 5. Transaction History and Dashboard
The application displays recorded transactions, fraud predictions, estimated probabilities, and summary charts.

## Installation

### Prerequisites

- Python 3.10 or a compatible version supported by the installed dependencies
- pip

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd real-time-fraud-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

From the project root:

```bash
uvicorn src.api:app --reload
```

The API documentation is available at:

http://127.0.0.1:8000/docs

Health endpoint:

http://127.0.0.1:8000/health

### 5. Start the Streamlit frontend

Open a second terminal, activate the same virtual environment, and run:

```bash
streamlit run app.py
```

## Dataset and Model Artifacts

The model was trained using the PaySim synthetic transaction dataset. The training notebook documents the preprocessing, chronological splitting, model training, threshold selection, and evaluation.

The deployed model and its matching metadata are stored in the `models/` directory. The API uses the metadata to determine the expected feature order and prediction threshold.

The dataset is not included in this repository. Obtain it from an authorized source if you want to reproduce training.

## Limitations

- PaySim is synthetic data and may not represent all real banking fraud patterns.
- Fraud probability is a model estimate, not proof of fraudulent activity.
- A high test score does not guarantee performance under real-world distribution shifts.
- SHAP explanations describe model behaviour; they do not establish causation.
- The banking interface is a demonstration and is not intended for processing real financial transactions.

## Future Improvements

- Evaluate the model on additional datasets and real-world-like scenarios.
- Monitor model drift and recalibrate the decision threshold when appropriate.
- Add authentication, authorization, and audit logging.
- Expand automated tests for API endpoints and transaction workflows.
- Improve fraud-alert prioritization and reporting.

## Disclaimer

This project is intended for educational and demonstration purposes only. It is not a production-ready banking security system.
