# Bank Fraud Detection System Using XGBoost and SHAP

## Overview

The Bank Fraud Detection System is a machine-learning-based application that identifies potentially fraudulent financial transactions. It uses an XGBoost classifier trained on the PaySim dataset to estimate fraud probability and SHAP (SHapley Additive exPlanations) to explain the features influencing individual predictions.

The application provides a banking simulation where users can create accounts, preview transactions, check fraud risk, confirm or cancel transactions, and review transaction history.

**Live Application:** [SHAP-Guard — Bank Fraud Detection](https://shap-guard-bank-fraud-detection-mateena-sadaf.streamlit.app/)

**Backend Health Check:** [FastAPI Health Endpoint](https://shap-guard-bank-fraud-detection.onrender.com/health)

**GitHub Repository:** [SHAP-Guard — Bank Fraud Detection](https://github.com/mateenasadaf/SHAP-Guard-Bank-Fraud-Detection)

---

## Dataset

This project uses the PaySim Mobile Money Transaction Dataset for training and evaluating the bank fraud detection model.

- **Dataset:** PaySim Mobile Money Transaction Dataset
- **Source:** [PaySim Dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Data type:** Simulated financial transactions, including legitimate and fraudulent transactions.
- **Purpose:** Training and evaluating the XGBoost fraud detection model.

### Download and Setup

1. Download the dataset from Kaggle.
2. Extract the downloaded file.
3. Place the CSV file at `data/raw/data.csv` in the project directory, ensuring the filename matches what the training notebook or data-loading code expects.

**Note:** The raw dataset is not included in this repository. The trained model and evaluation artifacts are stored in the `models/` directory. The dataset is only required if you want to reproduce the training process.

---

## Objectives

- Detect potentially fraudulent financial transactions using machine learning.
- Estimate fraud probability and classify transactions using a configurable decision threshold.
- Explain individual predictions using SHAP.
- Simulate banking operations with account balance validation.
- Store transaction records and prediction results in SQLite.
- Provide an interactive dashboard using Streamlit and a FastAPI backend.
- Isolate demo transaction histories using separate demo-session identifiers.

---

## Tech Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Machine learning | XGBoost, scikit-learn |
| Explainable AI | SHAP |
| Backend | FastAPI |
| Frontend | Streamlit |
| Database | SQLite |
| Data processing | Pandas, NumPy |
| Dataset | PaySim synthetic transaction dataset |
| Backend deployment | Render |
| Frontend deployment | Streamlit Community Cloud |
| Version control | Git and GitHub |

---

## Project Architecture

The application follows a frontend-backend architecture.

```text
                    User
                     |
                     v
            Streamlit Frontend
                 (app.py)
                     |
                     | HTTP requests
                     | X-Demo-Session
                     v
              FastAPI Backend
                (src/api.py)
                     |
           +---------+----------+
           |                    |
           v                    v
     XGBoost Model         SQLite Database
       + SHAP              (src/database.py)
           |
           v
    Fraud Prediction
    + SHAP Explanation
           |
           v
    Result to Frontend
```

### Architecture Components

1. **Streamlit frontend:** Provides the banking interface, transaction forms, fraud-risk preview, confirmation workflow, transaction history, and dashboard.
2. **FastAPI backend:** Receives requests, validates transaction operations, obtains model predictions, and coordinates database interactions.
3. **XGBoost model:** Estimates the probability that a transaction belongs to the fraud class.
4. **SHAP:** Explains the contribution of individual input features to a model prediction.
5. **SQLite database:** Stores demo account information and transaction-related records.
6. **Demo-session isolation:** Uses an `X-Demo-Session` header to separate records belonging to different demo sessions.

---

## Project Structure

```text
SHAP-Guard-Bank-Fraud-Detection/
│
├── app.py
│
├── src/
│   ├── api.py
│   └── database.py
│
├── models/
│   ├── xgboost_fraud_model.json
│   ├── metadata.json
│   └── evaluation_results.json
│
├── notebooks/
│   └── fraud_detection_model_training.ipynb
│
├── tests/
│   ├── test_session_isolation.py
│   └── test_api_session_isolation.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

The `data/raw/` directory may also be created locally when reproducing model training. It is not required to be committed to the repository when the raw dataset is excluded.

### Explanation of Each File

#### 1. `app.py` — Streamlit Frontend

This is the main entry point for the user interface.

Responsibilities include:

- Displaying the banking simulation dashboard.
- Allowing users to create demo accounts and select sender and receiver accounts.
- Collecting transaction details, such as the transaction amount.
- Requesting fraud predictions from the FastAPI backend.
- Displaying predicted fraud probability and SHAP explanations.
- Allowing users to preview, confirm, or cancel transactions.
- Displaying transaction history and dashboard summaries.
- Sending an `X-Demo-Session` identifier with relevant API requests to separate demo-session data.

**How it works:** The user interacts with the Streamlit interface, which sends HTTP requests to the backend. The backend returns results that the frontend displays.

#### 2. `src/api.py` — FastAPI Backend

This file implements the application's API endpoints and backend workflows.

Responsibilities include:

- Exposing HTTP endpoints for the frontend.
- Providing the `/health` endpoint for service-health checks.
- Processing transaction and fraud-prediction requests.
- Loading or accessing the trained model and its metadata.
- Coordinating transaction validation, prediction, and database operations.
- Handling demo-session identifiers for session-specific data access.

The backend runs independently of Streamlit, allowing the frontend and backend to be deployed as separate services.

#### 3. `src/database.py` — Database Management

This file manages the SQLite database and database operations.

Responsibilities include:

- Establishing database connections.
- Creating or initializing database tables.
- Storing and retrieving demo account and transaction records.
- Supporting transaction-history queries.
- Applying session-aware database operations where implemented.

SQLite provides a lightweight database without requiring a separate database server.

#### 4. `models/xgboost_fraud_model.json` — Trained Model

This file contains the saved XGBoost model in JSON format.

It is used to generate fraud predictions without retraining the model whenever the application starts.

The deployed application loads the existing trained model rather than training a new one for each transaction.

#### 5. `models/metadata.json` — Model Metadata

This file contains metadata associated with the trained model.

The metadata is used by the application to determine information such as:

- The expected model input feature order.
- The selected decision threshold.
- Other model configuration details recorded during training.

Keeping the metadata aligned with the saved model is important for consistent predictions.

#### 6. `models/evaluation_results.json` — Evaluation Results

This file stores the model evaluation results produced during the training and evaluation workflow.

It provides a record of the model's measured performance, making the results easier to inspect without repeating the training process.

#### 7. `notebooks/fraud_detection_model_training.ipynb` — Training Notebook

This notebook documents the machine-learning workflow.

It covers the training and evaluation process, including:

- Loading and inspecting the PaySim dataset.
- Data preprocessing and feature preparation.
- Chronological train-validation-test splitting.
- Training the XGBoost classifier.
- Evaluating classification performance.
- Selecting a decision threshold.
- Saving the trained model, metadata, and evaluation results.

The notebook is intended for training reproduction and model analysis. Normal application usage does not require retraining the model.

#### 8. `tests/test_session_isolation.py` — Database Session-Isolation Tests

This test file checks database-level session isolation.

It helps verify that records associated with one demo session are not incorrectly exposed to another session through the tested database operations.

#### 9. `tests/test_api_session_isolation.py` — API Session-Isolation Tests

This test file checks session isolation through the FastAPI layer.

It tests API behavior when requests use different demo-session identifiers, helping detect cross-session data leakage in the tested endpoints.

Both test files were run locally, and all seven tests passed.

#### 10. `requirements.txt` — Python Dependencies

This file lists the Python packages required by the application, including dependencies for the API, Streamlit interface, database integration, machine learning, data processing, and SHAP explanations.

Install these packages using:

```bash
pip install -r requirements.txt
```

#### 11. `.gitignore` — Git Exclusions

This file specifies files and directories that Git should ignore.

Typical exclusions include virtual environments, Python cache files, local environment variables, temporary artifacts, and datasets that should not be committed.

API keys and other secrets should never be committed to the public repository.

#### 12. `README.md` — Project Documentation

This file documents the project's purpose, dataset, architecture, model performance, setup instructions, deployment process, limitations, and future improvements.

---

## Model Performance

The model was trained and evaluated using a chronological train-validation-test split.

| Metric | Test Result |
|---|---:|
| Average Precision | 95.10% |
| ROC-AUC | 99.91% |
| Precision at selected threshold | 74.84% |
| Recall at selected threshold | 94.57% |
| F1-score at selected threshold | 83.56% |

**Selected decision threshold:** Approximately `0.97706`.

These results were measured on the held-out PaySim test set. They do not guarantee equivalent performance on real-world banking transactions.

---

## Key Features

### 1. Fraud Prediction

The XGBoost model estimates the probability that a transaction is fraudulent. The application compares this probability against the decision threshold stored in the model metadata to classify the transaction.

### 2. SHAP Explanations

SHAP identifies the features that contribute to an individual prediction, showing which factors push the model's output toward or away from the fraud class.

These explanations describe model behavior rather than proving that a transaction is fraudulent.

### 3. Banking Simulation

Users can create demo accounts, select sender and receiver accounts, enter transaction amounts, and preview projected balances before confirming a transaction.

### 4. Transaction Confirmation

Users can review, confirm, or cancel pending transactions. Flagged transactions require an explicit override in this demonstration application.

### 5. Transaction History and Dashboard

The application displays recorded transactions, fraud predictions, estimated probabilities, and summary charts.

### 6. Demo-Session Isolation

The application uses an `X-Demo-Session` header to associate relevant requests with separate demo sessions. This helps keep transaction histories separate between independent browser sessions.

This is a demonstration-level isolation mechanism, not a replacement for user authentication, authorization, or production-grade access control.

---

## Local Installation and Setup

### Prerequisites

- Python 3.10 or another version compatible with the installed dependencies.
- pip.
- Git.

### Step 1: Clone the repository

```bash
git clone https://github.com/mateenasadaf/SHAP-Guard-Bank-Fraud-Detection.git
cd SHAP-Guard-Bank-Fraud-Detection
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Start the FastAPI backend

From the project root:

```bash
uvicorn src.api:app --reload
```

API documentation:

`http://127.0.0.1:8000/docs`

Health endpoint:

`http://127.0.0.1:8000/health`

### Step 5: Start the Streamlit frontend

Open a second terminal, activate the same virtual environment, and run:

```bash
streamlit run app.py
```

The local Streamlit interface will display its URL in the terminal, usually `http://localhost:8501`.

Ensure that the frontend's API URL configuration points to the running backend.

---

## Deployment

The application is deployed using two separate services:

- **Frontend:** Streamlit Community Cloud.
- **Backend:** Render Web Services.

The source code is maintained in GitHub. Pushing changes to the configured branch can trigger a new backend deployment when automatic deployment is enabled.

### Deployment URLs

| Component | URL |
|---|---|
| Live Streamlit application | https://shap-guard-bank-fraud-detection-mateena-sadaf.streamlit.app/ |
| Render backend | https://shap-guard-bank-fraud-detection.onrender.com |
| Backend health check | https://shap-guard-bank-fraud-detection.onrender.com/health |
| FastAPI API documentation | https://shap-guard-bank-fraud-detection.onrender.com/docs |
| GitHub repository | https://github.com/mateenasadaf/SHAP-Guard-Bank-Fraud-Detection |

### A. Deploying the FastAPI Backend on Render

The backend is hosted as a Render web service.

#### Configuration

**Build command:**

```bash
pip install -r requirements.txt
```

**Start command:**

```bash
uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

The service must use the repository containing `src/api.py` and the model artifacts required by the application.

#### Backend deployment workflow

1. Push the latest backend code to the connected GitHub repository.
2. Open the Render Dashboard.
3. Select the SHAP-Guard backend web service.
4. Check the deployment events or logs.
5. Verify that the new commit finishes deploying successfully.
6. Confirm that the service status is `Live`.
7. Open the health endpoint to check that the backend responds.

The health endpoint is:

`https://shap-guard-bank-fraud-detection.onrender.com/health`

A successful health response confirms that the endpoint is reachable; it does not by itself verify every model or transaction workflow.

### B. Deploying the Streamlit Frontend on Streamlit Community Cloud

The frontend is hosted on Streamlit Community Cloud.

#### Configuration

- **Repository:** `mateenasadaf/SHAP-Guard-Bank-Fraud-Detection`
- **Branch:** `main`
- **Main file:** `app.py`

The frontend must use the deployed backend URL rather than the local development URL.

Backend base URL:

```text
https://shap-guard-bank-fraud-detection.onrender.com
```

If the application reads its API URL from Streamlit secrets or environment configuration, set the appropriate configuration value there. The exact key must match the one used by `app.py`.

#### Frontend deployment workflow

1. Connect the GitHub repository to Streamlit Community Cloud.
2. Select the `main` branch and `app.py` as the entry point.
3. Configure any required secrets or environment variables.
4. Ensure the API URL points to the deployed Render backend.
5. Deploy the application.
6. Open the public Streamlit URL and test the main workflows.

Live application:

https://shap-guard-bank-fraud-detection-mateena-sadaf.streamlit.app/

### C. Updating the Deployed Application

For future code changes:

1. Make the required changes locally.
2. Run the automated tests.
3. Check the modified files using Git.
4. Commit and push the changes to GitHub.
5. Verify the Render deployment if backend files changed.
6. Verify the Streamlit deployment if frontend files changed.
7. Test the updated application in the browser.

Example Git commands:

```bash
git status
git add app.py src/api.py src/database.py tests/
git commit -m "Describe the changes"
git push origin main
```

The deployed session-isolation changes were committed as:

`253ce4f — Add demo session isolation`

### D. Post-Deployment Verification

After deployment, verify the following:

- The Streamlit application opens successfully.
- The frontend can communicate with the Render backend.
- The backend health endpoint responds.
- Transactions can be previewed and processed.
- Fraud predictions and SHAP explanations are displayed.
- Transaction histories remain separate in normal and incognito browser sessions.
- Transaction confirmation and cancellation behave as expected.

The deployed application has been manually tested for transaction processing and session isolation. The seven automated tests also passed locally.

### Deployment Considerations

- Render services may take time to start after inactivity, depending on the hosting configuration.
- SQLite data stored on an ephemeral deployment filesystem may not persist across restarts or redeployments. Persistent storage or a managed database should be considered if transaction data must survive redeployments.
- Demo-session identifiers are not secure authentication credentials. Do not use this mechanism alone to protect real banking data.
- Never commit API keys, passwords, or other secrets to GitHub.

---

## Dataset and Model Artifacts

The model was trained using the PaySim synthetic transaction dataset. The training notebook documents preprocessing, chronological splitting, model training, threshold selection, and evaluation.

The deployed model and its matching metadata are stored in the `models/` directory. The API uses the metadata to determine the expected feature order and prediction threshold.

The raw dataset is not included in the repository. Obtain it from an authorized source if you want to reproduce training.

---

## Limitations

- PaySim is synthetic data and may not represent all real banking fraud patterns.
- Fraud probability is a model estimate, not proof of fraudulent activity.
- A high test score does not guarantee performance under real-world distribution shifts.
- SHAP explanations describe model behavior; they do not establish causation.
- The banking interface is a demonstration and is not intended for processing real financial transactions.
- Demo-session isolation does not provide full authentication or authorization.
- SQLite persistence depends on the hosting environment and storage configuration.

---

## Future Improvements

- Evaluate the model on additional datasets and real-world-like scenarios.
- Monitor model drift and recalibrate the decision threshold when appropriate.
- Add authentication, authorization, and audit logging.
- Expand automated tests for API endpoints and transaction workflows.
- Improve fraud-alert prioritization and reporting.
- Consider persistent database storage for deployment.
- Add monitoring and operational alerts for service failures.

---

## Disclaimer

This project is intended for educational and demonstration purposes only. It is not a production-ready banking security system.
