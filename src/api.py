from pathlib import Path
import json
import sqlite3
import uuid

import pandas as pd
import xgboost as xgb
import shap

from fastapi import FastAPI, HTTPException, Query, Header, Depends
from pydantic import BaseModel, Field

from src.database import (
    initialize_database,
    save_transaction,
    get_recent_transactions,
    create_account,
    get_account,
    get_accounts,
    finalize_transfer,
    finalize_account_transaction,
)

# ============================================================
# CONFIGURATION AND MODEL LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"  # Newly trained 9-feature model; keep the old models/ folder as backup
MODEL_PATH = MODEL_DIR / "xgboost_fraud_model.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))

FEATURES = metadata["features"]
MEDIANS = metadata.get("medians", metadata.get("train_medians", {}))
THRESHOLD = float(metadata.get("threshold", 0.50))

# Fail early if the model and metadata do not describe the same feature schema.
if not isinstance(FEATURES, list) or not FEATURES:
    raise ValueError("Model metadata must contain a non-empty 'features' list.")
if not 0.0 <= THRESHOLD <= 1.0:
    raise ValueError(f"Invalid fraud threshold in metadata: {THRESHOLD}")
if model.get_booster().num_features() != len(FEATURES):
    raise ValueError(
        f"Model expects {model.get_booster().num_features()} features, "
        f"but metadata lists {len(FEATURES)}. Check models_v2 files."
    )
booster_feature_names = model.get_booster().feature_names
if booster_feature_names and booster_feature_names != FEATURES:
    raise ValueError(
        "Feature order in metadata.json does not match the saved XGBoost model."
    )

TRANSACTION_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
}

# Create the SHAP explainer once when the API loads.
explainer = shap.TreeExplainer(model)

# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Financial Fraud Detection API",
    description=(
        "XGBoost fraud prediction, demo account management, "
        "SQLite transaction history, and SHAP explanations."
    ),
    version="3.0.0",
)


@app.on_event("startup")
def startup_event():
    initialize_database()


def require_session(x_demo_session: str = Header(..., alias="X-Demo-Session")) -> str:
    """Require a well-formed UUID demo-session identifier."""
    try:
        return str(uuid.UUID(x_demo_session))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail="Missing or invalid X-Demo-Session header. Refresh the app to create a new session.",
        )


# ============================================================
# REQUEST MODELS
# ============================================================

class Transaction(BaseModel):
    step: int = Field(ge=1)
    type: str
    amount: float = Field(ge=0)

    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)

    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)


class AccountCreate(BaseModel):
    account_id: str = Field(min_length=1, max_length=30)
    name: str = Field(min_length=1, max_length=100)
    starting_balance: float = Field(ge=0)


class TransferPreview(BaseModel):
    sender_account_id: str = Field(min_length=1, max_length=30)
    receiver_account_id: str = Field(min_length=1, max_length=30)
    amount: float = Field(gt=0)
    # PaySim step is simulation time, not wall-clock time.
    step: int = Field(default=600, ge=1)


class TransferDecision(BaseModel):
    approve: bool
    override_fraud: bool = False


class AccountTransactionPreview(BaseModel):
    account_id: str = Field(min_length=1, max_length=30)
    transaction_type: str
    amount: float = Field(gt=0)
    # PaySim step is simulation time, not wall-clock time.
    step: int = Field(default=600, ge=1)


# ============================================================
# FEATURE PREPARATION AND PREDICTION
# ============================================================

def prepare_features(transaction: Transaction) -> pd.DataFrame:
    data = transaction.model_dump()
    transaction_type = data.pop("type").strip().upper()

    if transaction_type not in TRANSACTION_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported transaction type: {transaction_type}. "
                f"Supported types: {sorted(TRANSACTION_TYPES)}"
            ),
        )

    row = pd.DataFrame([data])

    # Re-create the one-hot transaction-type features used in training.
    for category in sorted(TRANSACTION_TYPES):
        feature_name = f"type_{category}"
        if feature_name in FEATURES:
            row[feature_name] = int(transaction_type == category)

    # Match the model's exact expected columns and order.
    row = row.reindex(columns=FEATURES)

    # Fill missing values with saved training medians.
    for column in FEATURES:
        if row[column].isna().any():
            row[column] = row[column].fillna(MEDIANS.get(column, 0))

    return row.astype(float)


def score_transaction(transaction: Transaction):
    features = prepare_features(transaction)
    probability = float(model.predict_proba(features)[0][1])
    prediction = int(probability >= THRESHOLD)
    return features, probability, prediction


# ============================================================
# HOME AND HEALTH
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Financial Fraud Detection API is running",
        "model_loaded": True,
        "shap_enabled": True,
        "threshold": THRESHOLD,
        "documentation": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "shap_enabled": True,
        "threshold": THRESHOLD,
    }


# ============================================================
# DEMO ACCOUNT ENDPOINTS
# ============================================================

@app.get("/accounts")
def list_accounts(owner_session_id: str = Depends(require_session)):
    accounts = get_accounts(owner_session_id)
    return {"count": len(accounts), "accounts": accounts}


@app.post("/accounts", status_code=201)
def add_account(account: AccountCreate, owner_session_id: str = Depends(require_session)):
    try:
        created_account = create_account(
            account_id=account.account_id,
            name=account.name,
            starting_balance=account.starting_balance,
            owner_session_id=owner_session_id,
        )
        return {
            "message": "Demo account created successfully",
            "account": created_account,
        }
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="That account ID already exists or the account details are invalid.",
        )


@app.get("/accounts/{account_id}")
def account_details(account_id: str, owner_session_id: str = Depends(require_session)):
    account = get_account(account_id, owner_session_id)
    if account is None:
        raise HTTPException(status_code=404, detail="Demo account not found.")
    return {"account": account}


# ============================================================
# EXISTING PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(transaction: Transaction, owner_session_id: str = Depends(require_session)):
    try:
        _, probability, prediction = score_transaction(transaction)

        transaction_id = save_transaction(
            transaction=transaction,
            probability=probability,
            prediction=prediction,
            threshold=THRESHOLD,
            owner_session_id=owner_session_id,
        )

        return {
            "transaction_id": transaction_id,
            "prediction": prediction,
            "label": "Potential Fraud" if prediction == 1 else "Likely Legitimate",
            "fraud_probability": round(probability, 6),
            "threshold": THRESHOLD,
            "message": (
                "Flag for further review"
                if prediction == 1
                else "No fraud alert triggered"
            ),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )


# ============================================================
# DEMO TRANSFER: PREDICT FIRST, DO NOT UPDATE BALANCES YET
# ============================================================

@app.post("/transfers/preview")
def preview_transfer(request: TransferPreview, owner_session_id: str = Depends(require_session)):
    sender_id = request.sender_account_id.strip().upper()
    receiver_id = request.receiver_account_id.strip().upper()

    if sender_id == receiver_id:
        raise HTTPException(
            status_code=422,
            detail="Sender and receiver must be different accounts.",
        )

    sender = get_account(sender_id, owner_session_id)
    receiver = get_account(receiver_id, owner_session_id)

    if sender is None:
        raise HTTPException(
            status_code=404,
            detail=f"Sender account '{sender_id}' was not found.",
        )
    if receiver is None:
        raise HTTPException(
            status_code=404,
            detail=f"Receiver account '{receiver_id}' was not found.",
        )
    if request.amount <= 0:
        raise HTTPException(
            status_code=422,
            detail="Transfer amount must be greater than zero.",
        )
    if float(sender["balance"]) < request.amount:
        raise HTTPException(
            status_code=422,
            detail="Insufficient sender balance.",
        )

    sender_old = round(float(sender["balance"]), 2)
    receiver_old = round(float(receiver["balance"]), 2)
    sender_new = round(sender_old - request.amount, 2)
    receiver_new = round(receiver_old + request.amount, 2)

    transaction = Transaction(
        step=request.step,
        type="TRANSFER",
        amount=request.amount,
        oldbalanceOrg=sender_old,
        newbalanceOrig=sender_new,
        oldbalanceDest=receiver_old,
        newbalanceDest=receiver_new,
    )

    try:
        _, probability, prediction = score_transaction(transaction)

        # Save a pending check. The account balances do not change here.
        transaction_id = save_transaction(
            transaction=transaction,
            probability=probability,
            prediction=prediction,
            threshold=THRESHOLD,
            sender_account_id=sender_id,
            receiver_account_id=receiver_id,
            transaction_status="PENDING",
            owner_session_id=owner_session_id,
        )

        return {
            "transaction_id": transaction_id,
            "status": "PENDING",
            "sender": sender,
            "receiver": receiver,
            "amount": round(request.amount, 2),
            "projected_sender_balance": sender_new,
            "projected_receiver_balance": receiver_new,
            "prediction": prediction,
            "label": "Potential Fraud" if prediction == 1 else "Likely Legitimate",
            "fraud_probability": round(probability, 6),
            "threshold": THRESHOLD,
            "message": (
                "The model flagged this transfer. Review it carefully. "
                "Balances will not change unless you explicitly confirm the demo transfer."
                if prediction == 1
                else "The model did not flag this transfer. Review the details before confirming."
            ),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Transfer preview failed: {str(error)}",
        )


# ============================================================
# CONFIRM OR CANCEL A PENDING DEMO TRANSFER
# ============================================================

@app.post("/transfers/{transaction_id}/confirm")
def confirm_transfer(transaction_id: int, decision: TransferDecision, owner_session_id: str = Depends(require_session)):
    try:
        status, details = finalize_transfer(
            transaction_id=transaction_id,
            approve=decision.approve,
            owner_session_id=owner_session_id,
            override_fraud=decision.override_fraud,
        )

        if status == "NOT_FOUND":
            raise HTTPException(status_code=404, detail="Transaction not found.")

        if status == "NOT_PENDING":
            current_status = details.get("transaction_status", "unknown")
            raise HTTPException(
                status_code=409,
                detail=f"This transaction is no longer pending (status: {current_status}).",
            )

        if status == "OVERRIDE_REQUIRED":
            raise HTTPException(
                status_code=409,
                detail=(
                    "The model flagged this transfer. Explicitly acknowledge the warning "
                    "before completing this demo transfer."
                ),
            )

        if status == "ACCOUNT_NOT_FOUND":
            raise HTTPException(
                status_code=404,
                detail="Sender or receiver account could not be found.",
            )

        if status == "INVALID_ACCOUNTS":
            raise HTTPException(
                status_code=422,
                detail="Sender and receiver must be different valid accounts.",
            )

        if status == "INVALID_AMOUNT":
            raise HTTPException(
                status_code=422,
                detail="Transfer amount must be greater than zero.",
            )

        if status == "INSUFFICIENT_FUNDS":
            raise HTTPException(
                status_code=409,
                detail="Insufficient sender balance at confirmation time.",
            )

        if status == "CANCELLED":
            return {
                "status": "CANCELLED",
                "message": "Transfer cancelled. Account balances were not changed.",
            }

        if status == "COMPLETED":
            return {
                "status": "COMPLETED",
                "message": "Demo transfer completed; both account balances were updated.",
                "transaction": details["transaction"],
                "sender": details["sender"],
                "receiver": details["receiver"],
            }

        raise HTTPException(
            status_code=500,
            detail=f"Unexpected transfer status: {status}",
        )

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not finalize transfer: {str(error)}",
        )


# ============================================================
# DEMO DEPOSIT / WITHDRAWAL / PAYMENT PREVIEW
# ============================================================

@app.post("/account-transactions/preview")
def preview_account_transaction(request: AccountTransactionPreview, owner_session_id: str = Depends(require_session)):
    account_id = request.account_id.strip().upper()
    transaction_type = request.transaction_type.strip().upper()

    supported_types = {"CASH_IN", "CASH_OUT", "PAYMENT", "DEBIT"}
    if transaction_type not in supported_types:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported transaction type. Choose CASH_IN (deposit), "
                "CASH_OUT (withdrawal), PAYMENT, or DEBIT."
            ),
        )

    account = get_account(account_id, owner_session_id)
    if account is None:
        raise HTTPException(
            status_code=404,
            detail=f"Demo account '{account_id}' was not found.",
        )

    amount = round(float(request.amount), 2)
    current_balance = round(float(account["balance"]), 2)
    is_deposit = transaction_type == "CASH_IN"

    if amount <= 0:
        raise HTTPException(status_code=422, detail="Amount must be greater than zero.")
    if not is_deposit and current_balance < amount:
        raise HTTPException(status_code=422, detail="Insufficient account balance.")

    projected_balance = round(
        current_balance + amount if is_deposit else current_balance - amount,
        2,
    )

    # Represent the operation using the same PaySim features used in training.
    if is_deposit:
        sender_old, sender_new = 0.0, 0.0
        receiver_old, receiver_new = current_balance, projected_balance
        sender_id, receiver_id = None, account_id
    else:
        sender_old, sender_new = current_balance, projected_balance
        receiver_old, receiver_new = 0.0, 0.0
        sender_id, receiver_id = account_id, None

    transaction = Transaction(
        step=request.step,
        type=transaction_type,
        amount=amount,
        oldbalanceOrg=sender_old,
        newbalanceOrig=sender_new,
        oldbalanceDest=receiver_old,
        newbalanceDest=receiver_new,
    )

    try:
        _, probability, prediction = score_transaction(transaction)
        transaction_id = save_transaction(
            transaction=transaction,
            probability=probability,
            prediction=prediction,
            threshold=THRESHOLD,
            sender_account_id=sender_id,
            receiver_account_id=receiver_id,
            transaction_status="PENDING",
            owner_session_id=owner_session_id,
        )

        operation_name = {
            "CASH_IN": "Deposit",
            "CASH_OUT": "Withdrawal",
            "PAYMENT": "Payment",
            "DEBIT": "Debit",
        }[transaction_type]

        return {
            "transaction_id": transaction_id,
            "status": "PENDING",
            "operation": operation_name,
            "account": account,
            "account_id": account_id,
            "transaction_type": transaction_type,
            "amount": amount,
            "current_balance": current_balance,
            "projected_balance": projected_balance,
            "prediction": prediction,
            "label": "Potential Fraud" if prediction == 1 else "Likely Legitimate",
            "fraud_probability": round(probability, 6),
            "threshold": THRESHOLD,
            "message": (
                "The model flagged this transaction. Review it carefully; "
                "the balance will not change unless you explicitly confirm."
                if prediction == 1
                else "The model did not flag this transaction. Review it before confirming."
            ),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Account transaction preview failed: {str(error)}",
        )


@app.post("/account-transactions/{transaction_id}/confirm")
def confirm_account_transaction(transaction_id: int, decision: TransferDecision, owner_session_id: str = Depends(require_session)):
    try:
        status, details = finalize_account_transaction(
            transaction_id=transaction_id,
            approve=decision.approve,
            owner_session_id=owner_session_id,
            override_fraud=decision.override_fraud,
        )

        if status == "NOT_FOUND":
            raise HTTPException(status_code=404, detail="Transaction not found.")
        if status == "NOT_PENDING":
            current_status = details.get("transaction_status", "unknown")
            raise HTTPException(
                status_code=409,
                detail=f"This transaction is no longer pending (status: {current_status}).",
            )
        if status == "OVERRIDE_REQUIRED":
            raise HTTPException(
                status_code=409,
                detail=(
                    "The model flagged this transaction. Set override_fraud to true "
                    "only after acknowledging the warning in this demo."
                ),
            )
        if status == "ACCOUNT_NOT_FOUND":
            raise HTTPException(status_code=404, detail="Demo account could not be found.")
        if status == "INVALID_ACCOUNTS":
            raise HTTPException(status_code=422, detail="A valid account is required.")
        if status == "INVALID_AMOUNT":
            raise HTTPException(status_code=422, detail="Amount must be greater than zero.")
        if status == "INSUFFICIENT_FUNDS":
            raise HTTPException(status_code=409, detail="Insufficient balance at confirmation time.")
        if status == "UNSUPPORTED_TRANSACTION_TYPE":
            raise HTTPException(status_code=422, detail="Unsupported account transaction type.")
        if status == "CANCELLED":
            return {
                "status": "CANCELLED",
                "message": "Transaction cancelled. The account balance was not changed.",
            }
        if status == "COMPLETED":
            operation = details["operation"]
            return {
                "status": "COMPLETED",
                "message": f"Demo {operation.lower()} completed; the account balance was updated.",
                "operation": operation,
                "transaction": details["transaction"],
                "account": details["account"],
            }

        raise HTTPException(status_code=500, detail=f"Unexpected transaction status: {status}")
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not finalize account transaction: {str(error)}",
        )


# ============================================================
# SHAP EXPLANATION
# ============================================================

@app.post("/explain")
def explain_transaction(transaction: Transaction, owner_session_id: str = Depends(require_session)):
    try:
        features, probability, prediction = score_transaction(transaction)
        shap_output = explainer.shap_values(features)

        if isinstance(shap_output, list):
            shap_values = shap_output[-1][0]
        else:
            shap_values = shap_output[0]

        feature_values = features.iloc[0]
        explanations = []

        for index, feature_name in enumerate(FEATURES):
            explanations.append({
                "feature": feature_name,
                "value": float(feature_values.iloc[index]),
                "shap_value": float(shap_values[index]),
            })

        explanations.sort(
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )

        return {
            "prediction": prediction,
            "label": "Potential Fraud" if prediction == 1 else "Likely Legitimate",
            "fraud_probability": round(probability, 6),
            "threshold": THRESHOLD,
            "explanation_method": "SHAP TreeExplainer",
            "top_features": explanations[:10],
            "interpretation": (
                "Positive SHAP values push the model output toward the fraud class; "
                "negative values push it away. These values explain model output, not causation."
            ),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP explanation failed: {str(error)}",
        )


# ============================================================
# TRANSACTION HISTORY
# ============================================================

def _history_value(transaction: dict, *names, default=None):
    """Return the first available field name from a stored transaction."""
    for name in names:
        if name in transaction and transaction[name] is not None:
            return transaction[name]
    return default


def _shap_history_summary(transaction: dict) -> str:
    """Create a concise, per-transaction SHAP summary for the history table.

    The new model uses step, amount, oldbalanceOrg, oldbalanceDest, and the
    one-hot transaction type. Post-transaction balances are not model features.
    """
    transaction_type = str(_history_value(
        transaction, "type", "transaction_type", default=""
    )).strip().upper()

    step = _history_value(transaction, "step")
    amount = _history_value(transaction, "amount")
    sender_balance = _history_value(
        transaction, "oldbalanceOrg", "oldbalance_org", "old_balance_org", "sender_balance_before"
    )
    receiver_balance = _history_value(
        transaction, "oldbalanceDest", "oldbalance_dest", "old_balance_dest", "receiver_balance_before"
    )

    if not transaction_type or any(v is None for v in (
        step, amount, sender_balance, receiver_balance
    )):
        return "Unavailable: saved transaction is missing model input fields."

    try:
        tx = Transaction(
            step=int(step),
            type=transaction_type,
            amount=float(amount),
            oldbalanceOrg=float(sender_balance),
            # These two fields are accepted by the request model but are not
            # part of the v2 model's nine-feature schema.
            newbalanceOrig=float(_history_value(transaction, "newbalanceOrig", default=0) or 0),
            oldbalanceDest=float(receiver_balance),
            newbalanceDest=float(_history_value(transaction, "newbalanceDest", default=0) or 0),
        )
        features = prepare_features(tx)
        raw_values = explainer.shap_values(features)
        if isinstance(raw_values, list):
            values = raw_values[-1][0]
        else:
            values = raw_values[0]

        display_names = {
            "step": "Transaction time",
            "amount": "Amount",
            "oldbalanceOrg": "Sender balance before",
            "oldbalanceDest": "Receiver balance before",
            "type_CASH_IN": "Deposit type",
            "type_CASH_OUT": "Cash withdrawal type",
            "type_DEBIT": "Debit type",
            "type_PAYMENT": "Payment type",
            "type_TRANSFER": "Transfer type",
        }
        ranked = sorted(
            zip(FEATURES, values),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:3]
        parts = []
        for feature, shap_value in ranked:
            contribution = float(shap_value)
            direction = "raises" if contribution > 0 else "lowers" if contribution < 0 else "does not change"
            parts.append(
                f"{display_names.get(feature, feature)} {direction} fraud score "
                f"(SHAP {contribution:+.3f})"
            )
        return "Top drivers: " + "; ".join(parts) + "."
    except Exception as error:
        return f"Unavailable: could not calculate SHAP ({type(error).__name__})."


@app.get("/transactions")
def transaction_history(
    limit: int = Query(default=100, ge=1, le=1000),
    include_explanations: bool = Query(default=False),
    owner_session_id: str = Depends(require_session),
):
    try:
        transactions = get_recent_transactions(limit, owner_session_id=owner_session_id)
        if include_explanations:
            # Calculate explanations only for the history-table request, not for
            # every dashboard/alert refresh. Label is explicit: current model.
            transactions = [dict(item) for item in transactions]
            for transaction in transactions:
                transaction["shap_explanation"] = _shap_history_summary(transaction)
        return {
            "count": len(transactions),
            "transactions": transactions,
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve transactions: {str(error)}",
        )
