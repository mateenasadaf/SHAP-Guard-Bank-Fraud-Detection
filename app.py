import requests
import pandas as pd
import streamlit as st
import uuid
import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")
if "visitor_session_id" not in st.session_state:
    st.session_state["visitor_session_id"] = str(uuid.uuid4())

API_HEADERS = {
    "X-Demo-Session": st.session_state["visitor_session_id"]
}
REQUEST_TIMEOUT = 15


st.set_page_config(
    page_title="Easy Fraud Checker",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Easy Fraud Checker")
st.write(
    "A demo banking simulator that checks proposed transfers for potential "
    "fraud before you confirm them."
)
st.info(
    "This is a demonstration system using a machine-learning estimate. "
    "A low fraud score does not guarantee that a transfer is safe."
)


# ==================================================
# API HELPERS
# ==================================================

def api_error_message(error):
    """Return a readable message for an HTTP/API error."""
    response = getattr(error, "response", None)
    if response is not None:
        try:
            payload = response.json()
            detail = payload.get("detail", payload)
            return f"API returned HTTP {response.status_code}: {detail}"
        except (ValueError, requests.RequestException):
            return f"API returned HTTP {response.status_code}: {response.text}"
    return str(error)


def get_accounts():
    response = requests.get(
    f"{API_URL}/accounts",
    headers=API_HEADERS,
    timeout=REQUEST_TIMEOUT
)
    response.raise_for_status()
    return response.json().get("accounts", [])


def get_transactions(limit=100, include_explanations=False):
    response = requests.get(
        f"{API_URL}/transactions",
        params={"limit": limit, "include_explanations": include_explanations},
        headers=API_HEADERS,
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json().get("transactions", [])


# ==================================================
# CHECK BACKEND CONNECTION
# ==================================================

try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    health_response.raise_for_status()
except requests.RequestException:
    st.error(
        "The prediction service is not running. Start your FastAPI server "
        "in a separate terminal, then refresh this page."
    )
    st.stop()


# ==================================================
# SESSION STATE
# ==================================================

if "pending_transfer" not in st.session_state:
    st.session_state["pending_transfer"] = None

if "completed_transfer" not in st.session_state:
    st.session_state["completed_transfer"] = None

if "last_explanation" not in st.session_state:
    st.session_state["last_explanation"] = None


# ==================================================
# CREATE DEMO ACCOUNTS
# ==================================================

st.header("1. Demo Accounts")
st.caption(
    "Create accounts for this simulation. Account IDs must be unique within your session. "
    "Starting balances are set only when an account is created."
)

with st.expander("Create a demo account", expanded=False):
    with st.form("create_account_form", clear_on_submit=True):
        new_account_id = st.text_input(
            "Unique account ID",
            placeholder="For example, ACC1003"
        )
        new_account_name = st.text_input(
            "Account holder name",
            placeholder="For example, Alex Demo"
        )
        starting_balance = st.number_input(
            "Starting balance (₹)",
            min_value=0.0,
            value=10000.0,
            step=500.0,
            format="%.2f"
        )
        create_account_clicked = st.form_submit_button(
            "Create Account",
            use_container_width=True
        )

    if create_account_clicked:
        account_payload = {
            "account_id": new_account_id.strip(),
            "name": new_account_name.strip(),
            "starting_balance": float(starting_balance)
        }

        if not account_payload["account_id"] or not account_payload["name"]:
            st.error("Enter both an account ID and an account holder name.")
        else:
            try:
                with st.spinner("Creating demo account..."):
                    create_response = requests.post(
                        f"{API_URL}/accounts",
                        json=account_payload,
                        headers=API_HEADERS,
                        timeout=REQUEST_TIMEOUT
                    )
                    create_response.raise_for_status()
                    created = create_response.json()

                st.success(created.get("message", "Demo account created."))
                st.json(created.get("account", created))
                st.rerun()
            except requests.RequestException as error:
                st.error(f"Could not create account: {api_error_message(error)}")


# Load and show account list.
try:
    accounts = get_accounts()
    accounts_df = pd.DataFrame(accounts)

    if not accounts:
        st.info("No demo accounts exist yet. Create an account to get started.")
    else:
        display_accounts = accounts_df.rename(columns={
            "account_id": "Account ID",
            "name": "Account Holder",
            "balance": "Available Balance (₹)",
            "created_at": "Created At"
        })
        st.dataframe(
            display_accounts,
            use_container_width=True,
            hide_index=True
        )
except requests.RequestException as error:
    st.error(f"Could not load demo accounts: {api_error_message(error)}")
    accounts = []


# ==================================================
# PREVIEW A TRANSFER
# ==================================================

st.divider()
st.header("2. Make a Transfer")
st.write(
    "Choose existing accounts and enter the amount. Balances are retrieved "
    "from the database automatically; you do not enter them manually."
)

if len(accounts) < 2:
    st.warning("Create at least two demo accounts before making a transfer.")
else:
    account_by_id = {account["account_id"]: account for account in accounts}
    account_ids = list(account_by_id.keys())

    def keep_receiver_different_from_sender():
        """Ensure the receiver selection stays valid when sender changes."""
        selected_sender = st.session_state.get("sender_account_select")
        selected_receiver = st.session_state.get("receiver_account_select")
        valid_receivers = [
            account_id for account_id in account_ids
            if account_id != selected_sender
        ]
        if valid_receivers and selected_receiver not in valid_receivers:
            st.session_state["receiver_account_select"] = valid_receivers[0]

    # These widgets intentionally stay outside a Streamlit form so changing
    # the sender immediately reruns the app and refreshes the receiver choices.
    sender_id = st.selectbox(
        "Send money from",
        options=account_ids,
        format_func=lambda account_id: (
            f"{account_id} — {account_by_id[account_id]['name']} "
            f"(Balance: ₹{float(account_by_id[account_id]['balance']):,.2f})"
        ),
        key="sender_account_select",
        on_change=keep_receiver_different_from_sender
    )

    receiver_options = [
        account_id for account_id in account_ids
        if account_id != sender_id
    ]
    # Initialize a valid receiver selection before rendering its widget.
    if st.session_state.get("receiver_account_select") not in receiver_options:
        st.session_state["receiver_account_select"] = receiver_options[0]

    receiver_id = st.selectbox(
        "Send money to",
        options=receiver_options,
        format_func=lambda account_id: (
            f"{account_id} — {account_by_id[account_id]['name']} "
            f"(Balance: ₹{float(account_by_id[account_id]['balance']):,.2f})"
        ),
        key="receiver_account_select"
    )

    transfer_amount = st.number_input(
        "Transfer amount (₹)",
        min_value=0.01,
        value=500.0,
        step=100.0,
        format="%.2f",
        key="transfer_amount"
    )

    preview_clicked = st.button(
        "Check Transfer for Fraud",
        use_container_width=True,
        key="transfer_preview_button"
    )

    if preview_clicked:
        preview_payload = {
            "sender_account_id": sender_id,
            "receiver_account_id": receiver_id,
            "amount": float(transfer_amount)
        }

        try:
            with st.spinner("Checking balances and estimating fraud risk..."):
                preview_response = requests.post(
                    f"{API_URL}/transfers/preview",
                    json=preview_payload,
                    headers=API_HEADERS,
                    timeout=30
                )
                preview_response.raise_for_status()
                preview_result = preview_response.json()

            st.session_state["pending_transfer"] = preview_result
            st.session_state["last_explanation"] = None
            st.session_state["completed_transfer"] = None
            st.rerun()
        except requests.RequestException as error:
            st.error(f"Could not preview transfer: {api_error_message(error)}")


# ==================================================
# SHOW PREVIEW AND CONFIRM / CANCEL
# ==================================================

pending = st.session_state.get("pending_transfer")

if pending:
    st.divider()
    st.header("3. Review Transfer")

    prediction = int(pending.get("prediction", 0))
    probability = float(pending.get("fraud_probability", 0.0))
    sender = pending.get("sender", {})
    receiver = pending.get("receiver", {})

    if prediction == 1:
        st.error(
            "⚠️ The model flagged this transfer for review. "
            "Do not treat this prediction as proof of fraud."
        )
    else:
        st.success(
            "The model did not flag this transfer. Review the details before "
            "confirming; this is not a guarantee that it is safe."
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Transfer Amount", f"₹{float(pending['amount']):,.2f}")
    col2.metric("Estimated Fraud Probability", f"{probability * 100:.4f}%")
    col3.metric("Transaction ID", pending.get("transaction_id", "—"))

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        f"Model threshold: {float(pending.get('threshold', 0.5)) * 100:.0f}%"
    )

    before_col, after_col = st.columns(2)
    with before_col:
        st.subheader("Sender")
        st.write(f"**{sender.get('name', 'Unknown')}** (`{sender.get('account_id', '')}`)")
        st.metric("Current balance", f"₹{float(sender.get('balance', 0)):,.2f}")
        st.metric(
            "Balance after transfer",
            f"₹{float(pending.get('projected_sender_balance', 0)):,.2f}"
        )

    with after_col:
        st.subheader("Receiver")
        st.write(
            f"**{receiver.get('name', 'Unknown')}** "
            f"(`{receiver.get('account_id', '')}`)"
        )
        st.metric("Current balance", f"₹{float(receiver.get('balance', 0)):,.2f}")
        st.metric(
            "Balance after transfer",
            f"₹{float(pending.get('projected_receiver_balance', 0)):,.2f}"
        )

    if prediction == 1:
        override_fraud = st.checkbox(
            "I understand this transfer was flagged and want to override the "
            "demo fraud warning.",
            key=f"override_fraud_{pending.get('transaction_id')}"
        )
    else:
        override_fraud = False

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        confirm_clicked = st.button(
            "Confirm Transfer",
            type="primary",
            use_container_width=True,
            key=f"confirm_{pending.get('transaction_id')}"
        )
    with action_col2:
        cancel_clicked = st.button(
            "Cancel Transfer",
            use_container_width=True,
            key=f"cancel_{pending.get('transaction_id')}"
        )

    if confirm_clicked:
        if prediction == 1 and not override_fraud:
            st.warning(
                "This transfer was flagged. Check the override box before "
                "confirming it in this demo."
            )
        else:
            try:
                with st.spinner("Finalizing transfer..."):
                    confirm_response = requests.post(
                        f"{API_URL}/transfers/{pending['transaction_id']}/confirm",
                        json={
                            "approve": True,
                            "override_fraud": bool(override_fraud)
                        },
                        headers=API_HEADERS,
                        timeout=30
                    )
                    confirm_response.raise_for_status()
                    confirm_result = confirm_response.json()

                if confirm_result.get("status") == "COMPLETED":
                    st.session_state["completed_transfer"] = confirm_result
                    st.session_state["pending_transfer"] = None
                    st.session_state["last_explanation"] = None
                    st.success(confirm_result.get(
                        "message", "Transfer completed successfully."
                    ))
                    st.rerun()
                else:
                    st.warning(confirm_result.get(
                        "message", f"Transfer status: {confirm_result.get('status')}"
                    ))
            except requests.RequestException as error:
                st.error(f"Could not confirm transfer: {api_error_message(error)}")

    if cancel_clicked:
        try:
            cancel_response = requests.post(
                f"{API_URL}/transfers/{pending['transaction_id']}/confirm",
                json={"approve": False, "override_fraud": False},
                headers=API_HEADERS,
                timeout=REQUEST_TIMEOUT
            )
            cancel_response.raise_for_status()
            cancel_result = cancel_response.json()
            st.session_state["pending_transfer"] = None
            st.session_state["last_explanation"] = None
            st.info(cancel_result.get("message", "Transfer cancelled."))
            st.rerun()
        except requests.RequestException as error:
            st.error(f"Could not cancel transfer: {api_error_message(error)}")


# ==================================================
# SHOW COMPLETED TRANSFER
# ==================================================

completed = st.session_state.get("completed_transfer")
if completed:
    st.divider()
    st.header("4. Latest Completed Transfer")
    completed_transaction = completed.get("transaction", {})
    completed_sender = completed.get("sender", {})
    completed_receiver = completed.get("receiver", {})

    st.success(completed.get("message", "Demo transfer completed."))
    c1, c2, c3 = st.columns(3)
    c1.metric("Transaction ID", completed_transaction.get("id", "—"))
    c2.metric("Sender's New Balance", f"₹{float(completed_sender.get('balance', 0)):,.2f}")
    c3.metric("Receiver's New Balance", f"₹{float(completed_receiver.get('balance', 0)):,.2f}")


# ==================================================
# SHAP EXPLANATION FOR LAST PREVIEW
# ==================================================

if pending:
    st.divider()
    st.header("5. Why Did the Model Give This Result?")
    st.write(
        "The following explanation uses the transaction details calculated "
        "from the selected accounts and transfer amount."
    )

    explanation_transaction = {
        "step": 600,
        "type": "TRANSFER",
        "amount": float(pending["amount"]),
        "oldbalanceOrg": float(pending["sender"]["balance"]),
        "newbalanceOrig": float(pending["projected_sender_balance"]),
        "oldbalanceDest": float(pending["receiver"]["balance"]),
        "newbalanceDest": float(pending["projected_receiver_balance"])
    }

    if st.button(
        "Explain This Transfer",
        key=f"explain_{pending.get('transaction_id')}"
    ):
        try:
            with st.spinner("Preparing the explanation..."):
                explanation_response = requests.post(
                    f"{API_URL}/explain",
                    json=explanation_transaction,
                    headers=API_HEADERS,
                    timeout=30
                )
                explanation_response.raise_for_status()
                st.session_state["last_explanation"] = explanation_response.json()
        except requests.RequestException as error:
            st.error(f"Could not explain the result: {api_error_message(error)}")

    explanation = st.session_state.get("last_explanation")
    if explanation:
        features = explanation.get("top_features", [])
        if features:
            explanation_df = pd.DataFrame(features)
            explanation_df["simple_name"] = explanation_df["feature"].map({
                "step": "Transaction time",
                "amount": "Money moved",
                "oldbalanceOrg": "Sender's balance before",
                "newbalanceOrig": "Sender's balance after",
                "oldbalanceDest": "Receiver's balance before",
                "newbalanceDest": "Receiver's balance after",
                "type_CASH_IN": "Money added",
                "type_CASH_OUT": "Cash withdrawal",
                "type_DEBIT": "Money taken from account",
                "type_PAYMENT": "Payment",
                "type_TRANSFER": "Money transfer"
            }).fillna(explanation_df["feature"])

            explanation_df = explanation_df.sort_values(
                "shap_value",
                key=lambda values: values.abs(),
                ascending=False
            )

            chart_df = explanation_df[["simple_name", "shap_value"]].head(10)
            chart_df = chart_df.set_index("simple_name").rename(
                columns={"shap_value": "Effect on the model's fraud score"}
            )
            st.bar_chart(chart_df)

            st.write("**How to read this chart**")
            st.write("- A positive value pushes the model towards predicting fraud.")
            st.write("- A negative value pushes the model away from predicting fraud.")
            st.write("- A larger absolute value means a stronger influence on the model.")

            st.dataframe(
                explanation_df[["simple_name", "value", "shap_value"]].rename(
                    columns={
                        "simple_name": "Transaction detail",
                        "value": "Value used",
                        "shap_value": "Effect on model score"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )
            st.caption(
                "These values explain the model's output; they are not proof "
                "that a transfer is fraudulent."
            )
        else:
            st.info("The API did not return feature-level explanation details.")


# ==================================================
# SAVED TRANSACTIONS / HISTORY
# ==================================================

st.divider()
st.header("6. Transaction History")
st.write("Recent transfer checks and their current status.")

if st.button("Refresh Transaction History"):
    st.rerun()

try:
    transactions = get_transactions(limit=100, include_explanations=True)
    if transactions:
        history_df = pd.DataFrame(transactions)
        history_df = history_df.rename(columns={
            "id": "Transaction Number",
            "timestamp": "Date and Time",
            "step": "Time Step",
            "transaction_type": "Transaction Type",
            "amount": "Amount (₹)",
            "fraud_probability": "Fraud Probability",
            "prediction": "Model Prediction",
            "sender_account_id": "Sender Account",
            "receiver_account_id": "Receiver Account",
            "transaction_status": "Status",
            "shap_explanation": "SHAP Explanation (Current Model)"
        })

        if "Fraud Probability" in history_df.columns:
            history_df["Fraud Probability"] = (
                pd.to_numeric(history_df["Fraud Probability"], errors="coerce") * 100
            ).round(4).astype(str) + "%"

        if "Model Prediction" in history_df.columns:
            history_df["Model Prediction"] = history_df["Model Prediction"].map({
                0: "Not Flagged",
                1: "Flagged"
            }).fillna(history_df["Model Prediction"].astype(str))

        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.caption(
            "SHAP shows the top three feature contributions from the currently loaded model. "
            "Positive contributions raise the model's fraud score; negative contributions lower it. "
            "SHAP values are not percentages and do not prove fraud or causation."
        )
    else:
        st.info("No transactions have been checked yet.")
except requests.RequestException as error:
    st.error(f"Could not load transaction history: {api_error_message(error)}")


# ==================================================
# SUMMARY AND CHARTS
# ==================================================

st.divider()
st.header("7. Summary Dashboard")
st.write("Summary of the latest 1,000 saved transaction checks.")

try:
    saved_transactions = get_transactions(limit=1000)

    if saved_transactions:
        analytics_df = pd.DataFrame(saved_transactions)
        total = len(analytics_df)
        flagged = int((analytics_df["prediction"] == 1).sum())
        not_flagged = total - flagged

        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions Checked", total)
        col2.metric("Flagged by Model", flagged)
        col3.metric("Not Flagged", not_flagged)

        summary_df = pd.DataFrame({
            "Result": ["Not Flagged", "Flagged"],
            "Number of Transactions": [not_flagged, flagged]
        }).set_index("Result")
        st.subheader("Model Flagging Summary")
        st.bar_chart(summary_df)

        st.subheader("Estimated Fraud Probabilities")
        chance_df = analytics_df[["fraud_probability"]].rename(
            columns={"fraud_probability": "Estimated Fraud Chance (%)"}
        )
        chance_df["Estimated Fraud Chance (%)"] *= 100
        st.bar_chart(chance_df)

        st.caption(
            "These charts summarize model predictions, not confirmed fraud cases. "
            "Only the latest 1,000 saved records are shown."
        )
    else:
        st.info("Check a transfer to populate the summary dashboard.")
except requests.RequestException as error:
    st.error(f"Could not load the summary: {api_error_message(error)}")


# ==================================================
# FRAUD ALERTS
# ==================================================

st.divider()
st.header("8. Transfers That Need Review")
st.write(
    "These are transactions flagged by the model. Review them carefully; "
    "the model can make mistakes."
)

try:
    alert_transactions = get_transactions(limit=1000)

    if alert_transactions:
        alert_df = pd.DataFrame(alert_transactions)
        flagged_df = alert_df[alert_df["prediction"] == 1].copy()

        if not flagged_df.empty:
            flagged_df["Chance of Fraud (%)"] = (
                flagged_df["fraud_probability"] * 100
            ).round(4)
            flagged_df = flagged_df.sort_values(
                "fraud_probability",
                ascending=False
            )

            col1, col2 = st.columns(2)
            col1.metric("Transactions to Review", len(flagged_df))
            col2.metric(
                "Highest Estimated Fraud Chance",
                f"{flagged_df['Chance of Fraud (%)'].max():.4f}%"
            )

            alert_columns = [
                column for column in [
                    "id",
                    "timestamp",
                    "sender_account_id",
                    "receiver_account_id",
                    "transaction_type",
                    "amount",
                    "Chance of Fraud (%)",
                    "transaction_status"
                ]
                if column in flagged_df.columns
            ]

            st.dataframe(
                flagged_df[alert_columns].rename(columns={
                    "id": "Transaction Number",
                    "timestamp": "Date and Time",
                    "sender_account_id": "Sender Account",
                    "receiver_account_id": "Receiver Account",
                    "transaction_type": "Transaction Type",
                    "amount": "Amount (₹)",
                    "transaction_status": "Status"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No transactions in the latest 1,000 records were flagged.")
    else:
        st.info("No transactions have been checked yet.")
except requests.RequestException as error:
    st.error(f"Could not load fraud alerts: {api_error_message(error)}")
