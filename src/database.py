import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "fraud_detection.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    """Create required tables and add compatible fields to an existing DB."""
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                step INTEGER NOT NULL,
                transaction_type TEXT NOT NULL,
                amount REAL NOT NULL,
                oldbalance_org REAL NOT NULL,
                newbalance_orig REAL NOT NULL,
                oldbalance_dest REAL NOT NULL,
                newbalance_dest REAL NOT NULL,
                fraud_probability REAL NOT NULL,
                prediction INTEGER NOT NULL,
                threshold REAL NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                balance REAL NOT NULL CHECK (balance >= 0),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        existing_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(transactions)"
            ).fetchall()
        }

        # Add these columns to existing databases without deleting old rows.
        new_columns = {
            "sender_account_id": "TEXT",
            "receiver_account_id": "TEXT",
            "transaction_status": "TEXT NOT NULL DEFAULT 'CHECKED'",
        }
        for column_name, column_definition in new_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE transactions ADD COLUMN "
                    f"{column_name} {column_definition}"
                )


def create_account(account_id, name, starting_balance):
    """Create a persistent demo account. Account IDs are case-insensitive."""
    account_id = str(account_id).strip().upper()
    name = str(name).strip()
    starting_balance = float(starting_balance)

    if not account_id:
        raise ValueError("Account ID is required.")
    if not name:
        raise ValueError("Account holder name is required.")
    if starting_balance < 0:
        raise ValueError("Starting balance cannot be negative.")

    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO accounts (account_id, name, balance)
                VALUES (?, ?, ?)
                """,
                (account_id, name, starting_balance),
            )
    except sqlite3.IntegrityError as error:
        raise ValueError(
            f"Account ID '{account_id}' already exists or account details are invalid."
        ) from error

    return get_account(account_id)


def get_account(account_id):
    """Return one account as a dictionary, or None if it does not exist."""
    account_id = str(account_id).strip().upper()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT account_id, name, balance, created_at
            FROM accounts
            WHERE account_id = ?
            """,
            (account_id,),
        ).fetchone()
        return dict(row) if row else None


def get_accounts():
    """Return all demo accounts."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT account_id, name, balance, created_at
            FROM accounts
            ORDER BY created_at, account_id
            """
        ).fetchall()
        return [dict(row) for row in rows]


def save_transaction(
    transaction,
    probability,
    prediction,
    threshold,
    sender_account_id=None,
    receiver_account_id=None,
    transaction_status="CHECKED",
):
    """Save a prediction while retaining compatibility with the old API."""
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO transactions (
                step, transaction_type, amount,
                oldbalance_org, newbalance_orig,
                oldbalance_dest, newbalance_dest,
                fraud_probability, prediction, threshold,
                sender_account_id, receiver_account_id, transaction_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transaction.step,
                transaction.type,
                transaction.amount,
                transaction.oldbalanceOrg,
                transaction.newbalanceOrig,
                transaction.oldbalanceDest,
                transaction.newbalanceDest,
                probability,
                prediction,
                threshold,
                (
                    str(sender_account_id).strip().upper()
                    if sender_account_id else None
                ),
                (
                    str(receiver_account_id).strip().upper()
                    if receiver_account_id else None
                ),
                transaction_status,
            ),
        )
        return cursor.lastrowid


def get_transaction(transaction_id):
    """Return a saved transaction as a dictionary, or None."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM transactions WHERE id = ?",
            (transaction_id,),
        ).fetchone()
        return dict(row) if row else None


def finalize_transfer(transaction_id, approve, override_fraud=False):
    """
    Finalize a pending demo transfer.

    The sender debit, receiver credit, and transaction status change are
    committed together. A flagged transaction requires an explicit override.
    Returns (status, details).
    """
    connection = get_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM transactions WHERE id = ?",
            (transaction_id,),
        ).fetchone()

        if row is None:
            connection.rollback()
            return "NOT_FOUND", None

        transaction = dict(row)
        if transaction.get("transaction_status") != "PENDING":
            connection.rollback()
            return "NOT_PENDING", transaction

        if not approve:
            connection.execute(
                """
                UPDATE transactions
                SET transaction_status = 'CANCELLED'
                WHERE id = ?
                """,
                (transaction_id,),
            )
            connection.commit()
            return "CANCELLED", get_transaction(transaction_id)

        if transaction["prediction"] == 1 and not override_fraud:
            connection.rollback()
            return "OVERRIDE_REQUIRED", transaction

        sender_id = transaction.get("sender_account_id")
        receiver_id = transaction.get("receiver_account_id")
        amount = float(transaction["amount"])

        if not sender_id or not receiver_id or sender_id == receiver_id:
            connection.rollback()
            return "INVALID_ACCOUNTS", transaction
        if amount <= 0:
            connection.rollback()
            return "INVALID_AMOUNT", transaction

        sender = connection.execute(
            "SELECT * FROM accounts WHERE account_id = ?",
            (sender_id,),
        ).fetchone()
        receiver = connection.execute(
            "SELECT * FROM accounts WHERE account_id = ?",
            (receiver_id,),
        ).fetchone()

        if sender is None or receiver is None:
            connection.rollback()
            return "ACCOUNT_NOT_FOUND", transaction
        if float(sender["balance"]) < amount:
            connection.rollback()
            return "INSUFFICIENT_FUNDS", transaction

        sender_old = round(float(sender["balance"]), 2)
        receiver_old = round(float(receiver["balance"]), 2)
        sender_new = round(sender_old - amount, 2)
        receiver_new = round(receiver_old + amount, 2)

        connection.execute(
            "UPDATE accounts SET balance = ? WHERE account_id = ?",
            (sender_new, sender_id),
        )
        connection.execute(
            "UPDATE accounts SET balance = ? WHERE account_id = ?",
            (receiver_new, receiver_id),
        )
        connection.execute(
            """
            UPDATE transactions
            SET transaction_status = 'COMPLETED',
                oldbalance_org = ?,
                newbalance_orig = ?,
                oldbalance_dest = ?,
                newbalance_dest = ?
            WHERE id = ?
            """,
            (
                sender_old,
                sender_new,
                receiver_old,
                receiver_new,
                transaction_id,
            ),
        )
        connection.commit()

        return "COMPLETED", {
            "transaction": get_transaction(transaction_id),
            "sender": get_account(sender_id),
            "receiver": get_account(receiver_id),
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()



def finalize_account_transaction(transaction_id, approve, override_fraud=False):
    """Finalize a pending single-account deposit, withdrawal, or payment.

    CASH_IN credits receiver_account_id. CASH_OUT, DEBIT, and PAYMENT debit
    sender_account_id. The balance update and transaction status change are
    committed atomically. Returns (status, details).
    """
    connection = get_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM transactions WHERE id = ?",
            (transaction_id,),
        ).fetchone()

        if row is None:
            connection.rollback()
            return "NOT_FOUND", None

        transaction = dict(row)
        if transaction.get("transaction_status") != "PENDING":
            connection.rollback()
            return "NOT_PENDING", transaction

        if not approve:
            connection.execute(
                "UPDATE transactions SET transaction_status = 'CANCELLED' WHERE id = ?",
                (transaction_id,),
            )
            connection.commit()
            return "CANCELLED", get_transaction(transaction_id)

        if transaction["prediction"] == 1 and not override_fraud:
            connection.rollback()
            return "OVERRIDE_REQUIRED", transaction

        transaction_type = str(transaction.get("transaction_type", "")).upper()
        amount = round(float(transaction.get("amount", 0)), 2)
        if amount <= 0:
            connection.rollback()
            return "INVALID_AMOUNT", transaction

        if transaction_type == "CASH_IN":
            account_id = transaction.get("receiver_account_id")
            operation = "DEPOSIT"
        elif transaction_type in {"CASH_OUT", "DEBIT", "PAYMENT"}:
            account_id = transaction.get("sender_account_id")
            operation = "WITHDRAWAL" if transaction_type == "CASH_OUT" else "DEBIT"
        else:
            connection.rollback()
            return "UNSUPPORTED_TRANSACTION_TYPE", transaction

        if not account_id:
            connection.rollback()
            return "INVALID_ACCOUNTS", transaction

        account = connection.execute(
            "SELECT account_id, name, balance, created_at FROM accounts WHERE account_id = ?",
            (str(account_id).strip().upper(),),
        ).fetchone()
        if account is None:
            connection.rollback()
            return "ACCOUNT_NOT_FOUND", transaction

        old_balance = round(float(account["balance"]), 2)
        if operation != "DEPOSIT" and old_balance < amount:
            connection.rollback()
            return "INSUFFICIENT_FUNDS", transaction

        new_balance = round(old_balance + amount if operation == "DEPOSIT" else old_balance - amount, 2)
        connection.execute(
            "UPDATE accounts SET balance = ? WHERE account_id = ?",
            (new_balance, account["account_id"]),
        )

        if operation == "DEPOSIT":
            old_org = float(transaction.get("oldbalance_org", 0))
            new_org = float(transaction.get("newbalance_orig", 0))
            old_dest, new_dest = old_balance, new_balance
        else:
            old_org, new_org = old_balance, new_balance
            old_dest = float(transaction.get("oldbalance_dest", 0))
            new_dest = float(transaction.get("newbalance_dest", 0))

        connection.execute(
            """UPDATE transactions
               SET transaction_status = 'COMPLETED',
                   oldbalance_org = ?, newbalance_orig = ?,
                   oldbalance_dest = ?, newbalance_dest = ?
               WHERE id = ?""",
            (old_org, new_org, old_dest, new_dest, transaction_id),
        )
        connection.commit()
        return "COMPLETED", {
            "operation": operation,
            "transaction": get_transaction(transaction_id),
            "account": get_account(account["account_id"]),
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_recent_transactions(limit=100):
    """Return newest transaction records first."""
    limit = int(limit)
    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM transactions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
