import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "fraud_detection.db"
LEGACY_OWNER = "__legacy__"



class ClosingConnection(sqlite3.Connection):
    """SQLite connection that closes when leaving a with block."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def get_connection():
    connection = sqlite3.connect(
        DB_PATH,
        timeout=30,
        factory=ClosingConnection,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection



def initialize_database():
    """Create tables and migrate existing records into a private legacy owner."""
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
                threshold REAL NOT NULL,
                sender_account_id TEXT,
                receiver_account_id TEXT,
                transaction_status TEXT NOT NULL DEFAULT 'CHECKED',
                owner_session_id TEXT NOT NULL DEFAULT '__legacy__'
            )
        """)

        # Older databases may have a transactions table without these columns.
        tx_columns = {
            row["name"] for row in connection.execute(
                "PRAGMA table_info(transactions)"
            ).fetchall()
        }
        tx_new_columns = {
            "sender_account_id": "TEXT",
            "receiver_account_id": "TEXT",
            "transaction_status": "TEXT NOT NULL DEFAULT 'CHECKED'",
            "owner_session_id": "TEXT NOT NULL DEFAULT '__legacy__'",
        }
        for name, definition in tx_new_columns.items():
            if name not in tx_columns:
                connection.execute(
                    f"ALTER TABLE transactions ADD COLUMN {name} {definition}"
                )

        account_exists = connection.execute("""
            SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='accounts'
        """).fetchone()

        if not account_exists:
            connection.execute("""
                CREATE TABLE accounts (
                    account_id TEXT NOT NULL,
                    owner_session_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    balance REAL NOT NULL CHECK (balance >= 0),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (owner_session_id, account_id)
                )
            """)
        else:
            account_columns = {
                row["name"] for row in connection.execute(
                    "PRAGMA table_info(accounts)"
                ).fetchall()
            }
            # Rebuild the old globally-keyed table so duplicate account IDs can
            # exist for separate owners. Existing rows remain under LEGACY_OWNER.
            pk_columns = [
                row["name"] for row in sorted(
                    (r for r in connection.execute("PRAGMA table_info(accounts)").fetchall()
                     if r["pk"]),
                    key=lambda r: r["pk"]
                )
            ]
            if "owner_session_id" not in account_columns or pk_columns != [
                "owner_session_id", "account_id"
            ]:
                connection.execute("ALTER TABLE accounts RENAME TO accounts_migration_old")
                connection.execute("""
                    CREATE TABLE accounts (
                        account_id TEXT NOT NULL,
                        owner_session_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        balance REAL NOT NULL CHECK (balance >= 0),
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (owner_session_id, account_id)
                    )
                """)
                old_columns = {
                    row["name"] for row in connection.execute(
                        "PRAGMA table_info(accounts_migration_old)"
                    ).fetchall()
                }
                if {"account_id", "name", "balance"}.issubset(old_columns):
                    created_expr = "created_at" if "created_at" in old_columns else "CURRENT_TIMESTAMP"
                    owner_expr = "owner_session_id" if "owner_session_id" in old_columns else "?"
                    if "owner_session_id" in old_columns:
                        connection.execute(f"""
                            INSERT OR IGNORE INTO accounts
                                (account_id, owner_session_id, name, balance, created_at)
                            SELECT account_id, COALESCE(owner_session_id, ?), name, balance, {created_expr}
                            FROM accounts_migration_old
                        """, (LEGACY_OWNER,))
                    else:
                        connection.execute(f"""
                            INSERT OR IGNORE INTO accounts
                                (account_id, owner_session_id, name, balance, created_at)
                            SELECT account_id, ?, name, balance, {created_expr}
                            FROM accounts_migration_old
                        """, (LEGACY_OWNER,))
                connection.execute("DROP TABLE accounts_migration_old")

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_accounts_owner
            ON accounts(owner_session_id, account_id)
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_transactions_owner_id
            ON transactions(owner_session_id, id DESC)
        """)


def _owner(owner_session_id):
    owner = str(owner_session_id or "").strip()
    if not owner or owner == LEGACY_OWNER:
        raise ValueError("A valid visitor session is required.")
    return owner


def create_account(account_id, name, starting_balance, owner_session_id):
    """Create an account scoped to a visitor. IDs are case-insensitive per owner."""
    owner_session_id = _owner(owner_session_id)
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
            connection.execute("""
                INSERT INTO accounts
                    (account_id, owner_session_id, name, balance)
                VALUES (?, ?, ?, ?)
            """, (account_id, owner_session_id, name, starting_balance))
    except sqlite3.IntegrityError as error:
        raise ValueError(
            f"Account ID '{account_id}' already exists in this visitor session "
            "or account details are invalid."
        ) from error
    return get_account(account_id, owner_session_id)


def get_account(account_id, owner_session_id):
    """Return an account only if it belongs to this visitor."""
    owner_session_id = _owner(owner_session_id)
    account_id = str(account_id).strip().upper()
    with get_connection() as connection:
        row = connection.execute("""
            SELECT account_id, name, balance, created_at
            FROM accounts
            WHERE account_id = ? AND owner_session_id = ?
        """, (account_id, owner_session_id)).fetchone()
        return dict(row) if row else None


def get_accounts(owner_session_id):
    """Return only accounts belonging to this visitor."""
    owner_session_id = _owner(owner_session_id)
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT account_id, name, balance, created_at
            FROM accounts
            WHERE owner_session_id = ?
            ORDER BY created_at, account_id
        """, (owner_session_id,)).fetchall()
        return [dict(row) for row in rows]


def save_transaction(
    transaction, probability, prediction, threshold,
    sender_account_id=None, receiver_account_id=None,
    transaction_status="CHECKED", owner_session_id=None
):
    """Save a transaction under the requesting visitor's ownership."""
    owner_session_id = _owner(owner_session_id)
    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO transactions (
                step, transaction_type, amount,
                oldbalance_org, newbalance_orig,
                oldbalance_dest, newbalance_dest,
                fraud_probability, prediction, threshold,
                sender_account_id, receiver_account_id,
                transaction_status, owner_session_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            transaction.step, transaction.type, transaction.amount,
            transaction.oldbalanceOrg, transaction.newbalanceOrig,
            transaction.oldbalanceDest, transaction.newbalanceDest,
            probability, prediction, threshold,
            str(sender_account_id).strip().upper() if sender_account_id else None,
            str(receiver_account_id).strip().upper() if receiver_account_id else None,
            transaction_status, owner_session_id,
        ))
        return cursor.lastrowid


def get_transaction(transaction_id, owner_session_id):
    """Return a transaction only if it belongs to this visitor."""
    owner_session_id = _owner(owner_session_id)
    with get_connection() as connection:
        row = connection.execute("""
            SELECT * FROM transactions
            WHERE id = ? AND owner_session_id = ?
        """, (transaction_id, owner_session_id)).fetchone()
        return dict(row) if row else None


def finalize_transfer(transaction_id, approve, owner_session_id, override_fraud=False):
    """Atomically finalize a transfer only within the requesting visitor's data."""
    owner_session_id = _owner(owner_session_id)
    connection = get_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("""
            SELECT * FROM transactions
            WHERE id = ? AND owner_session_id = ?
        """, (transaction_id, owner_session_id)).fetchone()
        if row is None:
            connection.rollback()
            return "NOT_FOUND", None

        transaction = dict(row)
        if transaction.get("transaction_status") != "PENDING":
            connection.rollback()
            return "NOT_PENDING", transaction

        if not approve:
            connection.execute("""
                UPDATE transactions SET transaction_status = 'CANCELLED'
                WHERE id = ? AND owner_session_id = ?
            """, (transaction_id, owner_session_id))
            connection.commit()
            return "CANCELLED", get_transaction(transaction_id, owner_session_id)

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

        sender = connection.execute("""
            SELECT * FROM accounts WHERE account_id = ? AND owner_session_id = ?
        """, (sender_id, owner_session_id)).fetchone()
        receiver = connection.execute("""
            SELECT * FROM accounts WHERE account_id = ? AND owner_session_id = ?
        """, (receiver_id, owner_session_id)).fetchone()
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

        connection.execute("""
            UPDATE accounts SET balance = ?
            WHERE account_id = ? AND owner_session_id = ?
        """, (sender_new, sender_id, owner_session_id))
        connection.execute("""
            UPDATE accounts SET balance = ?
            WHERE account_id = ? AND owner_session_id = ?
        """, (receiver_new, receiver_id, owner_session_id))
        connection.execute("""
            UPDATE transactions
            SET transaction_status = 'COMPLETED',
                oldbalance_org = ?, newbalance_orig = ?,
                oldbalance_dest = ?, newbalance_dest = ?
            WHERE id = ? AND owner_session_id = ?
        """, (
            sender_old, sender_new, receiver_old, receiver_new,
            transaction_id, owner_session_id,
        ))
        connection.commit()
        return "COMPLETED", {
            "transaction": get_transaction(transaction_id, owner_session_id),
            "sender": get_account(sender_id, owner_session_id),
            "receiver": get_account(receiver_id, owner_session_id),
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def finalize_account_transaction(
    transaction_id, approve, owner_session_id, override_fraud=False
):
    """Atomically finalize a single-account transaction for its owner."""
    owner_session_id = _owner(owner_session_id)
    connection = get_connection()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("""
            SELECT * FROM transactions
            WHERE id = ? AND owner_session_id = ?
        """, (transaction_id, owner_session_id)).fetchone()
        if row is None:
            connection.rollback()
            return "NOT_FOUND", None

        transaction = dict(row)
        if transaction.get("transaction_status") != "PENDING":
            connection.rollback()
            return "NOT_PENDING", transaction
        if not approve:
            connection.execute("""
                UPDATE transactions SET transaction_status = 'CANCELLED'
                WHERE id = ? AND owner_session_id = ?
            """, (transaction_id, owner_session_id))
            connection.commit()
            return "CANCELLED", get_transaction(transaction_id, owner_session_id)
        if transaction["prediction"] == 1 and not override_fraud:
            connection.rollback()
            return "OVERRIDE_REQUIRED", transaction

        transaction_type = str(transaction.get("transaction_type", "")).upper()
        amount = round(float(transaction.get("amount", 0)), 2)
        if amount <= 0:
            connection.rollback()
            return "INVALID_AMOUNT", transaction

        if transaction_type == "CASH_IN":
            account_id, operation = transaction.get("receiver_account_id"), "DEPOSIT"
        elif transaction_type in {"CASH_OUT", "DEBIT", "PAYMENT"}:
            account_id = transaction.get("sender_account_id")
            operation = "WITHDRAWAL" if transaction_type == "CASH_OUT" else "DEBIT"
        else:
            connection.rollback()
            return "UNSUPPORTED_TRANSACTION_TYPE", transaction
        if not account_id:
            connection.rollback()
            return "INVALID_ACCOUNTS", transaction

        account = connection.execute("""
            SELECT account_id, name, balance, created_at FROM accounts
            WHERE account_id = ? AND owner_session_id = ?
        """, (str(account_id).strip().upper(), owner_session_id)).fetchone()
        if account is None:
            connection.rollback()
            return "ACCOUNT_NOT_FOUND", transaction

        old_balance = round(float(account["balance"]), 2)
        if operation != "DEPOSIT" and old_balance < amount:
            connection.rollback()
            return "INSUFFICIENT_FUNDS", transaction
        new_balance = round(
            old_balance + amount if operation == "DEPOSIT" else old_balance - amount, 2
        )
        connection.execute("""
            UPDATE accounts SET balance = ?
            WHERE account_id = ? AND owner_session_id = ?
        """, (new_balance, account["account_id"], owner_session_id))

        if operation == "DEPOSIT":
            old_org = float(transaction.get("oldbalance_org", 0))
            new_org = float(transaction.get("newbalance_orig", 0))
            old_dest, new_dest = old_balance, new_balance
        else:
            old_org, new_org = old_balance, new_balance
            old_dest = float(transaction.get("oldbalance_dest", 0))
            new_dest = float(transaction.get("newbalance_dest", 0))

        connection.execute("""
            UPDATE transactions
            SET transaction_status = 'COMPLETED',
                oldbalance_org = ?, newbalance_orig = ?,
                oldbalance_dest = ?, newbalance_dest = ?
            WHERE id = ? AND owner_session_id = ?
        """, (
            old_org, new_org, old_dest, new_dest, transaction_id, owner_session_id
        ))
        connection.commit()
        return "COMPLETED", {
            "operation": operation,
            "transaction": get_transaction(transaction_id, owner_session_id),
            "account": get_account(account["account_id"], owner_session_id),
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_recent_transactions(limit=100, owner_session_id=None):
    """Return newest transaction records belonging to the requesting visitor."""
    owner_session_id = _owner(owner_session_id)
    limit = int(limit)
    if limit < 1:
        raise ValueError("Limit must be at least 1.")
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT * FROM transactions
            WHERE owner_session_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (owner_session_id, limit)).fetchall()
        return [dict(row) for row in rows]
