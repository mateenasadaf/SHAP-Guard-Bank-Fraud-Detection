import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src import database


class SessionIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_db_path = database.DB_PATH
        database.DB_PATH = Path(self.tempdir.name) / "test.db"
        database.initialize_database()
        self.owner_a = "11111111-1111-4111-8111-111111111111"
        self.owner_b = "22222222-2222-4222-8222-222222222222"

    def tearDown(self):
        database.DB_PATH = self.original_db_path
        self.tempdir.cleanup()

    def test_same_account_id_is_allowed_for_different_owners(self):
        database.create_account("ACC1", "Owner A", 1000, self.owner_a)
        database.create_account("ACC1", "Owner B", 500, self.owner_b)
        self.assertEqual(database.get_account("ACC1", self.owner_a)["balance"], 1000)
        self.assertEqual(database.get_account("ACC1", self.owner_b)["balance"], 500)
        self.assertEqual(len(database.get_accounts(self.owner_a)), 1)
        self.assertEqual(len(database.get_accounts(self.owner_b)), 1)

    def test_foreign_transaction_cannot_be_read_or_confirmed(self):
        database.create_account("FROM", "A sender", 1000, self.owner_a)
        database.create_account("TO", "A receiver", 100, self.owner_a)
        tx = SimpleNamespace(
            step=600, type="TRANSFER", amount=100,
            oldbalanceOrg=1000, newbalanceOrig=900,
            oldbalanceDest=100, newbalanceDest=200,
        )
        transaction_id = database.save_transaction(
            transaction=tx, probability=0.1, prediction=0, threshold=0.5,
            sender_account_id="FROM", receiver_account_id="TO",
            transaction_status="PENDING", owner_session_id=self.owner_a,
        )
        self.assertIsNone(database.get_transaction(transaction_id, self.owner_b))
        status, _ = database.finalize_transfer(
            transaction_id, True, self.owner_b
        )
        self.assertEqual(status, "NOT_FOUND")
        self.assertEqual(database.get_account("FROM", self.owner_a)["balance"], 1000)
        self.assertEqual(len(database.get_recent_transactions(10, self.owner_b)), 0)

    def test_owner_can_finalize_own_transfer(self):
        database.create_account("FROM", "A sender", 1000, self.owner_a)
        database.create_account("TO", "A receiver", 100, self.owner_a)
        tx = SimpleNamespace(
            step=600, type="TRANSFER", amount=100,
            oldbalanceOrg=1000, newbalanceOrig=900,
            oldbalanceDest=100, newbalanceDest=200,
        )
        transaction_id = database.save_transaction(
            tx, 0.1, 0, 0.5, "FROM", "TO", "PENDING", self.owner_a
        )
        status, _ = database.finalize_transfer(transaction_id, True, self.owner_a)
        self.assertEqual(status, "COMPLETED")
        self.assertEqual(database.get_account("FROM", self.owner_a)["balance"], 900)
        self.assertEqual(database.get_account("TO", self.owner_a)["balance"], 200)


if __name__ == "__main__":
    unittest.main()
