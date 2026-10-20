
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src import database
from src.api import app


SESSION_A = "11111111-1111-4111-8111-111111111111"
SESSION_B = "22222222-2222-4222-8222-222222222222"


class APIIntegrationSessionIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_db_path = database.DB_PATH
        database.DB_PATH = Path(self.tempdir.name) / "api_test.db"

        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        try:
            self.client_context.__exit__(None, None, None)
        finally:
            database.DB_PATH = self.original_db_path
            self.tempdir.cleanup()

    @staticmethod
    def headers(session_id):
        return {"X-Demo-Session": session_id}

    def create_account(self, session_id, account_id, balance=1000):
        return self.client.post(
            "/accounts",
            headers=self.headers(session_id),
            json={
                "account_id": account_id,
                "name": f"Test {account_id}",
                "starting_balance": balance,
            },
        )

    def test_accounts_are_isolated_between_sessions(self):
        response_a = self.create_account(SESSION_A, "ACC1", 1000)
        response_b = self.create_account(SESSION_B, "ACC1", 500)

        self.assertEqual(response_a.status_code, 201, response_a.text)
        self.assertEqual(response_b.status_code, 201, response_b.text)

        accounts_a = self.client.get(
            "/accounts", headers=self.headers(SESSION_A)
        )
        accounts_b = self.client.get(
            "/accounts", headers=self.headers(SESSION_B)
        )

        self.assertEqual(accounts_a.status_code, 200, accounts_a.text)
        self.assertEqual(accounts_b.status_code, 200, accounts_b.text)
        self.assertEqual(accounts_a.json()["count"], 1)
        self.assertEqual(accounts_b.json()["count"], 1)

        balance_a = accounts_a.json()["accounts"][0]["balance"]
        balance_b = accounts_b.json()["accounts"][0]["balance"]
        self.assertEqual(balance_a, 1000)
        self.assertEqual(balance_b, 500)

    def test_foreign_account_cannot_be_opened(self):
        created = self.create_account(SESSION_A, "PRIVATE1", 900)
        self.assertEqual(created.status_code, 201, created.text)

        response = self.client.get(
            "/accounts/PRIVATE1",
            headers=self.headers(SESSION_B),
        )
        self.assertEqual(response.status_code, 404, response.text)

    def test_prediction_history_is_isolated(self):
        transaction = {
            "step": 600,
            "type": "PAYMENT",
            "amount": 100,
            "oldbalanceOrg": 1000,
            "newbalanceOrig": 900,
            "oldbalanceDest": 500,
            "newbalanceDest": 600,
        }

        prediction = self.client.post(
            "/predict",
            headers=self.headers(SESSION_A),
            json=transaction,
        )
        self.assertEqual(prediction.status_code, 200, prediction.text)

        history_a = self.client.get(
            "/transactions", headers=self.headers(SESSION_A)
        )
        history_b = self.client.get(
            "/transactions", headers=self.headers(SESSION_B)
        )

        self.assertEqual(history_a.status_code, 200, history_a.text)
        self.assertEqual(history_b.status_code, 200, history_b.text)
        self.assertEqual(history_a.json()["count"], 1)
        self.assertEqual(history_b.json()["count"], 0)

    def test_missing_or_invalid_session_header_is_rejected(self):
        missing = self.client.get("/accounts")
        invalid = self.client.get(
            "/accounts",
            headers={"X-Demo-Session": "not-a-valid-uuid"},
        )

        self.assertEqual(missing.status_code, 422, missing.text)
        self.assertEqual(invalid.status_code, 400, invalid.text)


if __name__ == "__main__":
    unittest.main()
