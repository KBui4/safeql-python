"""Run with: python -m unittest test_error_handling -v"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from error_handler import ErrorType, classify_error, inspect_query, save_result


class FakePsycopgError(Exception):
    def __init__(self, code, message, position=None, hint=None):
        super().__init__(message)
        self.sqlstate = code
        self.diag = types.SimpleNamespace(
            sqlstate=code,
            message_primary=message,
            statement_position=position,
            message_hint=hint,
        )


class FakeDb:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def execute(self, query, values=None):
        if self.error:
            raise self.error
        return self.result


class ErrorHandlerTests(unittest.TestCase):
    def test_missing_column(self):
        query = "SELECT name, dept FROM employees;"
        error = FakePsycopgError("42703", 'column "dept" does not exist', "14")
        report = classify_error(query, error)
        self.assertEqual(report["error_type"], ErrorType.UNDEFINED_COLUMN.value)
        self.assertEqual(report["sqlstate"], "42703")
        self.assertEqual(report["position"], 14)

    def test_missing_table(self):
        report = classify_error("SELECT * FROM employes;", FakePsycopgError("42P01", 'relation "employes" does not exist'))
        self.assertEqual(report["error_type"], "UNDEFINED_TABLE")

    def test_undefined_function(self):
        report = classify_error("SELECT funky(1);", FakePsycopgError("42883", "function funky(integer) does not exist"))
        self.assertEqual(report["error_type"], "UNDEFINED_FUNCTION")

    def test_undefined_operator(self):
        report = classify_error("SELECT 1 + 'x';", FakePsycopgError("42883", "operator does not exist: integer + text"))
        self.assertEqual(report["error_type"], "UNDEFINED_OPERATOR")

    def test_unknown_error(self):
        report = classify_error("SELECT 1", FakePsycopgError("99999", "unexpected"))
        self.assertEqual(report["error_type"], "OTHER")

    def test_write_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "error_result.json"
            save_result(classify_error("SELECT * FROM nope;", FakePsycopgError("42P01", "missing")), str(path))
            self.assertEqual(json.loads(path.read_text())["error_type"], "UNDEFINED_TABLE")

    def test_integrated_query_outcomes(self):
        # Stub psycopg for offline tests; real integration uses the installed package.
        fake_psycopg = types.ModuleType("psycopg")
        setattr(fake_psycopg, "Error", FakePsycopgError)
        with patch.dict(sys.modules, {"psycopg": fake_psycopg}), tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "error_result.json"
            report = inspect_query(FakeDb(result=[]), "SELECT * FROM t;", output_path=str(path))
            self.assertEqual(report["status"], "EMPTY_RESULT")
            self.assertEqual(json.loads(path.read_text())["status"], "EMPTY_RESULT")

            report = inspect_query(FakeDb(result=[(1,)]), "SELECT * FROM t;", output_path=str(path))
            self.assertEqual(report["status"], "SUCCESS")
            self.assertEqual(report["row_count"], 1)

            error = FakePsycopgError("42703", 'column "abc" does not exist', "8")
            report = inspect_query(FakeDb(error=error), "SELECT abc FROM t;", output_path=str(path))
            self.assertEqual(report["status"], "ERROR")
            self.assertEqual(json.loads(path.read_text())["error_type"], "UNDEFINED_COLUMN")


if __name__ == "__main__":
    unittest.main()