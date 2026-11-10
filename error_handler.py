# check for postgresql errors and output a JSON report per query

import json
from enum import Enum
from pathlib import Path

class ErrorType(str, Enum):
    UNDEFINED_TABLE = "UNDEFINED_TABLE"
    UNDEFINED_COLUMN = "UNDEFINED_COLUMN"
    UNDEFINED_FUNCTION = "UNDEFINED_FUNCTION"
    UNDEFINED_OPERATOR = "UNDEFINED_OPERATOR"
    AMBIGUOUS_COLUMN = "AMBIGUOUS_COLUMN"
    DATATYPE_MISMATCH = "DATATYPE_MISMATCH"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    INVALID_TEXT = "INVALID_TEXT"
    INSUFFICIENT_PRIVILEGE = "INSUFFICIENT_PRIVILEGE"
    OTHER = "OTHER"

# these are the main erorors SafeQL tried to correct
# the error codes are the SQLSTATE codes postgres assigns

ERROR_CODES = {
    "42P01": ErrorType.UNDEFINED_TABLE,
    "42703": ErrorType.UNDEFINED_COLUMN,
    "42883": ErrorType.UNDEFINED_FUNCTION,
    "42702": ErrorType.AMBIGUOUS_COLUMN,
    "42804": ErrorType.DATATYPE_MISMATCH,
    "42601": ErrorType.SYNTAX_ERROR,
    "22P02": ErrorType.INVALID_TEXT,
    "42501": ErrorType.INSUFFICIENT_PRIVILEGE,
}

# ------------------------------------------------------------------------------------------------------------

def classify_error(query, error): # Convert a psycopg postgresql exception to a JSON-ready dictionary
    diag = getattr(error, "diag", None)
    code = getattr(error, "sqlstate", None) or getattr(diag, "sqlstate", None)
    message = getattr(diag, "message_primary", None) or str(error)
    position = getattr(diag, "statement_position", None)

    error_type = ERROR_CODES.get(code, ErrorType.OTHER)
    if code == "42883" and "operator does not exist" in message.lower():
        error_type = ErrorType.UNDEFINED_OPERATOR

    return {
        "query": query,
        "status": "ERROR",
        "error_type": error_type.value,
        "sqlstate": code,
        "message": message,
        "position": int(position) if position else None,
        "hint": getattr(diag, "message_hint", None),
        "row_count": None,
    }

def save_result(report, path="error_result.json"): # Overwrite the JSON report with info on the latest query
    Path(path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


# run a query through the database.execute() and write down the output
# needs psycopg, only run checked, read-onnly SELECT queries during first testings
def inspect_query(db, query, values=None, output_path="error_result.json"):
    import psycopg

    try:
        rows = db.execute(query, values)
        report = {
            "query": query,
            "status": "EMPTY_RESULT" if rows == [] else "SUCCESS",
            "error_type": None,
            "sqlstate": None,
            "message": None,
            "position": None,
            "hint": None,
            "row_count": len(rows) if rows is not None else None,
        }
    except psycopg.Error as error:
        report = classify_error(query, error)

    return save_result(report, output_path)