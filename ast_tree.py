import copy
import re
from typing import Any, Dict, List, Optional

from sqlglot import exp

def extract_faulty_token(report: dict) -> Optional[str]:
    message = report.get("message", "")
    
    # 1. Match tokens enclosed in double quotes (common in Postgres error messages)
    # Examples: relation "users_tbl" does not exist -> users_tbl
    #           column "usr_name" does not exist    -> usr_name
    match = re.search(r'["\']([^"\']+)["\']', message)
    if match:
        return match.group(1)

    # 2. Fallback: derive token from 1-based character position
    position = report.get("position")
    query = report.get("query", "")
    if position and 0 < position <= len(query):
        idx = position - 1
        # Scan forward to grab the identifier starting around this index
        ident_match = re.match(r'[a-zA-Z0-9_]+', query[idx:])
        if ident_match:
            return ident_match.group(0)

    return None

class ASTMutator:
    def __init__(self, dialect: str = "postgres"):
        self.dialect = dialect

    def generate_candidate_trees(
        self,
        current_ast: exp.Expr,
        error_report: Dict[str, Any],
        schema_info: Dict[str, Any]
    ) -> List[exp.Expr]:
        """
        Takes the current AST, the dict returned by Gonzalo's classify_error,
        and Step 2 schema metadata to produce mutated candidate ASTs.
        """
        error_type = error_report.get("error_type")
        faulty_token = extract_faulty_token(error_report)

        if not faulty_token:
            return []

        faulty_token_lower = faulty_token.lower()

        if error_type == "UNDEFINED_COLUMN":
            return self._repair_undefined_column(current_ast, faulty_token_lower, schema_info)
        elif error_type == "UNDEFINED_TABLE":
            return self._repair_undefined_table(current_ast, faulty_token_lower, schema_info)
        elif error_type == "AMBIGUOUS_COLUMN":
            return self._repair_ambiguous_column(current_ast, faulty_token_lower)
        elif error_type in ("DATATYPE_MISMATCH", "UNDEFINED_OPERATOR"):
            return self._repair_type_mismatch(current_ast, faulty_token_lower)

        return []

    def _repair_undefined_column(
        self,
        ast: exp.Expr,
        target_token: str,
        schema_info: Dict[str, Any]
    ) -> List[exp.Expr]:
        available_columns = schema_info.get("columns", [])

        if not any(
            col_node.name.lower() == target_token
            for col_node in ast.find_all(exp.Column)
        ):
            return []

        candidates = []
        for candidate_col in available_columns:
            if candidate_col.lower() == target_token:
                continue
            cloned_ast = ast.copy()

            def replace_col(node):
                if isinstance(node, exp.Column) and node.name.lower() == target_token:
                    new_node = copy.deepcopy(node)
                    new_node.set("this", exp.to_identifier(candidate_col))
                    return new_node
                return node

            candidates.append(cloned_ast.transform(replace_col))

        return candidates

    def _repair_undefined_table(
        self,
        ast: exp.Expr,
        target_token: str,
        schema_info: Dict[str, Any]
    ) -> List[exp.Expr]:
        available_tables = schema_info.get("tables", [])

        if not any(
            table_node.name.lower() == target_token
            for table_node in ast.find_all(exp.Table)
        ):
            return []

        candidates = []
        for candidate_tbl in available_tables:
            if candidate_tbl.lower() == target_token:
                continue
            cloned_ast = ast.copy()

            def replace_tbl(node):
                if isinstance(node, exp.Table) and node.name.lower() == target_token:
                    new_node = copy.deepcopy(node)
                    new_node.set("this", exp.to_identifier(candidate_tbl))
                    return new_node
                return node

            candidates.append(cloned_ast.transform(replace_tbl))

        return candidates

    def _repair_ambiguous_column(
        self,
        ast: exp.Expr,
        target_token: str
    ) -> List[exp.Expr]:
        matching_columns = [
            node
            for node in ast.find_all(exp.Column)
            if node.name.lower() == target_token and not node.table
        ]
        if not matching_columns:
            return []

        tables_in_query = list(dict.fromkeys(
            table.alias_or_name for table in ast.find_all(exp.Table)
        ))
        candidates = []

        for tbl_name in tables_in_query:
            cloned_ast = ast.copy()

            def qualify(node):
                if (
                    isinstance(node, exp.Column)
                    and node.name.lower() == target_token
                    and not node.table
                ):
                    new_node = copy.deepcopy(node)
                    new_node.set("table", exp.to_identifier(tbl_name))
                    return new_node
                return node

            candidates.append(cloned_ast.transform(qualify))

        return candidates

    def _repair_type_mismatch(
        self,
        ast: exp.Expr,
        target_token: str
    ) -> List[exp.Expr]:
        candidates = []
        target_casts = ["TEXT", "INTEGER", "BIGINT", "NUMERIC", "BOOLEAN", "TIMESTAMP"]

        if not any(
            col_node.name.lower() == target_token
            for col_node in ast.find_all(exp.Column)
        ):
            return []

        for cast_type in target_casts:
            cloned_ast = ast.copy()

            def wrap_cast(node):
                if isinstance(node, exp.Column) and node.name.lower() == target_token:
                    return exp.Cast(
                        this=copy.deepcopy(node),
                        to=exp.DataType.build(cast_type)
                    )
                return node

            candidates.append(cloned_ast.transform(wrap_cast))

        return candidates