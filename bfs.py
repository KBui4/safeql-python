from __future__ import annotations

import heapq
import math
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlglot import exp, parse_one

from ast_tree import SafeQLASTMutator, extract_faulty_token

# Embedding backend
_EMBEDDER_FAILED = False


def _embed(text: str) -> List[float]:
    global _embedder, _EMBEDDER_FAILED
    if _EMBEDDER_FAILED:
        return _lexical_vec(text)
    if _embedder is None:
        try:
            from fastembed import TextEmbedding

            _embedder = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
        except Exception:
            _EMBEDDER_FAILED = True
            return _lexical_vec(text)
    return [float(x) for x in next(iter(_embedder.embed([text])))]


def _lexical_vec(text: str, dim: int = 64) -> List[float]:
    import hashlib

    digest = hashlib.sha256(text.lower().encode()).digest()
    return [b / 255.0 - 0.5 for b in digest[:dim]]


def _cosine_distance(a: List[float], b: List[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    return 1.0 - dot / (na * nb)


def _token_sequence(ast: exp.Expr) -> List[str]:

    # flatten the AST into a token sequence
    toks: List[str] = []
    for node in ast.walk():
        if isinstance(node, exp.Table):
            toks.append(f"tbl:{node.name.lower()}")
        elif isinstance(node, exp.Column):
            toks.append(f"col:{node.name.lower()}")
        elif isinstance(node, exp.Literal):
            toks.append(f"lit:{node.this!s}")
        elif isinstance(node, exp.Anonymous):
            toks.append(f"fn:{str(node.this).lower()}")
        elif isinstance(
            node,
            (
                exp.Select,
                exp.From,
                exp.Where,
                exp.Join,
                exp.Group,
                exp.Order,
                exp.Limit,
                exp.And,
                exp.Or,
                exp.EQ,
                exp.NEQ,
                exp.GT,
                exp.GTE,
                exp.LT,
                exp.LTE,
                exp.Star,
                exp.Cast,
            ),
        ):
            toks.append(f"op:{type(node).__name__.lower()}")
    return toks


def _bare(tok: str) -> str:
    return tok.split(":", 1)[1] if ":" in tok else tok


# distance metric for two strings
def _levenshtein(a: List[str], b: List[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, tb in enumerate(b, 1):
            cur[j] = min(
                cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (0 if ta == tb else 1)
            )
        prev = cur
    return prev[-1]


def d_struct(a: exp.Expr, b: exp.Expr) -> float:
    sa, sb = _token_sequence(a), _token_sequence(b)
    return _levenshtein(sa, sb) / max(len(sa), len(sb), 1)


def d_embed(a: exp.Expr, b: exp.Expr) -> float:
    ca, cb = Counter(_token_sequence(a)), Counter(_token_sequence(b))
    diff_a = list((ca - cb).elements())
    diff_b = list((cb - ca).elements())
    n = max(len(diff_a), len(diff_b))
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        ta = _bare(diff_a[i]) if i < len(diff_a) else ""
        tb = _bare(diff_b[i]) if i < len(diff_b) else ""
        total += _cosine_distance(_embed(ta), _embed(tb))
    return total / n


def semantic_distance(a: exp.Expr, b: exp.Expr, alpha: float = 0.35) -> float:
    return alpha * d_struct(a, b) + (1.0 - alpha) * d_embed(a, b)


def _refinement_category(error_type: str) -> str:
    return {
        "UNDEFINED_TABLE": "relation",
        "UNDEFINED_COLUMN": "attribute",
        "AMBIGUOUS_COLUMN": "attribute_qualification",
        "DATATYPE_MISMATCH": "function_or_cast",
        "UNDEFINED_OPERATOR": "function_or_cast",
    }.get(error_type, "unknown")


# --> Top-K Pruning
def _multiset_diff(a: List[str], b: List[str]) -> List[str]:
    return list((Counter(a) - Counter(b)).elements())


# --> Top-K Pruning
def _extract_new_token(
    original_ast: exp.Expr, candidate: exp.Expr, error_type: str
) -> Optional[str]:
    if error_type == "UNDEFINED_TABLE":
        d = _multiset_diff(
            [t.name.lower() for t in candidate.find_all(exp.Table)],
            [t.name.lower() for t in original_ast.find_all(exp.Table)],
        )
        return d[0] if d else None

    if error_type == "UNDEFINED_COLUMN":
        d = _multiset_diff(
            [c.name.lower() for c in candidate.find_all(exp.Column)],
            [c.name.lower() for c in original_ast.find_all(exp.Column)],
        )
        return d[0] if d else None

    if error_type == "AMBIGUOUS_COLUMN":
        d = _multiset_diff(
            [
                f"{c.table}.{c.name}".lower()
                for c in candidate.find_all(exp.Column)
                if c.table
            ],
            [
                f"{c.table}.{c.name}".lower()
                for c in original_ast.find_all(exp.Column)
                if c.table
            ],
        )
        return d[0].split(".", 1)[0] if d else None

    if error_type in ("DATATYPE_MISMATCH", "UNDEFINED_OPERATOR"):
        for cast in candidate.find_all(exp.Cast):
            try:
                return cast.args["to"].sql(dialect="postgres").upper()
            except Exception:
                return None
        return None

    return None


# --> Type-Based Pruning
_CMP_OPS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)

_NUMERIC = {
    "int",
    "integer",
    "bigint",
    "smallint",
    "numeric",
    "decimal",
    "real",
    "double",
    "float",
    "float4",
    "float8",
    "serial",
    "bigserial",
    "int2",
    "int4",
    "int8",
}
_STRING = {
    "text",
    "varchar",
    "char",
    "character",
    "character varying",
    "nvarchar",
    "name",
    "bpchar",
    "citext",
}
_TEMPORAL = {"date", "timestamp", "timestamptz", "time", "timetz"}
_BOOL = {"bool", "boolean"}


# --> Type-Based Pruning
def _type_family(t: Optional[str]) -> str:
    if t is None:
        return "unknown"
    t = t.lower().split("(")[0].strip()
    if t in _NUMERIC:
        return "numeric"
    if t in _STRING:
        return "string"
    if t in _TEMPORAL:
        return "temporal"
    if t in _BOOL:
        return "bool"
    return "other"


# --> Type-Based Pruning
def _types_compatible(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return True
    fa, fb = _type_family(a), _type_family(b)
    if "unknown" in (fa, fb) or fa == fb:
        return True
    return {fa, fb} <= {"numeric", "string", "bool"}


# --> Type-Based Pruning
def _literal_type(lit: exp.Literal) -> Optional[str]:
    if lit.is_string:
        return "text"
    if lit.is_number:
        s = str(lit.this)
        return "numeric" if ("." in s or "e" in s.lower()) else "int"
    return None


# --> Type-Based Pruning
def _column_type(col: exp.Column, schema_info: Dict[str, Any]) -> Optional[str]:
    parent = col.parent
    if isinstance(parent, exp.Cast):
        try:
            return parent.args["to"].sql(dialect="postgres").lower()
        except Exception:
            pass
    types = schema_info.get("column_types", {})
    name = col.name.lower()
    if name in types:
        return types[name]
    if col.table:
        q = f"{col.table}.{col.name}".lower()
        if q in types:
            return types[q]
    return None


# --> Type-Based Pruning
def _type_check_candidate(
    candidate: exp.Expr, target_token: str, schema_info: Dict[str, Any]
) -> bool:
    target = target_token.lower()
    for col in candidate.find_all(exp.Column):
        if col.name.lower() != target:
            continue
        parent = col.parent
        if parent is None:
            continue

        # comparison with a literal / another column
        if isinstance(parent, _CMP_OPS):
            other = parent.right if parent.left is col else parent.left
            if isinstance(other, exp.Literal):
                if not _types_compatible(
                    _column_type(col, schema_info), _literal_type(other)
                ):
                    return False
            elif isinstance(other, exp.Column) and not _types_compatible(
                _column_type(col, schema_info),
                _column_type(other, schema_info),
            ):
                return False

        # function argument
        if isinstance(parent, exp.Anonymous):
            fname = str(parent.this).lower()
            sigs = schema_info.get("function_signatures", {})
            if fname in sigs:
                for i, arg in enumerate(parent.expressions):
                    if (
                        i < len(sigs[fname])
                        and (
                            arg is col
                            or (isinstance(arg, exp.Cast) and arg.this is col)
                        )
                        and not _types_compatible(
                            sigs[fname][i], _column_type(col, schema_info)
                        )
                    ):
                        return False
            if (
                fname in ("sum", "avg")
                and _type_family(_column_type(col, schema_info)) == "string"
            ):
                return False

    return True


### Type-Based Pruning (main function)


def type_based_pruning(
    candidates: List[exp.Expr],
    error_type: str,
    faulty_token: str,
    schema_info: Dict[str, Any],
) -> List[exp.Expr]:
    if not faulty_token:
        return candidates
    return [
        c for c in candidates if _type_check_candidate(c, faulty_token, schema_info)
    ]


### Top-K Pruning (main function)


def top_k_pruning(
    candidates: List[exp.Expr],
    error_type: str,
    faulty_token: str,
    current_ast: exp.Expr,
    K: int = 3,
    embed_fn: Callable[[str], List[float]] = _embed,
) -> List[exp.Expr]:
    if K <= 0 or len(candidates) <= K:
        return candidates
    try:
        ref_vec = embed_fn(faulty_token)
    except Exception:
        return candidates

    buckets: Dict[str, List[Tuple[float, exp.Expr]]] = {}
    for cand in candidates:
        new_tok = _extract_new_token(current_ast, cand, error_type) or ""
        try:
            dist = _cosine_distance(ref_vec, embed_fn(new_tok))
        except Exception:
            dist = float("inf")
        buckets.setdefault(_refinement_category(error_type), []).append((dist, cand))

    out: List[exp.Expr] = []
    for items in buckets.values():
        items.sort(key=lambda t: t[0])
        out.extend(c for _, c in items[:K])
    return out


def prune_refinements(
    candidates: List[exp.Expr],
    error_report: Dict[str, Any],
    schema_info: Dict[str, Any],
    current_ast: exp.Expr,
    K: int = 3,
    embed_fn: Callable[[str], List[float]] = _embed,
) -> List[exp.Expr]:
    if not candidates:
        return []
    error_type = error_report.get("error_type", "")
    faulty = (extract_faulty_token(error_report) or "").lower()

    candidates = type_based_pruning(candidates, error_type, faulty, schema_info)
    candidates = top_k_pruning(
        candidates,
        error_type,
        faulty,
        current_ast,
        K=K,
        embed_fn=embed_fn,
    )
    return candidates


_SQLSTATE_TO_ERROR = {
    "42703": "UNDEFINED_COLUMN",
    "42P01": "UNDEFINED_TABLE",
    "42702": "AMBIGUOUS_COLUMN",
    "42804": "DATATYPE_MISMATCH",
    "42883": "UNDEFINED_OPERATOR",
    "42601": "SYNTAX_ERROR",
}


def _classify_psycopg_error(e: Exception, sql: str) -> Dict[str, Any]:
    sqlstate = getattr(e, "sqlstate", None)
    diag = getattr(e, "diag", None)
    msg = (getattr(diag, "message_primary", None) if diag else None) or str(e)
    detail = getattr(diag, "message_detail", None) if diag else None
    raw_pos = getattr(diag, "statement_position", None) if diag else None
    try:
        pos = int(raw_pos) if raw_pos else None
    except (TypeError, ValueError):
        pos = None
    return {
        "error_type": _SQLSTATE_TO_ERROR.get(sqlstate, "UNKNOWN"),
        "sqlstate": sqlstate,
        "message": msg,
        "detail": detail,
        "position": pos,
        "query": sql,
    }


def _run_query(db, sql: str) -> Tuple[Optional[List[Tuple]], Optional[Dict[str, Any]]]:
    if hasattr(db, "execute_with_diagnostics"):
        return db.execute_with_diagnostics(sql)

    try:
        rows = db.execute(sql)
        if isinstance(rows, list) and len(rows) == 0:
            return rows, {
                "error_type": "EMPTY_RESULT",
                "message": "no rows returned",
                "position": None,
                "query": sql,
            }
        return rows, None
    except Exception as e:
        return None, _classify_psycopg_error(e, sql)


def bfs(
    query: str,
    db,
    mutator: Optional[SafeQLASTMutator] = None,
    schema_info: Optional[Dict[str, Any]] = None,
    alpha: float = 0.35,
    K: int = 3,
    max_steps: int = 100,
) -> Tuple[Optional[List[Tuple]], Optional[str], Optional[str]]:

    
    mutator = mutator or SafeQLASTMutator(dialect="postgres")
    schema_info = {
        "tables":   ["Users", "Posts"],
        "columns":  ["ID", "name", "age", "UserID", "txt", "len", "time"],
        "column_types": {                 # this is Γ
            "id": "int",   "name": "text",  "age": "int",
            "userid": "int", "txt": "text", "len": "int", "time": "date",
        },
        "function_signatures": {           # optional
            "sum": ["numeric"], "count": ["any"], "date_part": ["text", "timestamp"],
        },
    } ### TODO: change from hardcoded values to schema fetched by the Postgres database

    try:
        root_ast = parse_one(query, dialect=mutator.dialect)
    except Exception:
        return None, None, "PARSE_ERROR"
    root_sql = root_ast.sql(dialect=mutator.dialect)

    refineTree: List[Tuple[float, int, str, exp.Expr]] = []
    tiebreak = 0
    heapq.heappush(refineTree, (0.0, tiebreak, root_sql, root_ast))

    visited: set = set()
    steps = 0

    while refineTree:
        _, _, current_sql, current_ast = heapq.heappop(refineTree)
        if current_sql in visited:  
            continue
        visited.add(current_sql)

        steps += 1
        if steps > max_steps:  
            return None, None, "MAX_STEPS"

        rows, err = _run_query(db, current_sql)
        if err is None:  # nearest safe query found
            return rows, current_sql, None

        if err.get("error_type") in ("SYNTAX_ERROR", "UNKNOWN"):
            continue

        try:
            raw_candidates = mutator.generate_candidate_trees(
                current_ast, err, schema_info
            )
        except Exception:
            raw_candidates = []

        pruned = prune_refinements(
            candidates=raw_candidates,
            error_report=err,
            schema_info=schema_info,
            current_ast=current_ast,
            K=K,
        )

        for cand in pruned:
            try:
                cand_sql = cand.sql(dialect=mutator.dialect)
            except Exception:
                continue
            if cand_sql in visited:
                continue
            tiebreak += 1
            dist = semantic_distance(root_ast, cand, alpha=alpha)  # δ
            heapq.heappush(refineTree, (dist, tiebreak, cand_sql, cand))

    return None, None, "EXHAUSTED"
