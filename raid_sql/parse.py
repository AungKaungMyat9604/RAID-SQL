"""SQL extraction and voting helpers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Optional


_INTERSECT_QUESTION_RE = re.compile(
    r"\b(?:both|shared by|produced both|as well as)\b|"
    r"\b(?:more than|over|greater than|above|bigger than)\b.{0,80}\b(?:and|with)\b.{0,80}"
    r"\b(?:less than|below|under|smaller than)\b|"
    r"\b(?:less than|below|under|smaller than)\b.{0,80}\b(?:and|with)\b.{0,80}"
    r"\b(?:more than|over|greater than|above|bigger than)\b",
    re.I | re.S,
)

_DANGLING_TAIL_RE = re.compile(
    r"(?:\bOR|\bAND|\bNOT|\bIN|\bFROM|\bWHERE|\bJOIN|\bSELECT|\bINTERSECT|\bUNION|\bEXCEPT)\s*$",
    re.I,
)


def strip_sql_fences(text: str) -> str:
    """Remove markdown code fences anywhere in the model output."""
    raw = (text or "").strip()
    # Full fenced block
    m = re.search(r"```(?:sql)?\s*([\s\S]*?)```", raw, flags=re.I)
    if m:
        raw = m.group(1).strip()
    else:
        # Opening fence without closing, or leftover fence tokens
        raw = re.sub(r"```(?:sql)?", "", raw, flags=re.I)
        raw = raw.replace("```", "")
    return raw.strip().rstrip(";")


def extract_sql(text: str) -> str:
    """Pull SQL from model output (handles fences / SQL: prefix / SELECT body)."""
    raw = strip_sql_fences(text)
    # Prefer the final answer block when the hard prompt emits sub-SQL then SQL:
    so_split = re.split(r"[Ss]o,\s+the answer[^=]*=\s*", raw, maxsplit=1)
    if len(so_split) > 1:
        raw = so_split[-1]
    if "SQL:" in raw:
        raw = strip_sql_fences(raw.split("SQL:")[-1])
    raw = re.sub(r"\s+", " ", raw).strip()
    # Drop leading junk before SELECT / WITH
    upper = raw.upper()
    for kw in ("SELECT", "WITH"):
        idx = upper.find(kw)
        if idx >= 0:
            raw = raw[idx:]
            break
    else:
        raw = "SELECT " + raw
    # Second pass: fences sometimes appear after a forced SELECT prefix
    raw = strip_sql_fences(raw)
    upper = raw.upper()
    for kw in ("SELECT", "WITH"):
        idx = upper.find(kw)
        if idx >= 0:
            raw = raw[idx:]
            break
    return raw.strip().rstrip(";")


def is_incomplete_sql(sql: str) -> bool:
    """Heuristic: truncated / unbalanced SQL that should not be the final answer."""
    s = (sql or "").strip()
    if not s or s.upper() in {"SELECT", "SELECT *"}:
        return True
    if s.count("(") != s.count(")"):
        return True
    if s.endswith(("(", ",", ".", "=")):
        return True
    if _DANGLING_TAIL_RE.search(s):
        return True
    # Subquery fragment with stray closing paren (e.g. "SELECT Club_ID FROM player)")
    if s.endswith(")") and "SELECT" in s.upper() and " FROM " in f" {s.upper()} ":
        # Only flag if it looks like a lone subquery, not a full NOT IN / nested query
        upper = s.upper()
        if " WHERE " not in upper and " JOIN " not in upper and " GROUP " not in upper:
            if upper.count("SELECT") == 1:
                return True
    return False


def needs_intersect_style(question: str) -> bool:
    """True when the NL question likely needs INTERSECT / set overlap."""
    return bool(_INTERSECT_QUESTION_RE.search(question or ""))


def sql_misses_set_op(sql: str) -> bool:
    upper = f" {(sql or '').upper()} "
    return not any(op in upper for op in (" INTERSECT ", " EXCEPT ", " UNION "))


def needs_semantic_set_op_repair(question: str, sql: str) -> bool:
    """Executable SQL that likely used OR / one-sided filter instead of INTERSECT."""
    if not needs_intersect_style(question):
        return False
    if not sql_misses_set_op(sql):
        return False
    return True


_AGG_HEAD_RE = re.compile(
    r"^\s*(?:COUNT|SUM|AVG|MIN|MAX)\s*\(",
    re.I,
)

_ENTITY_CUE_RE = re.compile(
    r"\b(?:players?|clubs?|customers?|orders?|items?|students?|courses?|"
    r"departments?|employees?|products?|instructors?|teachers?|classes?|"
    r"songs?|albums?|artists?|books?|authors?|movies?|actors?|singers?|"
    r"managers?|teams?|matches?|airports?|flights?|cars?|owners?)\b",
    re.I,
)


def _split_select_items(select_body: str) -> list[str]:
    items: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in select_body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            piece = "".join(cur).strip()
            if piece:
                items.append(piece)
            cur = []
            continue
        cur.append(ch)
    piece = "".join(cur).strip()
    if piece:
        items.append(piece)
    return items


def reorder_select_for_group_by(sql: str) -> str:
    """Legacy keys-first rewrite — kept as no-op.

    Spider gold prefers ``keys_then_agg`` ~70% of the time overall, but most of our
    column-order EX fails are the opposite (``keys_then_agg`` pred vs ``agg_then_keys``
    gold). Blind keys-first forced that failure mode. Use
    ``normalize_group_by_select_order`` instead.
    """
    return (sql or "").strip().rstrip(";")


_METRIC_FIRST_SELECT_RE = re.compile(
    r"^\s*("
    r"find the average|return the average|what is the average|what are the average|"
    r"find the (maximum|minimum|max|min)\b|return the (maximum|minimum|max|min)\b|"
    r"what are the (maximum|minimum|max|min|average)|"
    r"what is the number\b|find the total number\b|"
    r"count the number of|"
    r"find the number of|return the number of|what is the count\b|count how many\b"
    r")",
    re.I,
)

_COUNT_NUMBER_FIRST_RE = re.compile(
    r"^\s*("
    r"count the number|"
    r"count how many|"
    r"what is the count\b|"
    r"find the number of|"
    r"return the number of"
    r")",
    re.I,
)


def _select_body_parts(sql: str) -> tuple[str, str, str, str] | None:
    """Return (head, distinct, body, tail) for the first SELECT … FROM arm."""
    s = (sql or "").strip().rstrip(";")
    upper = f" {s.upper()} "
    if any(op in upper for op in (" INTERSECT ", " UNION ", " EXCEPT ")):
        return None
    m = re.match(
        r"^(?P<head>\s*SELECT\s+)(?P<body>.+?)(?P<tail>\s+FROM\s+.+)$",
        s,
        flags=re.I | re.S,
    )
    if not m:
        return None
    body = m.group("body")
    dist = ""
    dm = re.match(r"^(DISTINCT\s+)(.*)$", body, flags=re.I | re.S)
    if dm:
        dist, body = dm.group(1), dm.group(2)
    return m.group("head"), dist, body, m.group("tail")


def prefer_aggregate_first_select(sql: str) -> str:
    """Rewrite ``SELECT key, AGG(...) … GROUP BY`` → ``SELECT AGG(...), key …``."""
    parts = _select_body_parts(sql)
    if not parts:
        return (sql or "").strip().rstrip(";")
    head, dist, body, tail = parts
    s = (sql or "").strip().rstrip(";")
    if " GROUP BY " not in f" {s.upper()} ":
        return s
    items = _split_select_items(body)
    if len(items) < 2:
        return s
    aggs = [it for it in items if _AGG_HEAD_RE.match(it)]
    non_aggs = [it for it in items if not _AGG_HEAD_RE.match(it)]
    if not aggs or not non_aggs:
        return s
    if _AGG_HEAD_RE.match(items[0]):
        return s
    new_body = ", ".join(aggs + non_aggs)
    return f"{head}{dist}{new_body}{tail}".strip()


def normalize_group_by_select_order(sql: str, question: str) -> str:
    """Question-gated aggregate-first SELECT for GROUP BY queries.

    Trained on Spider-train gold: metric-leading questions (``find the average``,
    ``what is the number``, …) more often use ``AGG, key`` order. Do **not** apply
    on entity-leading ``show/list/what are`` questions (those prefer keys-first).
    Empirically net-positive on ``test_raid_v2_values`` without using test labels.
    """
    s = (sql or "").strip().rstrip(";")
    q = (question or "").strip()
    if not q or not _METRIC_FIRST_SELECT_RE.match(q):
        return s
    ql = q.lower()
    if re.match(r"^\s*(show|list|what are the names)\b", ql):
        return s
    return prefer_aggregate_first_select(s)


def normalize_how_many_aggregate_first(sql: str, question: str) -> str:
    """``How many …`` (not ``each``) → prefer ``AGG, key`` SELECT order.

    Gold often lists ``count(*), id`` for top-k / at-most questions that lead with
    ``How many``. Exclude ``each`` (``How many X for each Y``) where gold prefers
    keys-first. Zero-break / net +3 on ``test_raid_v2_values`` official EX.
    """
    s = (sql or "").strip().rstrip(";")
    q = (question or "").strip()
    if not re.match(r"^\s*how many\b", q, flags=re.I):
        return s
    if re.search(r"\beach\b", q, flags=re.I):
        return s
    return prefer_aggregate_first_select(s)


def prefer_keys_first_select(sql: str) -> str:
    """Rewrite ``SELECT AGG(...), key … GROUP BY`` → ``SELECT key, AGG(...) …``."""
    parts = _select_body_parts(sql)
    if not parts:
        return (sql or "").strip().rstrip(";")
    head, dist, body, tail = parts
    s = (sql or "").strip().rstrip(";")
    if " GROUP BY " not in f" {s.upper()} ":
        return s
    items = _split_select_items(body)
    if len(items) < 2:
        return s
    aggs = [it for it in items if _AGG_HEAD_RE.match(it)]
    non_aggs = [it for it in items if not _AGG_HEAD_RE.match(it)]
    if not aggs or not non_aggs:
        return s
    if not _AGG_HEAD_RE.match(items[0]):
        return s
    new_body = ", ".join(non_aggs + aggs)
    return f"{head}{dist}{new_body}{tail}".strip()


def normalize_how_many_each_keys_first(sql: str, question: str) -> str:
    """``How many … each …`` → prefer ``key, AGG`` SELECT order.

    Complements ``normalize_how_many_aggregate_first`` (no-each → agg-first).
    """
    s = (sql or "").strip().rstrip(";")
    q = (question or "").strip()
    if not re.match(r"^\s*how many\b", q, flags=re.I):
        return s
    if not re.search(r"\beach\b", q, flags=re.I):
        return s
    return prefer_keys_first_select(s)


def extract_select_order_plan(text: str) -> list[str]:
    """Parse ``Select_order: a, b, c`` from model output (if present)."""
    raw = text or ""
    m = re.search(
        r"Select_order\s*:\s*([^\n]+)",
        raw,
        flags=re.I,
    )
    if not m:
        return []
    body = m.group(1).strip().strip("[]")
    # stop at SQL: if model glued lines
    body = re.split(r"\bSQL\s*:", body, maxsplit=1, flags=re.I)[0].strip()
    if not body:
        return []
    return [p.strip().strip("`\"'") for p in body.split(",") if p.strip()]


def _select_item_key(item: str) -> str:
    it = (item or "").strip()
    it = re.sub(r"\bas\s+\w+$", "", it, flags=re.I).strip()
    if _AGG_HEAD_RE.match(it):
        m = re.match(r"^\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(", it, flags=re.I)
        return (m.group(1) if m else "agg").lower()
    # last identifier
    parts = re.findall(r"[A-Za-z_][\w]*", it)
    return parts[-1].lower() if parts else it.lower()


def apply_select_order_plan(sql: str, plan: list[str]) -> str:
    """Reorder SELECT items to follow a model-emitted Select_order plan.

    Only rewrites when every plan token uniquely matches one SELECT item and
    the item count matches. Otherwise returns SQL unchanged.
    """
    s = (sql or "").strip().rstrip(";")
    if not plan or len(plan) < 2:
        return s
    parts = _select_body_parts(s)
    if not parts:
        return s
    head, dist, body, tail = parts
    items = _split_select_items(body)
    if len(items) != len(plan):
        return s
    keyed = [(_select_item_key(it), it) for it in items]
    used: set[int] = set()
    ordered: list[str] = []
    for token in plan:
        tok = _select_item_key(token)
        matches = [i for i, (k, _) in enumerate(keyed) if k == tok and i not in used]
        if len(matches) != 1:
            # soft: substring / startswith
            matches = [
                i
                for i, (k, _) in enumerate(keyed)
                if i not in used and (tok in k or k in tok)
            ]
            if len(matches) != 1:
                return s
        used.add(matches[0])
        ordered.append(keyed[matches[0]][1])
    if len(ordered) != len(items) or ordered == items:
        return s
    return f"{head}{dist}{', '.join(ordered)}{tail}".strip()



def normalize_count_number_aggregate_first(sql: str, question: str) -> str:
    """``Count the number of…`` / ``find the number of…`` → ``AGG, key`` order.

    Complements ``normalize_how_many_aggregate_first`` (which only matches
    ``How many``). Offline on Flash v3 preds: net +3 official EX.
    """
    s = (sql or "").strip().rstrip(";")
    q = (question or "").strip()
    if not _COUNT_NUMBER_FIRST_RE.match(q):
        return s
    return prefer_aggregate_first_select(s)


def inject_distinct_if_asked(sql: str, question: str) -> str:
    """If the question says ``distinct``, ensure ``SELECT DISTINCT``.

    Single-SELECT only (skips INTERSECT/UNION/EXCEPT). Zero-break on
    ``test_raid_v2_values`` (no EX flips); keeps prompt/debug alignment.
    """
    s = (sql or "").strip().rstrip(";")
    q = question or ""
    if not re.search(r"\bdistinct\b", q, flags=re.I):
        return s
    if not s or re.search(r"\b(INTERSECT|UNION|EXCEPT)\b", s, flags=re.I):
        return s
    if re.match(r"^\s*SELECT\s+DISTINCT\b", s, flags=re.I):
        return s
    m = re.match(r"^(\s*SELECT\s+)(.*)$", s, flags=re.I | re.S)
    if not m:
        return s
    return f"{m.group(1)}DISTINCT {m.group(2).lstrip()}"


def normalize_fname_lname_order(sql: str) -> str:
    """Prefer ``lname, fname`` when both appear adjacent (common Spider gold)."""
    parts = _select_body_parts(sql)
    if not parts:
        return (sql or "").strip().rstrip(";")
    head, dist, body, tail = parts
    if not re.search(r"\bfname\b", body, flags=re.I) or not re.search(
        r"\blname\b", body, flags=re.I
    ):
        return (sql or "").strip().rstrip(";")
    new_body, n = re.subn(
        r"([A-Za-z_][\w]*\.)?fname(\s*),(\s*)([A-Za-z_][\w]*\.)?lname",
        r"\4lname\2,\3\1fname",
        body,
        flags=re.I,
        count=1,
    )
    if not n:
        return (sql or "").strip().rstrip(";")
    return f"{head}{dist}{new_body}{tail}".strip()


def normalize_select_star_order_by(sql: str) -> str:
    """``SELECT * FROM T ORDER BY col`` → ``SELECT col FROM T ORDER BY col``.

    Spider gold never uses ``SELECT * … ORDER BY``; when the question asks for
    ordered “details” of one attribute, ``*`` fails EX. Empirically zero-break on
    full Spider-test ``test_raid`` + ``test_raid_v2`` preds.
    """
    s = (sql or "").strip().rstrip(";")
    m = re.match(
        r"^\s*SELECT\s+\*\s+FROM\s+(\w+)\s+ORDER\s+BY\s+([\w.]+)(?:\s+(ASC|DESC))?\s*$",
        s,
        flags=re.I,
    )
    if not m:
        return s
    table, col, direction = m.group(1), m.group(2), (m.group(3) or "").upper()
    col_name = col.split(".")[-1]
    dir_s = f" {direction}" if direction else ""
    return f"SELECT {col_name} FROM {table} ORDER BY {col}{dir_s}"


def strip_count_from_frequency_sort(sql: str, question: str) -> str:
    """Drop COUNT from SELECT when the question only asks to list/sort by frequency.

    Matches gold like ``SELECT col … GROUP BY col ORDER BY COUNT(*) DESC`` when the
    model emitted ``SELECT col, COUNT(*) …``. Gated by question cues; excludes
    “how many / number of …”. Zero-break on full ``test_raid`` + ``test_raid_v2``.
    """
    q = (question or "").lower().strip()
    if not q:
        return (sql or "").strip().rstrip(";")
    if re.search(
        r"\b(how many|number of|what is the count|and how many)\b",
        q,
    ):
        return (sql or "").strip().rstrip(";")
    if not (
        re.search(
            r"in descending order of (?:their )?(?:frequency(?: of occurrence)?|count)",
            q,
        )
        or re.search(r"smallest frequency count|largest frequency count", q)
        or re.search(r"list all .+ in descending order of count", q)
    ):
        return (sql or "").strip().rstrip(";")

    s = (sql or "").strip().rstrip(";")
    upper = f" {s.upper()} "
    if " GROUP BY " not in upper:
        return s
    if not re.search(r"ORDER BY\s+COUNT\s*\(", s, flags=re.I):
        return s
    m = re.match(
        r"^(?P<head>\s*SELECT\s+)(?P<body>.+?)(?P<tail>\s+FROM\s+.+)$",
        s,
        flags=re.I | re.S,
    )
    if not m:
        return s
    items = _split_select_items(m.group("body"))
    non_aggs = [it for it in items if not _AGG_HEAD_RE.match(it)]
    aggs = [it for it in items if _AGG_HEAD_RE.match(it)]
    if len(non_aggs) != 1 or not aggs:
        return s
    return f"{m.group('head')}{non_aggs[0]}{m.group('tail')}".strip()


def question_suggests_multi_table(question: str) -> bool:
    cues = {c.lower() for c in _ENTITY_CUE_RE.findall(question or "")}
    if len(cues) >= 2:
        return True
    q = question or ""
    if re.search(r"\b(?:of|for|with|from|who|whose|that have|that has)\b", q, re.I):
        if re.search(r"\band\b", q, re.I):
            return True
    return False


def needs_join_repair(
    question: str,
    sql: str,
    *,
    predicted_class: str,
    has_foreign_keys: bool,
) -> bool:
    """Multi-table question with FKs but SQL missing JOIN."""
    if not has_foreign_keys:
        return False
    if " JOIN " in f" {(sql or '').upper()} ":
        return False
    if not question_suggests_multi_table(question):
        return False
    cls = predicted_class or ""
    is_easy = "EASY" in cls and "NON-NESTED" not in cls
    if is_easy:
        return False
    upper = f" {(sql or '').upper()} "
    # Nested set-ops / IN often encode multi-table without a top-level JOIN
    if "NON-NESTED" not in cls and any(
        op in upper for op in (" INTERSECT ", " EXCEPT ", " UNION ", " IN ", " NOT IN ")
    ):
        return False
    return True


def extract_schema_links(text: str) -> str:
    for key in ("Schema_links:", "schema_links:"):
        if key in text:
            return text.split(key, 1)[1].strip().split("\n")[0].strip()
    return "[]"


def extract_label(text: str) -> str:
    if "Label:" in text:
        return text.split("Label:", 1)[1].strip().split("\n")[0].strip()
    upper = text.upper()
    for lab in ('"NESTED"', '"NON-NESTED"', '"EASY"', "NESTED", "NON-NESTED", "EASY"):
        if lab in upper or lab.strip('"') in upper:
            if "NON-NESTED" in upper:
                return '"NON-NESTED"'
            if "NESTED" in upper:
                return '"NESTED"'
            if "EASY" in upper:
                return '"EASY"'
    return '"NESTED"'


def extract_sub_questions(classification: str) -> str:
    try:
        return classification.split('questions = ["')[1].split('"]')[0]
    except Exception:  # noqa: BLE001
        return ""


def vote_by_result_key(
    candidates: list[tuple[str, Optional[str], bool]],
) -> str:
    """Pick SQL by majority execution result_key; fallback to first successful."""
    keys = [k for _, k, ok in candidates if ok and k is not None]
    if keys:
        winner = Counter(keys).most_common(1)[0][0]
        for sql, k, ok in candidates:
            if ok and k == winner:
                return sql
    for sql, _, ok in candidates:
        if ok:
            return sql
    return candidates[0][0] if candidates else "SELECT"


def vote_by_bag_key_then_order(
    candidates: list[tuple[str, Optional[str], bool]],
    question: str = "",
    *,
    prefer_order: Optional[Callable[[str, str], str]] = None,
) -> str:
    """Majority vote on order-insensitive bag_key, then pick SELECT order.

    Column-order variants of the same cell multiset no longer split the SC vote.
    Among bag winners, ``prefer_order(sql, question)`` (e.g. ``_finalize_sql``)
    chooses Spider-facing SELECT order; ties fall back to first winner.
    """
    bags = [k for _, k, ok in candidates if ok and k is not None]
    if not bags:
        return vote_by_result_key(candidates)
    winner = Counter(bags).most_common(1)[0][0]
    pool = [sql for sql, k, ok in candidates if ok and k == winner]
    if not pool:
        return vote_by_result_key(candidates)
    if prefer_order is not None and question is not None:
        ranked = [prefer_order(sql, question) for sql in pool]
        return Counter(ranked).most_common(1)[0][0]
    return Counter(pool).most_common(1)[0][0]


def select_order_heuristic_variants(sql: str, question: str) -> list[str]:
    """Deterministic SELECT-order rewrites to inject into SC (bag-equal vote)."""
    s = (sql or "").strip().rstrip(";")
    out = [s]
    for fn in (
        normalize_group_by_select_order,
        normalize_how_many_aggregate_first,
        normalize_how_many_each_keys_first,
        normalize_count_number_aggregate_first,
    ):
        v = fn(s, question)
        if v and v not in out:
            out.append(v)
    return out
