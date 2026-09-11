"""Spider schema loading and DIN-style schema stringifiers."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class SpiderSchemaStore:
    """In-memory Spider tables.json / test_tables.json views."""

    def __init__(self, tables_json: Path) -> None:
        self.schema, self.primary, self.foreign = self._creating_schema(tables_json)

    @staticmethod
    def _creating_schema(
        dataset_json: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        schema_df = pd.read_json(dataset_json)
        schema_df = schema_df.drop(columns=["column_names", "table_names"], errors="ignore")
        schema: list[list[Any]] = []
        f_keys: list[list[Any]] = []
        p_keys: list[list[Any]] = []
        for _, row in schema_df.iterrows():
            tables = row["table_names_original"]
            col_names = row["column_names_original"]
            col_types = row["column_types"]
            foreign_keys = row["foreign_keys"]
            primary_keys = row["primary_keys"]
            for col, col_type in zip(col_names, col_types):
                index, col_name = col
                if index == -1:
                    for table in tables:
                        schema.append([row["db_id"], table, "*", "text"])
                else:
                    schema.append([row["db_id"], tables[index], col_name, col_type])
            for primary_key in primary_keys:
                index, column = col_names[primary_key]
                p_keys.append([row["db_id"], tables[index], column])
            for foreign_key in foreign_keys:
                first, second = foreign_key
                first_index, first_column = col_names[first]
                second_index, second_column = col_names[second]
                f_keys.append(
                    [
                        row["db_id"],
                        tables[first_index],
                        tables[second_index],
                        first_column,
                        second_column,
                    ]
                )
        spider_schema = pd.DataFrame(
            schema, columns=["Database name", " Table Name", " Field Name", " Type"]
        )
        spider_primary = pd.DataFrame(
            p_keys, columns=["Database name", "Table Name", "Primary Key"]
        )
        spider_foreign = pd.DataFrame(
            f_keys,
            columns=[
                "Database name",
                "First Table Name",
                "Second Table Name",
                "First Table Foreign Key",
                "Second Table Foreign Key",
            ],
        )
        return spider_schema, spider_primary, spider_foreign

    def find_foreign_keys_mysql_like(self, db_name: str) -> str:
        df = self.foreign[self.foreign["Database name"] == db_name]
        output = "["
        for _, row in df.iterrows():
            output += (
                f"{row['First Table Name']}.{row['First Table Foreign Key']} = "
                f"{row['Second Table Name']}.{row['Second Table Foreign Key']},"
            )
        if output.endswith(","):
            output = output[:-1]
        return output + "]"

    def find_fields_mysql_like(self, db_name: str) -> str:
        df = self.schema[self.schema["Database name"] == db_name]
        grouped = df.groupby(" Table Name")
        output = ""
        for name, group in grouped:
            output += f"Table {name}, columns = ["
            for _, row in group.iterrows():
                output += f"{row[' Field Name']},"
            output = output[:-1] + "]\n"
        return output

    def find_primary_keys_mysql_like(self, db_name: str) -> str:
        df = self.primary[self.primary["Database name"] == db_name]
        output = "["
        for _, row in df.iterrows():
            output += f"{row['Table Name']}.{row['Primary Key']},"
        if output.endswith(","):
            output = output[:-1]
        return output + "]\n"


def sample_db_values(
    db_path: Path,
    *,
    limit_per_column: int = 5,
    max_tables: int = 12,
    question: str = "",
    max_matched: int = 24,
) -> str:
    """Return capped distinct cell values per column for prompt grounding.

    When ``question`` is set, also surface values that fuzzy-match question
    tokens / quoted phrases first (helps Female/USA-style literal grounding).
    """
    if not db_path.exists():
        return ""

    q = (question or "").strip()
    q_lower = q.lower()
    # Quoted phrases + alphanumeric tokens length >= 3
    phrases = [m.strip() for m in re.findall(r"['\"]([^'\"]+)['\"]", q) if m.strip()]
    tokens = [
        t
        for t in re.findall(r"[A-Za-z0-9_]+", q)
        if len(t) >= 3 and t.lower() not in _STOP
    ]
    cues = list(dict.fromkeys([*phrases, *tokens]))  # preserve order, unique
    # Expand common NL phrases to DB spellings seen in Spider (word-boundary)
    for nl, alts in _VALUE_SYNONYMS.items():
        if re.search(rf"\b{re.escape(nl)}\b", q_lower):
            cues.extend(alts)
    cues = list(dict.fromkeys(cues))

    matched: list[str] = []
    lines: list[str] = ["Sample DB values (capped):"]
    conn = sqlite3.connect(str(db_path))
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ][:max_tables]
        for table in tables:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info('{table}')").fetchall()]
            for col in cols:
                try:
                    # Pull a wider pool when matching against the question
                    fetch_n = max(limit_per_column, 40) if cues else limit_per_column
                    rows = conn.execute(
                        f'SELECT DISTINCT "{col}" FROM "{table}" '
                        f'WHERE "{col}" IS NOT NULL LIMIT {fetch_n}'
                    ).fetchall()
                except sqlite3.Error:
                    continue
                raw_vals = [r[0] for r in rows if r[0] is not None]
                # Prefer question-matched cells (exact DB casing)
                hit_vals: list[str] = []
                other_vals: list[str] = []
                for v in raw_vals:
                    vs = str(v)
                    vs_l = vs.lower()
                    hit = False
                    for cue in cues:
                        cl = cue.lower()
                        if len(cl) <= 3:
                            # Short cues: exact match only (avoid initial/noise hits)
                            if cl == vs_l:
                                hit = True
                                break
                        elif cl == vs_l or cl in vs_l:
                            # Prefer cue⊆value (female→Female). Do NOT use value⊆cue
                            # (male⊆female would false-hit Male).
                            hit = True
                            break
                        elif len(vs_l) >= 5 and abs(len(vs_l) - len(cl)) <= 2 and vs_l in cl:
                            hit = True
                            break
                    if hit:
                        hit_vals.append(vs[:40])
                    else:
                        other_vals.append(vs[:40])
                for hv in hit_vals:
                    if len(matched) >= max_matched:
                        break
                    matched.append(f"  {table}.{col} = '{hv}'")
                # Cap general samples: matched first, then fillers
                shown = (hit_vals + other_vals)[:limit_per_column]
                if shown:
                    lines.append(f"  {table}.{col}: {shown}")
    finally:
        conn.close()

    parts: list[str] = []
    if matched:
        parts.append(
            "Question-matched DB values (use these exact spellings in WHERE):"
        )
        parts.extend(matched[:max_matched])
        parts.append(
            "Instruction: copy string literals from the matched/sample values "
            "exactly (case-sensitive); do not paraphrase (e.g. USA not United States)."
        )
    if len(lines) > 1:
        parts.extend(lines)
    return "\n".join(parts)


_VALUE_SYNONYMS: dict[str, list[str]] = {
    "united states": ["USA", "US", "United States"],
    "u.s.": ["USA", "US"],
    "u.s.a.": ["USA"],
    "america": ["USA", "US"],
    "female": ["Female", "F"],
    "male": ["Male", "M"],
    "yes": ["Yes", "Y", "true", "1"],
    "no": ["No", "N", "false", "0"],
}


_STOP = {
    "the",
    "and",
    "for",
    "are",
    "how",
    "many",
    "what",
    "which",
    "who",
    "whose",
    "with",
    "from",
    "that",
    "have",
    "has",
    "had",
    "was",
    "were",
    "this",
    "those",
    "these",
    "list",
    "show",
    "find",
    "give",
    "name",
    "names",
    "all",
    "each",
    "their",
    "them",
    "than",
    "more",
    "less",
    "most",
    "least",
    "order",
    "ordered",
    "ascending",
    "descending",
    "average",
    "total",
    "number",
    "count",
    "ids",
}


def load_schema_store(data_dir: Path, split: str) -> SpiderSchemaStore:
    data_dir = Path(data_dir)
    if split == "test":
        path = data_dir / "test_tables.json"
        if not path.exists():
            path = data_dir / "tables.json"
    else:
        path = data_dir / "tables.json"
    return SpiderSchemaStore(path)
