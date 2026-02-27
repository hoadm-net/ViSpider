#!/usr/bin/env python3
"""
Schema Utilities
Build CREATE TABLE schema strings from Spider tables.json for text-to-SQL prompts.
"""

import json
from pathlib import Path
from functools import lru_cache
from typing import Optional

_TABLES_CACHE: dict[str, dict] = {}


def load_tables(tables_file: str | Path) -> dict[str, dict]:
    """Load tables.json and return a dict keyed by db_id."""
    global _TABLES_CACHE
    key = str(tables_file)
    if key not in _TABLES_CACHE:
        with open(tables_file, encoding="utf-8") as f:
            entries = json.load(f)
        _TABLES_CACHE[key] = {e["db_id"]: e for e in entries}
    return _TABLES_CACHE[key]


def build_schema(db_id: str, tables_lookup: dict[str, dict]) -> str:
    """
    Return a CREATE TABLE schema string for the given db_id.

    Format:
        CREATE TABLE table1 (
            col1 TYPE PRIMARY KEY,
            col2 TYPE,
            FOREIGN KEY (col3) REFERENCES table2(col4)
        );

    Returns empty string if db_id not found.
    """
    if db_id not in tables_lookup:
        return f"-- schema not found for {db_id}"

    db = tables_lookup[db_id]
    table_names   = db["table_names_original"]
    col_names     = db["column_names_original"]   # [[table_idx, col_name], ...]
    col_types     = db["column_types"]            # ["text", "number", ...]
    primary_keys  = set(db.get("primary_keys", []))
    foreign_keys  = db.get("foreign_keys", [])    # [[col_idx, col_idx], ...]

    # Map: table_idx → list of (col_idx, col_name, col_type)
    table_cols: dict[int, list] = {i: [] for i in range(len(table_names))}
    for col_idx, (tbl_idx, col_name) in enumerate(col_names):
        if tbl_idx == -1:   # skip the wildcard column
            continue
        col_type = _spider_type_to_sql(col_types[col_idx])
        is_pk    = col_idx in primary_keys
        table_cols[tbl_idx].append((col_idx, col_name, col_type, is_pk))

    # Build FK lookup: col_idx → (ref_table_name, ref_col_name)
    fk_map: dict[int, tuple[str, str]] = {}
    for from_idx, to_idx in foreign_keys:
        to_tbl   = col_names[to_idx][0]
        to_col   = col_names[to_idx][1]
        from_col = col_names[from_idx][1]
        fk_map[from_idx] = (table_names[to_tbl], to_col)

    parts = []
    for tbl_idx, tbl_name in enumerate(table_names):
        cols = table_cols.get(tbl_idx, [])
        col_lines = []
        fk_lines  = []
        for col_idx, col_name, col_type, is_pk in cols:
            pk_str = " PRIMARY KEY" if is_pk else ""
            col_lines.append(f"    {col_name} {col_type}{pk_str}")
            if col_idx in fk_map:
                ref_tbl, ref_col = fk_map[col_idx]
                fk_lines.append(f"    FOREIGN KEY ({col_name}) REFERENCES {ref_tbl}({ref_col})")

        all_lines = col_lines + fk_lines
        body = ",\n".join(all_lines)
        parts.append(f"CREATE TABLE {tbl_name} (\n{body}\n);")

    return "\n\n".join(parts)


def build_schema_compact(db_id: str, tables_lookup: dict[str, dict]) -> str:
    """
    Compact one-liner format: table(col1, col2, ...)
    Shorter, useful for smaller context windows.
    """
    if db_id not in tables_lookup:
        return f"-- schema not found for {db_id}"

    db = tables_lookup[db_id]
    table_names = db["table_names_original"]
    col_names   = db["column_names_original"]

    table_cols: dict[int, list] = {i: [] for i in range(len(table_names))}
    for tbl_idx, col_name in col_names:
        if tbl_idx == -1:
            continue
        table_cols[tbl_idx].append(col_name)

    lines = []
    for tbl_idx, tbl_name in enumerate(table_names):
        cols = ", ".join(table_cols.get(tbl_idx, []))
        lines.append(f"{tbl_name}({cols})")
    return "\n".join(lines)


def _spider_type_to_sql(spider_type: str) -> str:
    mapping = {
        "text":    "TEXT",
        "number":  "REAL",
        "time":    "TEXT",
        "boolean": "INTEGER",
        "others":  "TEXT",
    }
    return mapping.get(spider_type.lower(), "TEXT")
