"""
engine/cleaner.py
Ingests raw SAP and Ops Tracker files, normalises column names, cleans
financial values, and returns standardised DataFrames.

Handles the wide variety of column-naming conventions produced by different
SAP transactions (KSB1, ME2N, ZFICO, S_ALR_87013611, etc.) and typical
hand-built contract trackers.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column alias maps
# Keys are the internal standard field names; values are lists of plausible
# column header variations found in the wild (all compared case-insensitively).
# ---------------------------------------------------------------------------

SAP_ALIASES: dict[str, list[str]] = {
    "cost_center":    ["cost center", "cost ctr", "costcenter", "cost centre",
                       "kostenstelle", "cost_center", "ctr"],
    "po_number":      ["purchase order", "po number", "po_number", "po",
                       "purchasing doc.", "purch. doc.", "purchase doc",
                       "order number", "po no", "po #"],
    "pr_number":      ["purchase req.", "pr number", "pr_number", "pr",
                       "purchase requisition", "purch. req.", "requisition",
                       "req. no", "req no", "pr no"],
    "vendor":         ["vendor", "supplier", "creditor", "vendor name",
                       "vendor/supplier", "name 1"],
    "amount":         ["amount in lc", "amount in doc. curr.", "actual costs",
                       "value in lc", "amount", "debit", "posted amount",
                       "actuals", "total amount", "cost", "val. in loc.cur."],
    "commitment":     ["commitment", "open commitments", "commitment amt",
                       "open commitment", "encumbrance", "assigned"],
    "date":           ["posting date", "document date", "value date",
                       "entry date", "doc. date"],
    "description":    ["text", "name", "description", "short text",
                       "item text", "posting text", "item description"],
    "gl_account":     ["g/l account", "gl account", "account", "cost element",
                       "g/l acct", "gl acct"],
    "fiscal_period":  ["period", "fiscal period", "posting period",
                       "fiscal yr/period", "month"],
}

OPS_ALIASES: dict[str, list[str]] = {
    "cost_center":    ["cost center", "cost ctr", "costcenter", "cost centre",
                       "cost_center", "cc"],
    "po_number":      ["po number", "po_number", "purchase order", "po",
                       "contract number", "order number", "po no"],
    "pr_number":      ["pr number", "pr_number", "purchase req.", "requisition",
                       "pr", "req no", "purchase requisition"],
    "vendor":         ["vendor", "supplier", "contractor", "vendor name",
                       "company"],
    "task":           ["project task", "task", "description", "scope",
                       "work description", "activity", "job description",
                       "contract scope", "work scope"],
    "sub_dept":       ["sub-department", "sub-dept", "department", "area",
                       "group", "sub dept", "mine area", "business unit",
                       "dept"],
    "baseline_value": ["baseline value", "original value", "contract value",
                       "baseline", "approved value", "original budget",
                       "contract amount", "original contract"],
    "change_orders":  ["change orders", "fco", "approved changes", "fco value",
                       "variation", "co value", "field change orders",
                       "variations", "amendments", "approved co"],
    "phased_budget":  ["phased budget", "monthly budget", "period budget",
                       "budget target", "budget", "approved budget",
                       "allocated budget", "budget allocation"],
    "pr_date":        ["pr date", "req. date", "requisition date",
                       "request date", "pr raised date", "pr created",
                       "date raised"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    """Return the first df column that matches any alias (case-insensitive)."""
    lookup = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def _normalize_cost_center(val) -> str:
    """
    Return a clean, zero-padded 10-digit string for numeric cost centres,
    or strip-only for alphanumeric ones.
    """
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if re.match(r"^\d+$", s):
        return s.zfill(10)
    return s


def _to_numeric(series: pd.Series) -> pd.Series:
    """
    Coerce a mixed-type column to float, mapping blanks / dashes / nulls → 0.
    Handles comma-formatted numbers ('1,234,567.89').
    """
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace(
            {"": "0", "nan": "0", "None": "0", "N/A": "0",
             "#VALUE!": "0", "-": "0", "–": "0", "—": "0"}
        )
        .pipe(lambda s: s.str.replace(r"^\s*[-–—]\s*$", "0", regex=True))
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def _empty(length: int, default="") -> pd.Series:
    return pd.Series([default] * length)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def read_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Read a .csv, .xlsx, or .xls file from raw bytes into a DataFrame.
    Tries multiple header rows (0 and 1) to skip common SAP title rows.
    """
    import io
    buf = io.BytesIO(file_bytes)
    name_lower = filename.lower()

    if name_lower.endswith(".csv"):
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                buf.seek(0)
                return pd.read_csv(buf, encoding=enc, dtype=str)
            except Exception:
                continue
        raise ValueError(f"Cannot decode CSV file '{filename}'.")

    # Excel
    engine = "xlrd" if name_lower.endswith(".xls") else "openpyxl"
    for header_row in (0, 1, 2):
        try:
            buf.seek(0)
            df = pd.read_excel(buf, engine=engine, header=header_row, dtype=str)
            # Accept if we found at least 2 non-empty columns
            valid_cols = [c for c in df.columns if str(c).strip() not in ("", "nan", "Unnamed")]
            if len(valid_cols) >= 2:
                return df
        except Exception:
            continue
    raise ValueError(f"Cannot parse Excel file '{filename}'.")


def clean_sap(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Normalise a raw SAP actuals export into the standard internal schema.
    Returns (cleaned_df, warning_messages).
    """
    warnings: list[str] = []
    n = len(df)

    def get(field: str, default="") -> pd.Series:
        col = _find_col(df, SAP_ALIASES[field])
        if col is None:
            warnings.append(f"SAP: '{field}' column not found — defaulting to empty/zero.")
            return _empty(n, default)
        return df[col]

    out = pd.DataFrame({
        "cost_center":   get("cost_center").astype(str).str.strip().apply(_normalize_cost_center),
        "po_number":     get("po_number").astype(str).str.strip().str.upper().replace({"NAN": "", "NONE": ""}),
        "pr_number":     get("pr_number").astype(str).str.strip().str.upper().replace({"NAN": "", "NONE": ""}),
        "vendor":        get("vendor").astype(str).str.strip(),
        "amount":        _to_numeric(get("amount", 0)),
        "commitment":    _to_numeric(get("commitment", 0)),
        "date":          pd.to_datetime(get("date"), errors="coerce"),
        "description":   get("description").astype(str).str.strip(),
        "gl_account":    get("gl_account").astype(str).str.strip(),
        "fiscal_period": get("fiscal_period").astype(str).str.strip(),
    })

    return out, warnings


def clean_ops(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Normalise a raw Ops Tracker spreadsheet into the standard internal schema.
    Returns (cleaned_df, warning_messages).
    """
    warnings: list[str] = []
    n = len(df)

    def get(field: str, default="") -> pd.Series:
        col = _find_col(df, OPS_ALIASES[field])
        if col is None:
            warnings.append(f"Ops Tracker: '{field}' column not found — defaulting to empty/zero.")
            return _empty(n, default)
        return df[col]

    out = pd.DataFrame({
        "cost_center":    get("cost_center").astype(str).str.strip().apply(_normalize_cost_center),
        "po_number":      get("po_number").astype(str).str.strip().str.upper().replace({"NAN": "", "NONE": ""}),
        "pr_number":      get("pr_number").astype(str).str.strip().str.upper().replace({"NAN": "", "NONE": ""}),
        "vendor":         get("vendor").astype(str).str.strip(),
        "task":           get("task").astype(str).str.strip(),
        "sub_dept":       get("sub_dept").astype(str).str.strip(),
        "baseline_value": _to_numeric(get("baseline_value", 0)),
        "change_orders":  _to_numeric(get("change_orders", 0)),
        "phased_budget":  _to_numeric(get("phased_budget", 0)),
        "pr_date":        pd.to_datetime(get("pr_date"), errors="coerce"),
    })

    return out, warnings
