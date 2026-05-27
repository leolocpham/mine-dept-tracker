"""
utils/persistence.py
Saves and loads the Contract Tracker and TMM tables as JSON files in a
local data/ folder so data survives app restarts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

_CONTRACT_FILE  = DATA_DIR / "contracts.json"
_TMM_FILE       = DATA_DIR / "tmm.json"
_SAP_DB_FILE    = DATA_DIR / "sap_actuals.json"

# ---------------------------------------------------------------------------
# Column schemas (defines order + dtypes for both tables)
# ---------------------------------------------------------------------------

CONTRACT_COLS: dict[str, type] = {
    "cost_center":     str,
    "sub_dept":        str,
    "vendor":          str,
    "task":            str,
    "pr_number":       str,
    "po_number":       str,
    "original_budget": float,
    "amount_spent":    float,
    "sap_synced":      bool,
    "notes":           str,
}

TMM_COLS: dict[str, type] = {
    "year":  float,   # stored as float for JSON round-trip; displayed as int
    "month": str,
    "tons":  float,
}


def _empty(schema: dict) -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype=typ) for col, typ in schema.items()})


def _coerce(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Add missing columns and coerce types to match the schema."""
    for col, typ in schema.items():
        if col not in df.columns:
            df[col] = pd.Series(dtype=typ)
        else:
            try:
                if typ == float:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                elif typ == bool:
                    df[col] = df[col].fillna(False).astype(bool)
                else:
                    df[col] = df[col].fillna("").astype(str)
            except Exception:
                pass
    return df[list(schema.keys())]


# ---------------------------------------------------------------------------
# Contract Tracker
# ---------------------------------------------------------------------------

def load_contracts() -> pd.DataFrame:
    if _CONTRACT_FILE.exists():
        try:
            df = pd.read_json(_CONTRACT_FILE, orient="records", dtype=False)
            return _coerce(df, CONTRACT_COLS)
        except Exception:
            pass
    return _empty(CONTRACT_COLS)


def save_contracts(df: pd.DataFrame) -> None:
    # Only persist the base (non-calculated) columns
    cols = [c for c in CONTRACT_COLS if c in df.columns]
    df[cols].to_json(_CONTRACT_FILE, orient="records", indent=2)


# ---------------------------------------------------------------------------
# TMM Tracker
# ---------------------------------------------------------------------------

def load_tmm() -> pd.DataFrame:
    if _TMM_FILE.exists():
        try:
            df = pd.read_json(_TMM_FILE, orient="records", dtype=False)
            return _coerce(df, TMM_COLS)
        except Exception:
            pass
    return _empty(TMM_COLS)


def save_tmm(df: pd.DataFrame) -> None:
    cols = [c for c in TMM_COLS if c in df.columns]
    df[cols].to_json(_TMM_FILE, orient="records", indent=2)


# ---------------------------------------------------------------------------
# SAP Actuals Database
# ---------------------------------------------------------------------------

def load_sap_db() -> pd.DataFrame:
    """
    Load the persistent SAP actuals database.
    Returns an empty DataFrame (not None) if no data exists yet.
    Date column is restored to datetime.
    """
    if _SAP_DB_FILE.exists():
        try:
            df = pd.read_json(_SAP_DB_FILE, orient="records", dtype=False)
            if df.empty:
                return pd.DataFrame()
            df["date"]   = pd.to_datetime(df.get("date", pd.Series(dtype=str)), errors="coerce")
            df["amount"] = pd.to_numeric(df.get("amount", 0), errors="coerce").fillna(0.0)
            return df
        except Exception:
            pass
    return pd.DataFrame()


def save_sap_db(df: pd.DataFrame) -> None:
    """Persist the SAP actuals database. Serialises datetime → ISO string."""
    out = df.copy()
    if "date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["date"]):
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_json(_SAP_DB_FILE, orient="records", indent=2, default_handler=str)


def upsert_sap_period(
    existing_db: pd.DataFrame,
    new_records: pd.DataFrame,
    year: int,
    month: str,
) -> pd.DataFrame:
    """
    Replace all records in existing_db for (year, month) with new_records.
    If existing_db is empty, returns new_records as-is.
    """
    if existing_db.empty:
        return new_records.reset_index(drop=True)

    # Derive year/month from date for filtering existing rows
    ex = existing_db.copy()
    ex["_yr"] = ex["date"].dt.year
    ex["_mo"] = ex["date"].dt.strftime("%B")
    keep = ex[~((ex["_yr"] == year) & (ex["_mo"] == month))].drop(columns=["_yr", "_mo"])

    return pd.concat([keep, new_records], ignore_index=True)


def sap_period_summary(db: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary of the SAP database by year + month:
    columns = [Year, Month, Lines, Total Amount ($)]
    Sorted chronologically.
    """
    if db.empty or "date" not in db.columns:
        return pd.DataFrame(columns=["Year", "Month", "Lines", "Total Amount ($)"])

    df = db.copy()
    df["_yr"] = df["date"].dt.year
    df["_mo"] = df["date"].dt.strftime("%B")
    df["_mo_n"] = df["date"].dt.month

    summary = (
        df.groupby(["_yr", "_mo", "_mo_n"], as_index=False)
        .agg(Lines=("amount", "count"), Total=("amount", "sum"))
        .sort_values(["_yr", "_mo_n"])
        .rename(columns={"_yr": "Year", "_mo": "Month", "Total": "Total Amount ($)"})
        .drop(columns=["_mo_n"])
    )
    summary["Year"] = summary["Year"].astype(int)
    return summary.reset_index(drop=True)
