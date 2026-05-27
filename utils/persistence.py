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

_CONTRACT_FILE = DATA_DIR / "contracts.json"
_TMM_FILE      = DATA_DIR / "tmm.json"

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
    "period":     str,
    "area":       str,
    "tons":       float,
    "total_cost": float,
    "notes":      str,
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
