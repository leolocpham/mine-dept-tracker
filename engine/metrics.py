"""
engine/metrics.py
Computes all derived financial KPIs from the reconciled dataset.

Metric definitions
------------------
true_contract_value      = baseline_value + change_orders
outstanding_obligation   = max(true_contract_value - sap_posted_amount, 0)
total_committed          = sap_posted_amount + outstanding_obligation
remaining_budget         = phased_budget - total_committed
utilisation_pct          = sap_posted_amount / true_contract_value * 100
remaining_contract       = max(true_contract_value - sap_posted_amount, 0)

Burn flag thresholds (per cost centre aggregate):
  Green  : burn_rate_pct < 75
  Yellow : 75 <= burn_rate_pct < 95
  Red    : burn_rate_pct >= 95  OR  remaining_budget < 0
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived KPI columns to the reconciled DataFrame."""
    out = df.copy()

    out["true_contract_value"]    = out["baseline_value"] + out["change_orders"]
    out["outstanding_obligation"] = (out["true_contract_value"] - out["sap_posted_amount"]).clip(lower=0)
    out["total_committed"]        = out["sap_posted_amount"] + out["outstanding_obligation"]
    out["remaining_budget"]       = out["phased_budget"] - out["total_committed"]
    out["remaining_contract"]     = (out["true_contract_value"] - out["sap_posted_amount"]).clip(lower=0)

    out["utilisation_pct"] = np.where(
        out["true_contract_value"] > 0,
        (out["sap_posted_amount"] / out["true_contract_value"] * 100).round(1),
        0.0,
    )

    return out


def compute_summary(df: pd.DataFrame) -> dict[str, float]:
    """Return a dict of top-level financial KPIs for the metric cards."""
    return {
        "total_phased_budget":       df["phased_budget"].sum(),
        "total_posted_actuals":      df["sap_posted_amount"].sum(),
        "total_open_commitments":    df["outstanding_obligation"].sum(),
        "total_committed":           df["total_committed"].sum(),
        "remaining_free_cash":       df["remaining_budget"].sum(),
        "total_true_contract_value": df["true_contract_value"].sum(),
        "total_change_orders":       df["change_orders"].sum(),
        "contract_count":            len(df),
        "vendor_count":              df["vendor"].nunique(),
    }


def compute_subdept_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Budget vs actuals vs commitments rolled up by sub-department."""
    col = "sub_dept"
    if col not in df.columns or df[col].str.strip().eq("").all():
        return pd.DataFrame()

    grp = (
        df[df[col].str.strip() != ""]
        .groupby(col, as_index=False)
        .agg(
            phased_budget      = ("phased_budget",       "sum"),
            posted_actuals     = ("sap_posted_amount",   "sum"),
            open_commitments   = ("outstanding_obligation", "sum"),
            true_contract_value= ("true_contract_value", "sum"),
        )
        .sort_values("phased_budget", ascending=False)
    )
    return grp


def get_stuck_prs(df: pd.DataFrame, days: int = 14) -> pd.DataFrame:
    """
    Return PRs that have been open > `days` days without converting to a PO.
    Conditions: pr_number is set, po_number is blank, pr_date is set and old.
    """
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=days))

    has_pr  = df["pr_number"].str.strip() != ""
    no_po   = df["po_number"].str.strip().isin(["", "NAN", "NONE"])
    has_date = df["pr_date"].notna()
    is_old  = df["pr_date"] < cutoff

    stuck = df[has_pr & no_po & has_date & is_old].copy()
    if stuck.empty:
        return pd.DataFrame()

    stuck["days_open"] = (datetime.now() - stuck["pr_date"].dt.to_pydatetime()).apply(
        lambda x: x.days if hasattr(x, "days") else 0
    )
    # Simpler vectorised approach
    stuck["days_open"] = (pd.Timestamp.now() - stuck["pr_date"]).dt.days

    cols = ["pr_number", "cost_center", "sub_dept", "vendor", "task",
            "baseline_value", "pr_date", "days_open"]
    return (
        stuck[[c for c in cols if c in stuck.columns]]
        .sort_values("days_open", ascending=False)
        .reset_index(drop=True)
    )


def compute_burn_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-cost-centre burn rate and Red / Yellow / Green budget flag.
    Returns a DataFrame sorted by burn_rate_pct descending.
    """
    grp = (
        df[df["cost_center"].str.strip() != ""]
        .groupby("cost_center", as_index=False)
        .agg(
            phased_budget   = ("phased_budget",   "sum"),
            total_committed = ("total_committed",  "sum"),
            remaining_budget= ("remaining_budget", "sum"),
            posted_actuals  = ("sap_posted_amount","sum"),
        )
    )

    grp["burn_rate_pct"] = np.where(
        grp["phased_budget"] > 0,
        (grp["total_committed"] / grp["phased_budget"] * 100).round(1),
        0.0,
    )

    def _flag(row: pd.Series) -> str:
        if row["remaining_budget"] < 0 or row["burn_rate_pct"] >= 95:
            return "🔴 Red"
        if row["burn_rate_pct"] >= 75:
            return "🟡 Yellow"
        return "🟢 Green"

    grp["status"] = grp.apply(_flag, axis=1)
    return grp.sort_values("burn_rate_pct", ascending=False).reset_index(drop=True)
