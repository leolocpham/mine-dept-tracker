"""
engine/reconciler.py
Joins SAP actuals to the Ops Tracker contract lines on PO/PR number,
then builds a mismatch / integrity-warning log for any rows that cannot
be cleanly bridged.
"""

from __future__ import annotations

import pandas as pd


def reconcile(
    sap_df: pd.DataFrame,
    ops_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconcile SAP actuals against Ops Tracker contract lines.

    Join strategy
    -------------
    Primary key : PO Number  (preferred — represents an approved commitment)
    Fallback key: PR Number  (used when no PO has been raised yet)

    A single "join_key" column is derived from both DataFrames before
    aggregating and merging, so the logic stays symmetric.

    Returns
    -------
    reconciled_df : All Ops Tracker rows with SAP actuals merged in.
                    Rows with no matching SAP data get zeros for money fields.
    mismatch_log  : DataFrame of integrity warnings (cost-centre mismatches,
                    SAP rows with no Ops match, blank join keys, etc.).
    """
    mismatch_parts: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # 1. Aggregate SAP by join key (sum money; keep latest date & first text)
    # ------------------------------------------------------------------
    sap = sap_df.copy()
    sap["join_key"] = sap["po_number"].where(sap["po_number"] != "", sap["pr_number"])

    # Flag SAP rows with no key at all
    blank_sap = sap[sap["join_key"].str.strip() == ""]
    if not blank_sap.empty:
        _warn = blank_sap[["cost_center", "vendor", "amount", "description"]].copy()
        _warn.insert(0, "issue", "SAP row: no PO or PR number")
        mismatch_parts.append(_warn.rename(columns={
            "cost_center": "Cost Center", "vendor": "Vendor",
            "amount": "Posted Amount", "description": "Description",
        }))

    sap_keyed = sap[sap["join_key"].str.strip() != ""]

    sap_agg = (
        sap_keyed
        .groupby("join_key", as_index=False)
        .agg(
            sap_posted_amount  = ("amount",      "sum"),
            sap_commitment     = ("commitment",   "sum"),
            sap_vendor         = ("vendor",       "first"),
            sap_description    = ("description",  "first"),
            sap_cost_center    = ("cost_center",  "first"),
            sap_latest_date    = ("date",         "max"),
        )
    )

    # ------------------------------------------------------------------
    # 2. Build join key on Ops side
    # ------------------------------------------------------------------
    ops = ops_df.copy()
    ops["join_key"] = ops["po_number"].where(ops["po_number"] != "", ops["pr_number"])

    # ------------------------------------------------------------------
    # 3. Left-join (Ops is the master — every contract line is preserved)
    # ------------------------------------------------------------------
    merged = ops.merge(sap_agg, on="join_key", how="left")
    merged["sap_posted_amount"] = merged["sap_posted_amount"].fillna(0.0)
    merged["sap_commitment"]    = merged["sap_commitment"].fillna(0.0)

    # Resolve vendor: Ops tracker name wins; fall back to SAP if blank
    merged["vendor"] = merged["vendor"].where(
        merged["vendor"].str.strip() != "",
        merged["sap_vendor"].fillna(""),
    )

    # ------------------------------------------------------------------
    # 4. Integrity checks → mismatch log
    # ------------------------------------------------------------------

    # 4a. Cost-centre mismatch between Ops and SAP for the same PO
    has_sap_cc = merged["sap_cost_center"].notna() & (merged["sap_cost_center"].str.strip() != "")
    cc_mismatch = has_sap_cc & (merged["cost_center"] != merged["sap_cost_center"])
    if cc_mismatch.any():
        rows = merged.loc[cc_mismatch, ["join_key", "cost_center", "sap_cost_center", "vendor", "task"]].copy()
        rows.insert(0, "issue", "Cost centre mismatch (Ops ≠ SAP)")
        rows.columns = ["Issue", "PO/PR", "Ops CC", "SAP CC", "Vendor", "Task"]
        mismatch_parts.append(rows)

    # 4b. SAP rows with no matching Ops contract line
    ops_keys = set(ops["join_key"].dropna()) - {""}
    sap_keys = set(sap_agg["join_key"].dropna()) - {""}
    orphan_keys = sap_keys - ops_keys
    if orphan_keys:
        orphan = sap_agg[sap_agg["join_key"].isin(orphan_keys)][
            ["join_key", "sap_cost_center", "sap_vendor", "sap_posted_amount"]
        ].copy()
        orphan.insert(0, "issue", "In SAP but not in Ops Tracker")
        orphan.columns = ["Issue", "PO/PR", "Cost Center", "Vendor", "Posted Amount"]
        mismatch_parts.append(orphan)

    mismatch_log = (
        pd.concat(mismatch_parts, ignore_index=True)
        if mismatch_parts
        else pd.DataFrame(columns=["Issue", "PO/PR", "Detail"])
    )

    return merged, mismatch_log
