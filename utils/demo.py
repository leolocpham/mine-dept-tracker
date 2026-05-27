"""
utils/demo.py
Generates realistic synthetic SAP and Ops Tracker DataFrames for
demonstration / testing without needing real files.
"""

from __future__ import annotations

import io
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

SUB_DEPTS = ["Mine Operations", "Mine Maintenance", "Engineering", "Geology"]

VENDORS = [
    "Caterpillar Financial Products",
    "Komatsu Mining Corp",
    "Hitachi Construction Machinery",
    "Epiroc Australia",
    "Sandvik Mining",
    "WesTrac Equipment",
    "Hastings Deering",
    "Thiess Services",
    "Downer EDI Mining",
    "MACA Limited",
    "Fugro Geosciences",
    "Core Geotechnics",
]

TASKS = [
    "Mobile Fuel Island Refurbishment",
    "Shovel Overhaul — P&H 4100",
    "Blast Pattern Optimisation Study",
    "Pit Dewatering System Upgrade",
    "Haul Road Maintenance Contract",
    "Dragline Electrical Overhaul",
    "Grade Control Drilling Program",
    "Hydraulic Excavator Major Service",
    "Tyres & GET Supply — Cat 793F",
    "SAP Plant Maintenance Integration",
    "Geotechnical Investigation — Phase 2",
    "Conveyor Belt Replacement",
    "Stockpile Management System",
    "Mine Planning Software Licences",
    "Emergency Repair — Dozer D11",
    "Environmental Monitoring Program",
]

COST_CENTRES = [
    "0001101001",  # Mine Operations
    "0001101002",  # Mine Operations
    "0001201001",  # Mine Maintenance
    "0001201002",  # Mine Maintenance
    "0001301001",  # Engineering
    "0001401001",  # Geology
    "0001401002",  # Geology
]

CC_TO_DEPT = {
    "0001101001": "Mine Operations",
    "0001101002": "Mine Operations",
    "0001201001": "Mine Maintenance",
    "0001201002": "Mine Maintenance",
    "0001301001": "Engineering",
    "0001401001": "Geology",
    "0001401002": "Geology",
}


def _rand_date(start_days_ago: int, end_days_ago: int = 0) -> datetime:
    delta = random.randint(end_days_ago, start_days_ago)
    return datetime.now() - timedelta(days=delta)


def generate_ops_tracker() -> pd.DataFrame:
    """Return a realistic Ops Tracker DataFrame (mimics a manual Excel sheet)."""
    rows = []
    po_counter = 4500100
    pr_counter = 9000200

    for i, task in enumerate(TASKS):
        cc = random.choice(COST_CENTRES)
        vendor = random.choice(VENDORS)
        baseline = round(random.uniform(80_000, 2_500_000), -3)
        fco = round(random.uniform(0, baseline * 0.15), -3) if random.random() > 0.5 else 0
        budget = round(baseline * random.uniform(1.05, 1.25), -3)

        # Most tasks have a PO; a few are still at PR stage (stuck PRs)
        if i < 3:
            # Stuck PR: raised > 14 days ago, no PO yet
            po_num = ""
            pr_num = f"PR{pr_counter}"
            pr_counter += 1
            pr_date = _rand_date(30, 15)
        elif i < 5:
            # Recent PR not yet stuck
            po_num = ""
            pr_num = f"PR{pr_counter}"
            pr_counter += 1
            pr_date = _rand_date(10, 1)
        else:
            po_num = f"PO{po_counter}"
            po_counter += 1
            pr_num = f"PR{pr_counter}"
            pr_counter += 1
            pr_date = _rand_date(120, 30)

        rows.append({
            "Cost Center":    cc,
            "Sub-Department": CC_TO_DEPT[cc],
            "Vendor":         vendor,
            "Project Task":   task,
            "PO Number":      po_num,
            "PR Number":      pr_num,
            "Baseline Value": baseline,
            "Change Orders":  fco,
            "Phased Budget":  budget,
            "PR Date":        pr_date.strftime("%Y-%m-%d") if pr_date else "",
        })

    return pd.DataFrame(rows)


def generate_sap_actuals(ops_df: pd.DataFrame) -> pd.DataFrame:
    """Return SAP actuals that partially match the Ops Tracker POs."""
    rows = []
    # Generate SAP postings for contracts that have a PO
    po_rows = ops_df[ops_df["PO Number"].str.strip() != ""]

    for _, row in po_rows.iterrows():
        baseline = float(row["Baseline Value"])
        fco = float(row["Change Orders"])
        contract_val = baseline + fco
        # Post between 20% and 90% of contract value
        total_posted = round(contract_val * random.uniform(0.20, 0.90), 2)
        # Split across 1-4 posting documents
        n_docs = random.randint(1, 4)
        amounts = np.diff(np.sort(np.append(
            np.random.uniform(0, total_posted, n_docs - 1), [0, total_posted]
        )))

        for amt in amounts:
            post_date = _rand_date(180, 10)
            rows.append({
                "Cost Center":   row["Cost Center"],
                "Purchase Order": row["PO Number"],
                "Purchase Req.":  row["PR Number"],
                "Vendor":         row["Vendor"],
                "Amount in LC":   round(amt, 2),
                "Commitment":     0,
                "Posting Date":   post_date.strftime("%Y-%m-%d"),
                "Text":           row["Project Task"],
                "G/L Account":    f"63{random.randint(10000, 99999)}",
                "Period":         post_date.strftime("%m/%Y"),
            })

    # Add one SAP row with no Ops Tracker match (orphan)
    rows.append({
        "Cost Center":   "0001101001",
        "Purchase Order": "PO_ORPHAN_99",
        "Purchase Req.":  "",
        "Vendor":         "Legacy Contractor Pty Ltd",
        "Amount in LC":   45_000.00,
        "Commitment":     0,
        "Posting Date":   _rand_date(60).strftime("%Y-%m-%d"),
        "Text":           "Legacy Invoice — no contract on tracker",
        "G/L Account":    "6310099",
        "Period":         datetime.now().strftime("%m/%Y"),
    })

    return pd.DataFrame(rows)


def ops_to_excel_bytes(ops_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    ops_df.to_excel(buf, index=False)
    return buf.getvalue()


def sap_to_excel_bytes(sap_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    sap_df.to_excel(buf, index=False)
    return buf.getvalue()
