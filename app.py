"""
MineDept Cost & Contract Tracker
=================================
Run:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.cleaner    import read_file, clean_sap, is_cost_element_report, clean_cost_element_report
from engine.metrics    import compute_burn_flags
from utils.persistence import (
    load_contracts, save_contracts,
    load_tmm,       save_tmm,
    load_sap_db,    save_sap_db,
    upsert_sap_period, sap_period_summary,
    save_contract_doc, list_contract_docs,
    load_contract_doc, delete_contract_doc,
    CONTRACT_COLS,  TMM_COLS,
)
from utils.exporter import export_reconciled_matrix

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MineDept Cost & Contract Tracker",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root { --navy:#1B3A5C; --amber:#C9872A; --light:#F4F6F9; }
[data-testid="stSidebar"]   { background-color: var(--navy); }
[data-testid="stSidebar"] * { color:#FFFFFF !important; }
h1,h2,h3 { font-family:Arial,sans-serif; color:var(--navy); }
.stButton>button {
    background-color:var(--amber); color:white; border:none;
    font-family:Arial,sans-serif; font-weight:bold;
}
.stButton>button:hover { background-color:#a56e1c; }
div[data-testid="stMetric"] {
    background:white; border-radius:8px; padding:12px 16px;
    border-left:4px solid var(--amber);
    box-shadow:0 1px 4px rgba(0,0,0,.08);
}
.help-box {
    background:#F0F4F8; border-left:4px solid #1B3A5C;
    padding:16px 20px; border-radius:6px; margin:8px 0;
    font-family:Arial,sans-serif; font-size:14px;
}
.tip-box {
    background:#FFF8E1; border-left:4px solid #C9872A;
    padding:12px 16px; border-radius:6px; margin:8px 0;
    font-family:Arial,sans-serif; font-size:13px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SUB_DEPTS = [
    "Mine Operations", "Mine Maintenance", "Engineering",
    "Geology", "Technical Services", "Environmental", "Other",
]

PAGES = [
    "📖 How to Use",
    "📈 Executive Summary",
    "📊 Dashboard",
    "📋 Contract Tracker",
    "⛏️ TMM Tracker",
    "🔗 SAP Sync",
    "📤 Export",
]

# ── Session state init ────────────────────────────────────────────────────────
def _init_state() -> None:
    if "contracts" not in st.session_state:
        st.session_state["contracts"] = load_contracts()
    if "tmm" not in st.session_state:
        st.session_state["tmm"] = load_tmm()
    if "sap_df" not in st.session_state:
        # Load persistent SAP database on startup
        st.session_state["sap_df"] = load_sap_db()
    if "sap_filename" not in st.session_state:
        st.session_state["sap_filename"] = ""
    if "sap_bytes" not in st.session_state:
        st.session_state["sap_bytes"] = None
    if "page" not in st.session_state:
        st.session_state["page"] = "📖 How to Use"

_init_state()

# ── Helpers ───────────────────────────────────────────────────────────────────

def contracts() -> pd.DataFrame:
    return st.session_state["contracts"]

def tmm() -> pd.DataFrame:
    return st.session_state["tmm"]

def _add_contract_calcs(df: pd.DataFrame) -> pd.DataFrame:
    """Append read-only calculated columns to the contract table."""
    out = df.copy()
    out["original_budget"] = pd.to_numeric(out["original_budget"], errors="coerce").fillna(0)
    out["amount_spent"]    = pd.to_numeric(out["amount_spent"],    errors="coerce").fillna(0)
    out["amount_left"]     = out["original_budget"] - out["amount_spent"]

    def _status(row: pd.Series) -> str:
        if row["original_budget"] == 0:
            return "⚪ No Budget"
        pct = row["amount_spent"] / row["original_budget"] * 100
        if row["amount_left"] < 0:
            return "🔴 Over Budget"
        if pct >= 85:
            return "🟡 At Risk"
        return "🟢 On Track"

    out["status"] = out.apply(_status, axis=1)
    return out

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTH_ORDER = {m: i for i, m in enumerate(MONTHS)}


def _sap_monthly_costs(sap_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SAP posted amounts by calendar year + month name."""
    df = sap_df[sap_df["date"].notna()].copy()
    df["year"]  = df["date"].dt.year.astype(float)
    df["month"] = df["date"].dt.strftime("%B")   # "January" … "December"
    return (
        df.groupby(["year", "month"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "sap_cost"})
    )


def _add_tmm_calcs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append total_cost (from SAP actuals) and cost_per_ton to the TMM table.
    If no SAP data is loaded, total_cost is shown as 0.
    """
    out = df.copy()
    out["tons"] = pd.to_numeric(out["tons"], errors="coerce").fillna(0)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").fillna(0).astype(float)

    sap_df = st.session_state.get("sap_df")
    if sap_df is not None and not sap_df.empty and "date" in sap_df.columns:
        sap_costs = _sap_monthly_costs(sap_df)
        out = out.merge(sap_costs, on=["year", "month"], how="left")
        out["total_cost"] = out["sap_cost"].fillna(0)
        out = out.drop(columns=["sap_cost"], errors="ignore")
    else:
        out["total_cost"] = 0.0

    out["cost_per_ton"] = np.where(
        out["tons"] > 0,
        (out["total_cost"] / out["tons"]).round(2),
        0.0,
    )
    return out

def _fmt_dollar(v: float) -> str:
    return f"${v:,.0f}"

# ── SAP analytics helpers ─────────────────────────────────────────────────────

def _sap_db() -> pd.DataFrame:
    df = st.session_state.get("sap_df")
    return df if (df is not None and not df.empty) else pd.DataFrame()

def _has_plan(sap_df: pd.DataFrame) -> bool:
    return "plan" in sap_df.columns and sap_df["plan"].abs().sum() > 0

def _available_periods(sap_df: pd.DataFrame) -> list[tuple[int, str]]:
    if sap_df.empty or "date" not in sap_df.columns:
        return []
    d = sap_df[sap_df["date"].notna()].copy()
    d["_yr"]  = d["date"].dt.year
    d["_mo"]  = d["date"].dt.strftime("%B")
    d["_mo_n"]= d["date"].dt.month
    rows = d[["_yr", "_mo", "_mo_n"]].drop_duplicates().sort_values(["_yr", "_mo_n"])
    return [(int(r["_yr"]), r["_mo"]) for _, r in rows.iterrows()]

def _filter_period(sap_df: pd.DataFrame, year: int, month: str) -> pd.DataFrame:
    df = sap_df[sap_df["date"].notna()].copy()
    return df[
        (df["date"].dt.year == year) &
        (df["date"].dt.strftime("%B") == month)
    ].reset_index(drop=True)

def _filter_ytd(sap_df: pd.DataFrame, year: int, through_month: str) -> pd.DataFrame:
    mo_num = _MONTH_ORDER.get(through_month, 0) + 1
    df = sap_df[sap_df["date"].notna()].copy()
    return df[
        (df["date"].dt.year == year) &
        (df["date"].dt.month <= mo_num)
    ].reset_index(drop=True)

def _build_treemap(df: pd.DataFrame, hp: bool) -> go.Figure:
    """3-level interactive treemap: Top → Mid → GL code. Colored by vs-plan %."""
    hh = "mid_cost_element" in df.columns
    ids, labels, parents, values, colors, hover = [], [], [], [], [], []

    def _var_color(actual: float, plan: float) -> float:
        if not hp or plan == 0:
            return 0.0
        return max(-60.0, min(60.0, (plan - actual) / plan * 100))

    def _add(id_, label, parent, actual, plan, tip=""):
        ids.append(id_); labels.append(label); parents.append(parent)
        values.append(max(0.01, float(actual)))
        colors.append(_var_color(float(actual), float(plan)))
        hover.append(tip or label)

    total_actual = df["amount"].sum()
    total_plan   = df["plan"].sum() if hp else total_actual
    _add("root", f"All Costs", "", total_actual, total_plan,
         f"Total Actual: {_fmt_dollar(total_actual)}"
         + (f"<br>Plan: {_fmt_dollar(total_plan)}" if hp else ""))

    skipped = []
    for top, tgrp in df.groupby("vendor"):
        if not top or str(top) in ("0", "nan", ""):
            continue
        t_act = tgrp["amount"].sum()
        if t_act <= 0:
            skipped.append(str(top))
            continue
        t_plan = tgrp["plan"].sum() if hp else t_act
        top_id = f"T|{top}"
        _add(top_id, top, "root", t_act, t_plan,
             f"<b>{top}</b><br>Actual: {_fmt_dollar(t_act)}"
             + (f"<br>Plan: {_fmt_dollar(t_plan)}<br>Var: {_fmt_dollar(t_plan-t_act)}" if hp else ""))

        mids = (
            [m for m in tgrp["mid_cost_element"].unique()
             if m and str(m) not in ("0", "nan", "")]
            if hh else []
        )

        def _add_gl_rows(subgrp, parent_id):
            for _, row in subgrp.iterrows():
                act = float(row["amount"])
                if act <= 0:
                    continue
                gl   = str(row.get("gl_account", ""))
                desc = str(row.get("description", ""))[:50]
                pln  = float(row.get("plan", act)) if hp else act
                gl_id = f"GL|{parent_id}|{gl}"
                _add(gl_id, desc or gl, parent_id, act, pln,
                     f"{desc}<br>GL: {gl}<br>Actual: {_fmt_dollar(act)}"
                     + (f"<br>Plan: {_fmt_dollar(pln)}" if hp else ""))

        if mids:
            for mid, mgrp in tgrp.groupby("mid_cost_element"):
                if not mid or str(mid) in ("0", "nan", ""):
                    _add_gl_rows(mgrp, top_id)
                    continue
                m_act  = mgrp["amount"].sum()
                m_plan = mgrp["plan"].sum() if hp else m_act
                mid_id = f"M|{top}|{mid}"
                _add(mid_id, mid, top_id, m_act, m_plan,
                     f"<b>{top} › {mid}</b><br>Actual: {_fmt_dollar(m_act)}"
                     + (f"<br>Plan: {_fmt_dollar(m_plan)}" if hp else ""))
                _add_gl_rows(mgrp, mid_id)
        else:
            _add_gl_rows(tgrp, top_id)

    colorscale = [
        [0.00, "#B71C1C"], [0.30, "#EF9A9A"],
        [0.50, "#F5F5F5"],
        [0.70, "#A5D6A7"], [1.00, "#1B5E20"],
    ]
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values,
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        texttemplate="<b>%{label}</b>",
        marker=dict(
            colors=colors, colorscale=colorscale, cmid=0,
            showscale=hp,
            colorbar=dict(
                title=dict(text="vs Plan %", side="right"),
                tickformat="+.0f", ticksuffix="%",
                len=0.6, thickness=14,
            ),
        ),
        branchvalues="total",
        maxdepth=3,
        pathbar=dict(visible=True),
    ))
    fig.update_layout(
        height=520, margin=dict(t=10, l=10, r=10, b=10),
        paper_bgcolor="white",
    )
    if skipped:
        fig.add_annotation(
            text=f"Note: {', '.join(skipped)} excluded (net credit this period)",
            xref="paper", yref="paper", x=0, y=-0.02,
            showarrow=False, font=dict(size=10, color="#888"),
        )
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='color:white;font-family:Arial;margin-bottom:2px;'>⛏️ MineDept Tracker</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#C9872A;font-size:12px;font-family:Arial;margin-top:0;'>"
        "Cost · Contracts · TMM</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio("Navigation", PAGES,
                    index=PAGES.index(st.session_state["page"]),
                    label_visibility="collapsed")
    st.session_state["page"] = page

    st.divider()
    # Quick status
    c = contracts()
    t = tmm()
    st.markdown(f"**Contracts:** {len(c):,} rows")
    total_budget  = pd.to_numeric(c["original_budget"], errors="coerce").sum()
    total_spent   = pd.to_numeric(c["amount_spent"],    errors="coerce").sum()
    st.markdown(f"**Budget:** {_fmt_dollar(total_budget)}")
    st.markdown(f"**Spent:**  {_fmt_dollar(total_spent)}")
    total_tons = pd.to_numeric(t["tons"], errors="coerce").sum()
    st.markdown(f"**TMM Rows:** {len(t):,} | **Tons:** {total_tons:,.0f}")
    _sap = st.session_state["sap_df"]
    _sap_lines = len(_sap) if (_sap is not None and not _sap.empty) else 0
    if _sap_lines:
        st.markdown(f"**SAP DB:** ✅ {_sap_lines:,} lines")
    else:
        st.markdown("**SAP DB:** ⚪ Empty")

# ── Page router ───────────────────────────────────────────────────────────────
page = st.session_state["page"]

# =============================================================================
# PAGE: 📖 How to Use
# =============================================================================
if page == "📖 How to Use":
    st.title("📖 How to Use — Training Guide")
    st.markdown(
        "_This guide explains every feature of the MineDept Cost & Contract Tracker. "
        "Print or share this page with new hires._"
    )

    # ── Overview ──
    st.markdown("---")
    st.markdown("## What This App Does")
    st.markdown("""
<div class="help-box">
The <strong>MineDept Cost & Contract Tracker</strong> is a financial control tool that lets you:
<ul>
<li>Maintain a live contract register (budget, spend, and remaining balance per contract)</li>
<li>Track Total Material Moved (TMM) in tons and automatically calculate <strong>Cost per Ton</strong></li>
<li>Optionally upload SAP financial data to auto-fill actual spend figures</li>
<li>Monitor budget burn rates and flag at-risk cost centres in real time</li>
<li>Export a fully reconciled spreadsheet for reporting</li>
</ul>
</div>
""", unsafe_allow_html=True)

    # ── Step by step ──
    st.markdown("---")
    st.markdown("## Step-by-Step Workflow")

    with st.expander("✅ Step 1 — Set Up Your Contract Tracker", expanded=True):
        st.markdown("""
Go to **📋 Contract Tracker** in the sidebar.

1. Click **+ Add row** at the bottom of the table to insert a new contract line.
2. Fill in the following fields for each contract:

| Column | What to Enter |
|---|---|
| **Cost Center** | Your 10-digit SAP cost centre (e.g. `0001101001`) |
| **Sub-Department** | Select from the dropdown (Mine Operations, Maintenance, etc.) |
| **Vendor** | Contractor or supplier name |
| **Project Task** | Short description — e.g. "Shovel Overhaul P&H 4100" |
| **PR Number** | Purchase Requisition number from SAP |
| **PO Number** | Purchase Order number once approved |
| **Original Budget ($)** | The approved contract value |
| **Amount Spent ($)** | Actual spend to date — enter manually OR sync from SAP (Step 3) |
| **Notes** | Any comments, status updates, or action items |

3. **Amount Left** and **Status** are calculated automatically — do not enter them.
4. Click **💾 Save Changes** after every editing session. Data is stored locally so it survives app restarts.

<div class="tip-box">
💡 <strong>Tip:</strong> You can edit any cell directly by clicking it.
To delete a row, select it and press the <strong>Delete</strong> key or use the row checkbox.
</div>
""", unsafe_allow_html=True)

    with st.expander("✅ Step 2 — Track TMM (Total Material Moved)"):
        st.markdown("""
Go to **⛏️ TMM Tracker** in the sidebar.

1. Click **+ Add row** to insert a new entry — one row per month.
2. Fill in only these fields:

| Column | What to Enter |
|---|---|
| **Year** | The calendar year — e.g. `2025` |
| **Month** | Select the month from the dropdown |
| **Tons Moved** | Total tonnes moved that month |

3. **Total Cost ($)** and **Cost per Ton** are **read-only — calculated automatically from SAP data**.
   Upload your SAP actuals on the **SAP Sync** page and the app matches costs by posting date.
4. Click **💾 Save Changes** after editing.

<div class="tip-box">
💡 <strong>How cost is linked:</strong> The app sums all SAP posting amounts whose Posting Date
falls in the selected Year + Month, then divides by Tons Moved to produce Cost per Ton.
Cost columns show 0 until SAP data is uploaded on the <strong>SAP Sync</strong> page.
</div>
""", unsafe_allow_html=True)

    with st.expander("✅ Step 3 — (Optional) Sync Actual Spend from SAP"):
        st.markdown("""
Go to **🔗 SAP Sync** in the sidebar.

This step is **optional but recommended**. Instead of manually typing Amount Spent for each
contract, you can upload a SAP financial export and the app will match PO/PR numbers
and fill in actual spend automatically.

1. Export your SAP actuals — see "SAP vs Spreadsheet" section below for the best report to use.
2. Upload the file on the **SAP Sync** page.
3. Click **🔄 Sync Actuals to Contract Tracker**.
4. The app will match PO and PR numbers and update Amount Spent for any matched rows.
5. Rows updated from SAP are marked with ✅ so you know which figures are system-sourced.
""")

    with st.expander("✅ Step 4 — Review the Dashboard"):
        st.markdown("""
Go to **📊 Dashboard** in the sidebar.

The Dashboard shows:
- **KPI cards** — Total Budget, Total Spent, Total Remaining, Overall Cost per Ton
- **Budget vs Spend by Sub-Department** — horizontal bar chart
- **Burn Rate Flags** — Red 🔴 / Yellow 🟡 / Green 🟢 per cost centre
- **TMM Trend** — Cost per Ton over time as a line chart

The Dashboard updates live as you edit the Contract Tracker or TMM tables — no refresh needed.
""")

    with st.expander("✅ Step 5 — Export for Reporting"):
        st.markdown("""
Go to **📤 Export** in the sidebar.

- Click **Generate Export** to download the full reconciled table as a formatted Excel workbook.
- The export includes all contracts, calculated fields, and any notes you've entered.
- Rows where Remaining Budget is negative are highlighted in red automatically.
""")

    # ── SAP vs Spreadsheet ──
    st.markdown("---")
    st.markdown("## SAP Export vs Manual Spreadsheet — Which Is Better?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
<div class="help-box">
<strong>✅ SAP Export (Recommended)</strong><br><br>
• <strong>Source of truth</strong> — every posted document is captured<br>
• No manual data entry = no transcription errors<br>
• Supports automatic reconciliation against your contract register<br>
• Provides GL account, fiscal period, and document-level detail<br>
• Full audit trail<br><br>
<strong>Best SAP reports to export:</strong><br>
• <code>KSB1</code> — Actual cost line items by cost centre<br>
• <code>ME2N</code> — Purchase orders by cost centre<br>
• <code>S_ALR_87013611</code> — Cost centre actual vs plan<br><br>
Export as <em>.xlsx</em> or <em>.csv</em> and upload on the SAP Sync page.
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="tip-box">
<strong>⚠️ Manual Spreadsheet (Use when necessary)</strong><br><br>
• No SAP access required<br>
• You control which data is included<br>
• Useful for commitments not yet in SAP (verbal orders, estimates)<br><br>
<strong>Downsides:</strong><br>
• Human error risk on every entry<br>
• Must be updated manually each period<br>
• Easy to miss postings or double-count<br><br>
<strong>Recommendation:</strong> Use SAP for actuals, and use the built-in
<em>Contract Tracker</em> table in this app for commitments and
outstanding obligations that are not yet posted in SAP.
</div>
""", unsafe_allow_html=True)

    # ── Column reference ──
    st.markdown("---")
    st.markdown("## Column Definitions — Quick Reference")
    st.markdown("""
| Term | Definition |
|---|---|
| **Original Budget** | The approved contract or purchase order value |
| **Amount Spent** | Actual cost posted to SAP (or manually entered) |
| **Amount Left** | Original Budget − Amount Spent (can be negative if over-run) |
| **Status** | 🟢 On Track (<85% spent) · 🟡 At Risk (≥85%) · 🔴 Over Budget (negative remaining) |
| **PR Number** | Purchase Requisition — raised internally, not yet a commitment in SAP |
| **PO Number** | Purchase Order — approved commitment visible in SAP |
| **TMM** | Total Material Moved — tonnes of overburden or ore moved in a period |
| **Cost per Ton** | Total Cost ÷ Tons Moved — key efficiency metric for mining operations |
| **Burn Rate** | How fast a cost centre is consuming its budget relative to its target |
""")

    st.markdown("---")
    st.info(
        "**Need help?** Contact your site Finance Business Partner or the app administrator. "
        "All data is stored locally on this machine — nothing is sent to external servers."
    )


# =============================================================================
# PAGE: 📈 Executive Summary
# =============================================================================
elif page == "📈 Executive Summary":
    st.title("📈 Executive Summary")
    st.markdown("_Monthly cost performance, year-to-date tracking, and key areas of focus._")

    sap = _sap_db()
    if sap.empty:
        st.info(
            "No SAP cost data loaded yet. Upload your monthly cost element file "
            "on the **🔗 SAP Sync** page to enable this page."
        )
        st.stop()

    periods = _available_periods(sap)
    if not periods:
        st.stop()

    hp = _has_plan(sap)
    period_labels = [f"{mo} {yr}" for yr, mo in periods]
    pc, _ = st.columns([2, 5])
    with pc:
        sel_label = st.selectbox(
            "Reporting Period", period_labels,
            index=len(period_labels) - 1, key="exec_period",
        )
    sel_yr, sel_mo = next((yr, mo) for yr, mo in periods if f"{mo} {yr}" == sel_label)

    curr        = _filter_period(sap, sel_yr, sel_mo)
    curr_actual = curr["amount"].sum()
    curr_plan   = curr["plan"].sum() if hp else 0
    curr_var    = curr_plan - curr_actual
    curr_vp     = curr_var / curr_plan * 100 if curr_plan != 0 else 0

    mo_idx = _MONTH_ORDER.get(sel_mo, 0)
    p_yr, p_mo = (sel_yr - 1, "December") if mo_idx == 0 else (sel_yr, MONTHS[mo_idx - 1])
    prior        = _filter_period(sap, p_yr, p_mo)
    prior_actual = prior["amount"].sum()
    mom_chg      = curr_actual - prior_actual
    mom_pct      = mom_chg / prior_actual * 100 if prior_actual else 0

    ytd        = _filter_ytd(sap, sel_yr, sel_mo)
    ytd_actual = ytd["amount"].sum()
    ytd_plan   = ytd["plan"].sum() if hp else 0
    ytd_var    = ytd_plan - ytd_actual
    ytd_vp     = ytd_var / ytd_plan * 100 if ytd_plan != 0 else 0
    mo_count   = _MONTH_ORDER.get(sel_mo, 0) + 1

    # ── KPI row 1: current month ──────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Actual — {sel_mo[:3]} {sel_yr}", _fmt_dollar(curr_actual))
    if hp:
        k2.metric("Plan", _fmt_dollar(curr_plan))
        k3.metric(
            "Variance vs Plan", _fmt_dollar(curr_var),
            delta=f"{curr_vp:+.1f}%",
            delta_color="normal" if curr_var >= 0 else "inverse",
            help="Positive = under plan (good). Negative = over plan.",
        )
    else:
        k2.metric("Plan", "N/A — upload cost element file")
        k3.metric("Variance", "N/A")
    if prior_actual:
        k4.metric(
            f"vs {p_mo[:3]} {p_yr}", _fmt_dollar(mom_chg),
            delta=f"{mom_pct:+.1f}%",
            delta_color="inverse" if mom_chg > 0 else "normal",
        )
    else:
        k4.metric("vs Prior Month", "—")

    # ── KPI row 2: YTD ───────────────────────────────────────────────────────
    if hp:
        y1, y2, y3, y4 = st.columns(4)
        y1.metric(f"YTD Actual ({sel_yr})", _fmt_dollar(ytd_actual))
        y2.metric("YTD Plan",               _fmt_dollar(ytd_plan))
        y3.metric(
            "YTD Variance", _fmt_dollar(ytd_var),
            delta=f"{ytd_vp:+.1f}%",
            delta_color="normal" if ytd_var >= 0 else "inverse",
        )
        y4.metric("Months in YTD", str(mo_count))

    st.divider()

    # ── Category breakdown ────────────────────────────────────────────────────
    cat_df = (
        curr.groupby("vendor", as_index=False)
        .agg(actual=("amount", "sum"),
             plan  =("plan",   "sum") if hp else ("amount", "sum"))
    )
    cat_df = cat_df[cat_df["actual"].abs() > 0].copy()
    if hp:
        cat_df["variance"] = cat_df["plan"] - cat_df["actual"]
        cat_df["var_pct"]  = (
            cat_df["variance"] / cat_df["plan"].replace(0, np.nan) * 100
        ).fillna(0)
        over_plan  = cat_df[cat_df["variance"] < 0].sort_values("variance")
        under_plan = cat_df[cat_df["variance"] > 0].sort_values("variance", ascending=False)
    else:
        over_plan = under_plan = pd.DataFrame()

    # ── Insight narrative ─────────────────────────────────────────────────────
    trend_sap = (
        sap[sap["date"].notna()].copy()
        .assign(_yr=lambda d: d["date"].dt.year, _mo_n=lambda d: d["date"].dt.month)
        .groupby(["_yr", "_mo_n"], as_index=False)
        .agg(actual=("amount", "sum"))
        .sort_values(["_yr", "_mo_n"])
    )
    if len(trend_sap) >= 3:
        l = trend_sap.tail(3)["actual"].tolist()
        trend_txt = (
            "📈 Costs have been rising over the last 3 months."
            if l[2] > l[1] > l[0] else
            "📉 Costs have been declining over the last 3 months — positive trend."
            if l[2] < l[1] < l[0] else
            "↔️ Costs have been relatively stable over the last 3 months."
        )
    else:
        trend_txt = ""

    lines = []
    if hp:
        icon = "✅" if curr_var >= 0 else "⚠️"
        word = "under plan" if curr_var >= 0 else "over plan"
        lines.append(
            f"{icon} **{sel_mo} {sel_yr}:** Total actual spend is "
            f"**{_fmt_dollar(curr_actual)}**, **{abs(curr_vp):.1f}% {word}** "
            f"(plan: {_fmt_dollar(curr_plan)})."
        )
        lines.append(
            f"**YTD through {sel_mo} {sel_yr}:** {_fmt_dollar(ytd_actual)} actual vs "
            f"{_fmt_dollar(ytd_plan)} plan — "
            f"**{abs(ytd_vp):.1f}% {'under' if ytd_var >= 0 else 'over'} plan**."
        )
        if not over_plan.empty:
            r = over_plan.iloc[0]
            lines.append(
                f"🔴 **Biggest over-plan area:** {r['vendor']} "
                f"({_fmt_dollar(r['actual'])} actual vs {_fmt_dollar(r['plan'])} plan, "
                f"{abs(r['var_pct']):.1f}% over plan)."
            )
        if not under_plan.empty:
            r = under_plan.iloc[0]
            lines.append(
                f"🟢 **Best performing:** {r['vendor']} "
                f"({abs(r['var_pct']):.1f}% under plan, "
                f"{_fmt_dollar(r['variance'])} saving vs budget)."
            )
    else:
        lines.append(
            f"**{sel_mo} {sel_yr}:** Total actual spend is **{_fmt_dollar(curr_actual)}**."
        )
        lines.append(
            f"**YTD through {sel_mo} {sel_yr}:** {_fmt_dollar(ytd_actual)} cumulative."
        )
    if trend_txt:
        lines.append(trend_txt)
    if prior_actual:
        direction = "increased" if mom_chg > 0 else "decreased"
        lines.append(
            f"📅 **Month-on-month:** costs {direction} by "
            f"{_fmt_dollar(abs(mom_chg))} ({abs(mom_pct):.1f}%) vs {p_mo} {p_yr}."
        )

    st.markdown(
        '<div class="help-box">' + "<br>".join(lines) + "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Spend chart + table ───────────────────────────────────────────────────
    ch_col, tbl_col = st.columns([3, 2])

    with ch_col:
        st.markdown(f"### Spend by Category — {sel_mo} {sel_yr}")
        cat_sorted = cat_df.sort_values("actual", ascending=True)

        bar_colors = []
        for _, row in cat_sorted.iterrows():
            if hp:
                bar_colors.append(
                    "#C62828" if row["variance"] < 0
                    else "#2E7D32" if row["variance"] > 0
                    else "#C9872A"
                )
            else:
                bar_colors.append("#1B3A5C")

        fig_main = go.Figure()
        if hp:
            fig_main.add_trace(go.Bar(
                name="Plan", y=cat_sorted["vendor"], x=cat_sorted["plan"],
                orientation="h", marker_color="#B0BEC5", opacity=0.85,
            ))
        fig_main.add_trace(go.Bar(
            name="Actual", y=cat_sorted["vendor"], x=cat_sorted["actual"],
            orientation="h", marker_color=bar_colors,
        ))
        fig_main.update_layout(
            barmode="overlay", height=max(320, len(cat_sorted) * 62),
            margin=dict(l=0, r=10, t=10, b=10),
            xaxis=dict(tickprefix="$", tickformat=",.0f",
                       showgrid=True, gridcolor="#EEE"),
            legend=dict(orientation="h", y=1.08),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial", size=12),
        )
        st.plotly_chart(fig_main, use_container_width=True)

    with tbl_col:
        st.markdown("### Performance Table")
        disp = cat_df.sort_values("actual", ascending=False).copy()
        disp["Actual"] = disp["actual"].apply(_fmt_dollar)
        if hp:
            disp["Plan"]     = disp["plan"].apply(_fmt_dollar)
            disp["Variance"] = disp["variance"].apply(
                lambda v: ("▲ " if v >= 0 else "▼ ") + _fmt_dollar(abs(v))
            )
            disp["Var %"] = disp["var_pct"].apply(lambda v: f"{v:+.1f}%")
            out_cols = ["vendor", "Actual", "Plan", "Variance", "Var %"]
        else:
            out_cols = ["vendor", "Actual"]
        st.dataframe(
            disp[out_cols].rename(columns={"vendor": "Category"}),
            use_container_width=True, hide_index=True,
        )

    # ── Month-on-month change ─────────────────────────────────────────────────
    if not prior.empty:
        st.divider()
        st.markdown(f"### Month-on-Month Change vs {p_mo} {p_yr}")
        prior_cat = prior.groupby("vendor", as_index=False).agg(prior=("amount", "sum"))
        mom_df = cat_df.merge(prior_cat, on="vendor", how="outer").fillna(0)
        mom_df["change"] = mom_df["actual"] - mom_df["prior"]
        mom_df = mom_df.sort_values("change")
        mom_colors = ["#C62828" if v > 0 else "#2E7D32" for v in mom_df["change"]]
        fig_mom = go.Figure(go.Bar(
            y=mom_df["vendor"], x=mom_df["change"],
            orientation="h", marker_color=mom_colors,
            text=mom_df["change"].apply(
                lambda v: ("+" if v > 0 else "") + _fmt_dollar(v)
            ),
            textposition="outside",
        ))
        fig_mom.update_layout(
            height=max(260, len(mom_df) * 55),
            margin=dict(l=0, r=110, t=10, b=10),
            xaxis=dict(tickprefix="$", tickformat=",.0f",
                       showgrid=True, gridcolor="#EEE",
                       zeroline=True, zerolinecolor="#555"),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial", size=12),
            showlegend=False,
        )
        st.caption(
            f"Red = cost increased vs {p_mo} {p_yr} (watch these areas). "
            f"Green = cost decreased (improvement)."
        )
        st.plotly_chart(fig_mom, use_container_width=True)

    # ── Areas of focus ────────────────────────────────────────────────────────
    if hp and not over_plan.empty:
        st.divider()
        st.markdown("### 🎯 Areas of Focus — Over Plan")
        cols = st.columns(min(3, len(over_plan)))
        for i, (_, row) in enumerate(over_plan.head(3).iterrows()):
            with cols[i]:
                st.markdown(f"""
<div style="background:#FDECEA;border-left:4px solid #C62828;
padding:14px 16px;border-radius:6px;font-family:Arial;font-size:14px;">
<strong>⚠️ {row['vendor']}</strong><br><br>
Actual: <strong>{_fmt_dollar(row['actual'])}</strong><br>
Plan: {_fmt_dollar(row['plan'])}<br>
Over by: <strong style="color:#C62828">{_fmt_dollar(abs(row['variance']))}</strong>
<span style="color:#C62828"> ({abs(row['var_pct']):.1f}%)</span>
</div>""", unsafe_allow_html=True)

    if hp and not under_plan.empty:
        st.divider()
        st.markdown("### ✅ Well-Performing Areas")
        cols = st.columns(min(3, len(under_plan)))
        for i, (_, row) in enumerate(under_plan.head(3).iterrows()):
            with cols[i]:
                st.markdown(f"""
<div style="background:#E8F5E9;border-left:4px solid #2E7D32;
padding:14px 16px;border-radius:6px;font-family:Arial;font-size:14px;">
<strong>✅ {row['vendor']}</strong><br><br>
Actual: <strong>{_fmt_dollar(row['actual'])}</strong><br>
Plan: {_fmt_dollar(row['plan'])}<br>
Under by: <strong style="color:#2E7D32">{_fmt_dollar(row['variance'])}</strong>
<span style="color:#2E7D32"> ({abs(row['var_pct']):.1f}%)</span>
</div>""", unsafe_allow_html=True)


# =============================================================================
# PAGE: 📊 Dashboard
# =============================================================================
elif page == "📊 Dashboard":
    st.title("📊 Cost Explorer")

    sap   = _sap_db()
    c_df  = _add_contract_calcs(contracts())
    t_df  = _add_tmm_calcs(tmm())
    has_sap = not sap.empty
    hp      = _has_plan(sap) if has_sap else False

    # ── Period selector ───────────────────────────────────────────────────────
    period_df = pd.DataFrame()
    prior_df  = pd.DataFrame()
    sel_yr, sel_mo, p_yr, p_mo = 0, "", 0, ""

    if has_sap:
        periods       = _available_periods(sap)
        period_labels = [f"{mo} {yr}" for yr, mo in periods]
        pc1, pc2, _ = st.columns([2, 3, 3])
        with pc1:
            sel_label = st.selectbox(
                "Analysis Period", period_labels,
                index=len(period_labels) - 1, key="dash_period",
            )
        sel_yr, sel_mo = next((yr, mo) for yr, mo in periods if f"{mo} {yr}" == sel_label)
        period_df = _filter_period(sap, sel_yr, sel_mo)
        mo_idx = _MONTH_ORDER.get(sel_mo, 0)
        p_yr, p_mo = (sel_yr - 1, "December") if mo_idx == 0 else (sel_yr, MONTHS[mo_idx - 1])
        prior_df  = _filter_period(sap, p_yr, p_mo)
        with pc2:
            st.markdown(
                f"<div style='padding-top:28px;color:#666;font-size:13px;'>"
                f"Comparing vs {p_mo} {p_yr}</div>",
                unsafe_allow_html=True,
            )

    # ── KPI row ───────────────────────────────────────────────────────────────
    total_tons     = t_df["tons"].sum()     if not t_df.empty else 0
    total_tmm_cost = t_df["total_cost"].sum() if not t_df.empty else 0
    overall_cpt    = total_tmm_cost / total_tons if total_tons > 0 else 0

    if has_sap and not period_df.empty:
        curr_act  = period_df["amount"].sum()
        curr_pln  = period_df["plan"].sum() if hp else 0
        prior_act = prior_df["amount"].sum() if not prior_df.empty else 0
        mom_chg   = curr_act - prior_act
        mom_pct   = mom_chg / prior_act * 100 if prior_act else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            f"{sel_mo[:3]} {sel_yr} Actual", _fmt_dollar(curr_act),
            delta=f"{mom_pct:+.1f}% vs {p_mo[:3]}",
            delta_color="inverse" if mom_chg > 0 else "normal",
        )
        if hp:
            var = curr_pln - curr_act
            k2.metric("Plan", _fmt_dollar(curr_pln))
            k3.metric(
                "vs Plan", _fmt_dollar(var),
                delta=f"{var/curr_pln*100:+.1f}%" if curr_pln else None,
                delta_color="normal" if var >= 0 else "inverse",
            )
        else:
            k2.metric("Plan", "—"); k3.metric("vs Plan", "—")
        k4.metric("Cost / Ton", f"${overall_cpt:,.2f}" if overall_cpt else "—",
                  help="From TMM Tracker")

        over_count = (c_df["amount_left"] < 0).sum() if not c_df.empty else 0
        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Active Contracts", f"{len(c_df):,}" if not c_df.empty else "0")
        k6.metric("Over-Budget", f"{over_count:,}",
                  delta="⚠️ Review" if over_count > 0 else "All within budget",
                  delta_color="inverse" if over_count > 0 else "normal")
        k7.metric("Total Tons Moved", f"{total_tons:,.0f} t")
        k8.metric("TMM Periods", f"{len(t_df):,}" if not t_df.empty else "0")
    else:
        total_budget = c_df["original_budget"].sum() if not c_df.empty else 0
        total_spent  = c_df["amount_spent"].sum()    if not c_df.empty else 0
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Budget",    _fmt_dollar(total_budget))
        k2.metric("Total Spent",     _fmt_dollar(total_spent))
        k3.metric("Remaining",       _fmt_dollar(total_budget - total_spent))
        k4.metric("Cost / Ton",      f"${overall_cpt:,.2f}" if overall_cpt else "—")

    st.divider()

    # ── Treemap + Actual vs Plan ──────────────────────────────────────────────
    if has_sap and not period_df.empty:
        tm_col, bar_col = st.columns([3, 2])

        with tm_col:
            st.markdown("### 🗺️ Cost Breakdown — Click tiles to drill in")
            if hp:
                st.caption("Size = actual spend · Color: 🟢 under plan → 🔴 over plan")
            st.plotly_chart(_build_treemap(period_df, hp), use_container_width=True)

        with bar_col:
            st.markdown("### Actual vs Plan" if hp else "### Spend by Category")
            cat_df = (
                period_df[period_df["amount"] > 0]
                .groupby("vendor", as_index=False)
                .agg(actual=("amount", "sum"),
                     plan  =("plan",   "sum") if hp else ("amount", "sum"))
                .sort_values("actual", ascending=True)
            )
            fig_bar = go.Figure()
            if hp:
                fig_bar.add_trace(go.Bar(
                    name="Plan", y=cat_df["vendor"], x=cat_df["plan"],
                    orientation="h", marker_color="#B0BEC5", opacity=0.85,
                ))
            fig_bar.add_trace(go.Bar(
                name="Actual", y=cat_df["vendor"], x=cat_df["actual"],
                orientation="h", marker_color="#C9872A",
            ))
            fig_bar.update_layout(
                barmode="overlay", height=max(300, len(cat_df) * 55),
                margin=dict(l=0, r=10, t=10, b=10),
                xaxis=dict(tickprefix="$", tickformat=",.0f",
                           showgrid=True, gridcolor="#EEE"),
                legend=dict(orientation="h", y=1.08),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Arial", size=12),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    elif not has_sap:
        st.info(
            "Upload SAP cost data on the **🔗 SAP Sync** page to enable the "
            "interactive cost explorer. Contract charts appear below."
        )

    # ── Monthly trend ─────────────────────────────────────────────────────────
    if has_sap and len(_available_periods(sap)) > 1:
        st.divider()
        st.markdown("### 📈 Monthly Cost Trend — Actual vs Plan")
        trend = (
            sap[sap["date"].notna()].copy()
            .assign(_yr  =lambda d: d["date"].dt.year,
                    _mo  =lambda d: d["date"].dt.strftime("%B"),
                    _mo_n=lambda d: d["date"].dt.month)
            .groupby(["_yr", "_mo", "_mo_n"], as_index=False)
            .agg(actual=("amount", "sum"),
                 plan  =("plan",   "sum") if hp else ("amount", "sum"))
            .sort_values(["_yr", "_mo_n"])
        )
        trend["label"] = trend["_yr"].astype(int).astype(str) + " " + trend["_mo"]
        fig_tr = go.Figure()
        if hp:
            fig_tr.add_trace(go.Bar(
                name="Plan", x=trend["label"], y=trend["plan"],
                marker_color="#B0BEC5", opacity=0.7,
            ))
        fig_tr.add_trace(go.Scatter(
            name="Actual", x=trend["label"], y=trend["actual"],
            mode="lines+markers",
            line=dict(color="#C9872A", width=3),
            marker=dict(size=8, color="#1B3A5C"),
        ))
        fig_tr.update_layout(
            height=300, barmode="overlay",
            margin=dict(l=0, r=10, t=10, b=10),
            yaxis=dict(tickprefix="$", tickformat=",.0f",
                       showgrid=True, gridcolor="#EEE"),
            legend=dict(orientation="h", y=1.08),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial", size=12),
        )
        st.plotly_chart(fig_tr, use_container_width=True)

    # ── Contract burn rate + TMM ──────────────────────────────────────────────
    st.divider()
    burn_col, tmm_col = st.columns(2)

    with burn_col:
        st.markdown("### 🚦 Contract Burn Rate")
        if not c_df.empty and c_df["cost_center"].str.strip().ne("").any():
            burn = compute_burn_flags(c_df.rename(columns={
                "original_budget": "phased_budget",
                "amount_spent":    "sap_posted_amount",
                "amount_left":     "remaining_budget",
            }))
            def _flag_style(val: str) -> str:
                if "Red"    in val: return "color:#C62828;font-weight:bold;"
                if "Yellow" in val: return "color:#F57C00;font-weight:bold;"
                return "color:#2E7D32;font-weight:bold;"
            burn_disp = burn[["cost_center", "burn_rate_pct", "status"]].rename(columns={
                "cost_center": "Cost Center", "burn_rate_pct": "Burn %", "status": "Status",
            })
            burn_disp["Burn %"] = burn_disp["Burn %"].apply(lambda v: f"{v:.1f}%")
            st.dataframe(
                burn_disp.style.applymap(_flag_style, subset=["Status"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Add Cost Centre values to contracts to see burn rate flags.")

        # Sub-dept budget chart
        if not c_df.empty and "sub_dept" in c_df.columns and c_df["sub_dept"].str.strip().ne("").any():
            st.markdown("#### Budget vs Spend by Sub-Dept")
            grp = (
                c_df[c_df["sub_dept"].str.strip() != ""]
                .groupby("sub_dept", as_index=False)
                .agg(budget=("original_budget", "sum"), spent=("amount_spent", "sum"))
                .sort_values("budget", ascending=True)
            )
            fig_sub = go.Figure()
            fig_sub.add_trace(go.Bar(
                name="Budget", y=grp["sub_dept"], x=grp["budget"],
                orientation="h", marker_color="#1B3A5C",
            ))
            fig_sub.add_trace(go.Bar(
                name="Spent", y=grp["sub_dept"], x=grp["spent"],
                orientation="h", marker_color="#C9872A",
            ))
            fig_sub.update_layout(
                barmode="overlay", height=max(200, len(grp) * 55),
                margin=dict(l=0, r=10, t=10, b=10),
                xaxis=dict(tickprefix="$", tickformat=",.0f",
                           showgrid=True, gridcolor="#EEE"),
                legend=dict(orientation="h", y=1.08),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Arial", size=11),
            )
            st.plotly_chart(fig_sub, use_container_width=True)

    with tmm_col:
        st.markdown("### ⛏️ Cost per Ton Trend")
        if not t_df.empty and t_df["tons"].sum() > 0:
            t_plot = t_df[t_df["tons"] > 0].copy()
            t_plot["_mo"] = t_plot["month"].map(_MONTH_ORDER).fillna(0)
            t_plot = t_plot.sort_values(["year", "_mo"])
            t_plot["period_label"] = (
                t_plot["year"].astype(int).astype(str) + " " + t_plot["month"]
            )
            fig_tmm = go.Figure()
            fig_tmm.add_trace(go.Bar(
                name="Tons", x=t_plot["period_label"], y=t_plot["tons"],
                marker_color="#B0BEC5", opacity=0.7, yaxis="y",
            ))
            fig_tmm.add_trace(go.Scatter(
                name="Cost/Ton", x=t_plot["period_label"], y=t_plot["cost_per_ton"],
                mode="lines+markers",
                line=dict(color="#C9872A", width=3),
                marker=dict(size=8, color="#1B3A5C"),
                yaxis="y2",
            ))
            if len(t_plot) > 1:
                avg = t_plot["cost_per_ton"].mean()
                fig_tmm.add_hline(
                    y=avg, line_dash="dot", line_color="#888",
                    annotation_text=f"Avg ${avg:,.2f}/t",
                    annotation_position="bottom right",
                    yref="y2",
                )
            fig_tmm.update_layout(
                height=340,
                yaxis =dict(title="Tons",     tickformat=",.0f",  showgrid=True, gridcolor="#EEE"),
                yaxis2=dict(title="$/ton",    tickprefix="$", tickformat=",.2f",
                            overlaying="y", side="right"),
                legend=dict(orientation="h", y=1.08),
                margin=dict(l=0, r=10, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Arial", size=12),
            )
            st.plotly_chart(fig_tmm, use_container_width=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Tons", f"{t_df['tons'].sum():,.0f} t")
            m2.metric("Avg $/Ton",  f"${t_df[t_df['tons']>0]['cost_per_ton'].mean():,.2f}"
                      if not t_df[t_df['tons']>0].empty else "—")
            m3.metric("Best $/Ton", f"${t_df[t_df['cost_per_ton']>0]['cost_per_ton'].min():,.2f}"
                      if not t_df[t_df['cost_per_ton']>0].empty else "—")
        else:
            st.info("Add TMM data on the **TMM Tracker** page to see cost per ton trend.")


# =============================================================================
# PAGE: 📋 Contract Tracker
# =============================================================================
elif page == "📋 Contract Tracker":
    st.title("📋 Contract Tracker")
    st.markdown(
        "Add and update contracts below. Click **+ Add row** to create a new entry. "
        "**Amount Left** and **Status** are calculated automatically. "
        "Click **💾 Save Changes** when done."
    )

    raw = contracts().copy()
    display_df = _add_contract_calcs(raw)

    # Column config
    col_cfg = {
        "cost_center":     st.column_config.TextColumn("Cost Center", width=130,
                           help="10-digit SAP cost centre, e.g. 0001101001"),
        "sub_dept":        st.column_config.SelectboxColumn("Sub-Department",
                           options=SUB_DEPTS, width=160),
        "vendor":          st.column_config.TextColumn("Vendor / Contractor", width=200),
        "task":            st.column_config.TextColumn("Project Task / Description", width=240),
        "pr_number":       st.column_config.TextColumn("PR Number", width=120),
        "po_number":       st.column_config.TextColumn("PO Number", width=120),
        "original_budget": st.column_config.NumberColumn("Original Budget ($)",
                           format="$%.0f", min_value=0, width=150),
        "amount_spent":    st.column_config.NumberColumn("Amount Spent ($)",
                           format="$%.0f", min_value=0, width=150,
                           help="Enter manually or use SAP Sync to auto-fill"),
        "sap_synced":      st.column_config.CheckboxColumn("SAP ✅", width=70,
                           disabled=True,
                           help="Checked when Amount Spent was auto-filled from SAP"),
        "notes":           st.column_config.TextColumn("Notes", width=220),
        # Calculated — read-only
        "amount_left":     st.column_config.NumberColumn("Amount Left ($)",
                           format="$%.0f", disabled=True, width=140),
        "status":          st.column_config.TextColumn("Status", disabled=True, width=130),
    }

    # Show only meaningful columns (drop internal ones from display)
    show_cols = [c for c in display_df.columns if c != "sap_synced" or display_df["sap_synced"].any()]

    edited = st.data_editor(
        display_df[show_cols] if show_cols else display_df,
        column_config=col_cfg,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="contract_editor",
        disabled=["amount_left", "status", "sap_synced"],
    )

    col_save, col_info = st.columns([1, 5])
    with col_save:
        if st.button("💾 Save Changes", key="save_contracts"):
            # Strip calculated columns before persisting
            save_cols = [c for c in CONTRACT_COLS if c in edited.columns]
            clean = edited[save_cols].copy()
            clean["original_budget"] = pd.to_numeric(clean["original_budget"], errors="coerce").fillna(0)
            clean["amount_spent"]    = pd.to_numeric(clean["amount_spent"],    errors="coerce").fillna(0)
            save_contracts(clean)
            st.session_state["contracts"] = clean
            st.success(f"Saved {len(clean):,} contract rows.")
            st.rerun()
    with col_info:
        total_b = pd.to_numeric(edited["original_budget"], errors="coerce").sum()
        total_s = pd.to_numeric(edited["amount_spent"],    errors="coerce").sum()
        total_l = total_b - total_s
        st.markdown(
            f"**Budget:** {_fmt_dollar(total_b)} &nbsp;|&nbsp; "
            f"**Spent:** {_fmt_dollar(total_s)} &nbsp;|&nbsp; "
            f"**Remaining:** {_fmt_dollar(total_l)}",
            unsafe_allow_html=True,
        )

    # Over-budget alert
    over = edited[pd.to_numeric(edited.get("amount_left", 0), errors="coerce").fillna(0) < 0]
    if not over.empty:
        st.warning(
            f"⚠️ {len(over)} contract(s) are over budget: "
            + ", ".join(over["vendor"].dropna().unique()[:5].tolist())
        )

    # Per-sub-dept summary
    if not contracts().empty:
        st.divider()
        st.markdown("### Summary by Sub-Department")
        c_calc = _add_contract_calcs(contracts())
        if "sub_dept" in c_calc.columns:
            grp = (
                c_calc[c_calc["sub_dept"].str.strip() != ""]
                .groupby("sub_dept")
                .agg(
                    Contracts=("vendor", "count"),
                    Budget=("original_budget", "sum"),
                    Spent=("amount_spent", "sum"),
                    Remaining=("amount_left", "sum"),
                )
                .reset_index()
                .rename(columns={"sub_dept": "Sub-Department"})
            )
            for col in ("Budget", "Spent", "Remaining"):
                grp[col] = grp[col].apply(_fmt_dollar)
            st.dataframe(grp, use_container_width=True, hide_index=True)

    # ── Contract Documents ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📎 Contract Documents")
    st.markdown(
        "Attach PDFs, Excel files, or any supporting documents to individual "
        "contracts. Files are saved locally and survive app restarts."
    )

    cont_df = contracts()
    if cont_df.empty:
        st.info("Add contracts above before attaching documents.")
    else:
        def _contract_label(row: pd.Series) -> str:
            vendor = str(row.get("vendor", "")).strip()
            task   = str(row.get("task",   "")).strip()
            ref    = str(row.get("po_number", "")).strip() or str(row.get("pr_number", "")).strip()
            parts  = [p for p in [vendor, task] if p]
            if ref:
                parts.append(f"({ref})")
            return " — ".join(parts) if parts else f"Contract #{row.name}"

        contract_map = {
            _contract_label(row): (
                str(row.get("po_number", "")).strip()
                or str(row.get("pr_number", "")).strip()
                or f"row_{idx}"
            )
            for idx, row in cont_df.iterrows()
        }

        dc1, dc2 = st.columns([2, 3])
        with dc1:
            sel_contract = st.selectbox(
                "Select contract", list(contract_map.keys()), key="doc_contract_sel"
            )
            contract_key = contract_map[sel_contract]

        with dc2:
            uploaded_docs = st.file_uploader(
                "Upload documents (multiple files OK)",
                type=["pdf", "xlsx", "xls", "docx", "doc",
                      "png", "jpg", "jpeg", "csv", "txt", "pptx"],
                accept_multiple_files=True,
                key=f"doc_upload_{contract_key}",
            )
            if uploaded_docs:
                for f in uploaded_docs:
                    save_contract_doc(contract_key, f.name, f.read())
                st.success(f"Saved {len(uploaded_docs)} file(s) to **{sel_contract}**.")
                st.rerun()

        existing_docs = list_contract_docs(contract_key)
        if existing_docs:
            st.markdown(f"**Attached to:** {sel_contract}")
            for fname in existing_docs:
                fc1, fc2, fc3 = st.columns([5, 1, 1])
                with fc1:
                    st.markdown(f"📄 `{fname}`")
                with fc2:
                    doc_bytes = load_contract_doc(contract_key, fname)
                    st.download_button(
                        "⬇️", data=doc_bytes, file_name=fname,
                        key=f"dl_{contract_key}_{fname}",
                    )
                with fc3:
                    if st.button("🗑️", key=f"del_{contract_key}_{fname}",
                                 help=f"Delete {fname}"):
                        delete_contract_doc(contract_key, fname)
                        st.rerun()
        else:
            st.caption(f"No documents attached to **{sel_contract}** yet.")


# =============================================================================
# PAGE: ⛏️ TMM Tracker
# =============================================================================
elif page == "⛏️ TMM Tracker":
    st.title("⛏️ TMM Tracker — Total Material Moved")
    st.markdown(
        "Enter **Year**, **Month**, and **Tons Moved** for each period. "
        "**Total Cost** and **Cost per Ton** are calculated automatically from SAP data. "
        "Click **💾 Save Changes** when done."
    )

    sap_loaded = st.session_state.get("sap_df") is not None
    if not sap_loaded:
        st.info(
            "💡 SAP data not loaded — Cost per Ton will show 0. "
            "Go to **SAP Sync** to upload actuals and cost columns will populate automatically."
        )

    raw_tmm   = tmm().copy()
    display_tmm = _add_tmm_calcs(raw_tmm)

    col_cfg_tmm = {
        "year":  st.column_config.NumberColumn(
            "Year", format="%.0f", min_value=2000, max_value=2100,
            width=90, help="Calendar year — e.g. 2025",
        ),
        "month": st.column_config.SelectboxColumn(
            "Month", options=MONTHS, width=130,
        ),
        "tons":  st.column_config.NumberColumn(
            "Tons Moved", format="%.0f t", min_value=0, width=150,
        ),
        # Read-only — sourced from SAP
        "total_cost":   st.column_config.NumberColumn(
            "SAP Cost ($)", format="$%.0f", disabled=True, width=140,
            help="Summed from SAP posting amounts for this Year + Month",
        ),
        "cost_per_ton": st.column_config.NumberColumn(
            "Cost / Ton ($)", format="$%.2f", disabled=True, width=140,
            help="SAP Cost ÷ Tons Moved",
        ),
    }

    edited_tmm = st.data_editor(
        display_tmm,
        column_config=col_cfg_tmm,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="tmm_editor",
        disabled=["total_cost", "cost_per_ton"],
        column_order=["year", "month", "tons", "total_cost", "cost_per_ton"],
    )

    col_s1, col_s2 = st.columns([1, 5])
    with col_s1:
        if st.button("💾 Save Changes", key="save_tmm"):
            save_cols = [c for c in TMM_COLS if c in edited_tmm.columns]
            clean_tmm = edited_tmm[save_cols].copy()
            clean_tmm["tons"] = pd.to_numeric(clean_tmm["tons"], errors="coerce").fillna(0)
            clean_tmm["year"] = pd.to_numeric(clean_tmm["year"], errors="coerce").fillna(0)
            save_tmm(clean_tmm)
            st.session_state["tmm"] = clean_tmm
            st.success(f"Saved {len(clean_tmm):,} TMM rows.")
            st.rerun()
    with col_s2:
        tot_tons = pd.to_numeric(edited_tmm["tons"], errors="coerce").sum()
        tot_cost = display_tmm["total_cost"].sum()
        avg_cpt  = (tot_cost / tot_tons) if tot_tons > 0 else 0
        sap_note = " (from SAP)" if sap_loaded else " (upload SAP to populate)"
        st.markdown(
            f"**Total Tons:** {tot_tons:,.0f} t &nbsp;|&nbsp; "
            f"**SAP Cost:** {_fmt_dollar(tot_cost)}{sap_note} &nbsp;|&nbsp; "
            f"**Avg Cost/Ton:** ${avg_cpt:,.2f}",
            unsafe_allow_html=True,
        )

    # Chart — sort by year then month order
    if not display_tmm.empty and display_tmm["tons"].sum() > 0:
        st.divider()
        st.markdown("### Cost per Ton Trend")
        plot_data = _add_tmm_calcs(tmm())
        plot_data = plot_data[pd.to_numeric(plot_data["tons"], errors="coerce") > 0].copy()
        # Sort chronologically
        plot_data["_mo"] = plot_data["month"].map(_MONTH_ORDER).fillna(0)
        plot_data = plot_data.sort_values(["year", "_mo"]).drop(columns=["_mo"])
        plot_data["period_label"] = (
            plot_data["year"].astype(int).astype(str) + " " + plot_data["month"]
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=plot_data["period_label"], y=plot_data["tons"],
            name="Tons Moved", marker_color="#1B3A5C", yaxis="y",
        ))
        fig.add_trace(go.Scatter(
            x=plot_data["period_label"], y=plot_data["cost_per_ton"],
            name="Cost / Ton ($)", mode="lines+markers",
            marker=dict(size=8, color="#C9872A"),
            line=dict(color="#C9872A", width=3),
            yaxis="y2",
        ))
        fig.update_layout(
            yaxis =dict(title="Tons Moved",     tickformat=",.0f",  showgrid=True, gridcolor="#EEE"),
            yaxis2=dict(title="Cost / Ton ($)", tickformat="$,.2f", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.05),
            height=350, margin=dict(l=0, r=10, t=20, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial", size=12),
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: 🔗 SAP Sync
# =============================================================================
elif page == "🔗 SAP Sync":
    st.title("🔗 SAP Data — Upload Monthly Actuals")
    st.markdown(
        "Upload your monthly SAP export and it will be added to the persistent database. "
        "Cost per Ton in the TMM Tracker and Amount Spent in the Contract Tracker "
        "are both calculated from this database."
    )

    st.markdown("""
<div class="tip-box">
💡 <strong>How to export from SAP (KSB1):</strong>
Enter transaction <code>KSB1</code> → set your Cost Centre(s) and Period →
Execute (F8) → List → Export → Spreadsheet → save as <code>.xlsx</code>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Section 1: Upload new monthly file ──────────────────────────────────
    st.markdown("### 1 · Upload Monthly File")

    sap_file = st.file_uploader(
        "Upload SAP export (.csv / .xlsx / .xls)",
        type=["csv", "xlsx", "xls"],
        key="sap_uploader",
    )

    if sap_file:
        file_bytes = sap_file.read()
        try:
            raw = read_file(file_bytes, sap_file.name)

            # Auto-detect format: cost-element summary vs PO/PR line-item extract
            if is_cost_element_report(raw):
                new_data, warns = clean_cost_element_report(raw)
                fmt_label = "📊 Cost Element Report"
                preview_cols = ["vendor", "gl_account", "description", "amount",
                                "commitment", "date", "fiscal_period"]
                fmt_note = (
                    "Cost element format detected — **TMM Cost/Ton will populate** from this data. "
                    "PO/PR matching (Section 3) is not available for this format."
                )
            else:
                new_data, warns = clean_sap(raw)
                fmt_label = "📋 PO/PR Line-Item Extract"
                preview_cols = ["cost_center", "po_number", "pr_number",
                                "vendor", "amount", "date", "description"]
                fmt_note = (
                    "PO/PR line-item format detected — **SAP Sync to Contract Tracker** (Section 3) "
                    "is available for this format."
                )

            st.info(f"**Format detected:** {fmt_label} — {fmt_note}")

            if warns:
                with st.expander(f"⚠️ {len(warns)} column detection warning(s)"):
                    for w in warns:
                        st.warning(w)

            if new_data.empty:
                st.error("No valid data rows found after cleaning. Check column detection warnings above.")
            else:
                # Detect periods in the uploaded file
                new_data = new_data[new_data["date"].notna()].copy()
                new_data["_yr"] = new_data["date"].dt.year
                new_data["_mo"] = new_data["date"].dt.strftime("%B")

                periods_found = (
                    new_data[["_yr", "_mo"]]
                    .drop_duplicates()
                    .sort_values(["_yr", "_mo"])
                    .values.tolist()
                )

                if not periods_found:
                    st.error(
                        "No valid posting dates found in the file. "
                        "Check that a 'Posting Date', 'Document Date', or 'Year'+'Month' column exists."
                    )
                else:
                    period_labels = [f"{mo} {int(yr)}" for yr, mo in periods_found]
                    st.success(
                        f"**{len(new_data):,} lines** detected | "
                        f"**{len(period_labels)} period(s):** {', '.join(period_labels[:6])}"
                        + (" …" if len(period_labels) > 6 else "")
                        + f" | **Total: {_fmt_dollar(new_data['amount'].sum())}**"
                    )

                    with st.expander("Preview (first 20 rows)"):
                        st.dataframe(
                            new_data[[c for c in preview_cols if c in new_data.columns]].head(20),
                            use_container_width=True, hide_index=True,
                        )

                    # Warn if these periods already exist in the DB
                    existing_db = st.session_state["sap_df"]
                    existing_summary = sap_period_summary(existing_db)
                    overlap = []
                    if not existing_summary.empty:
                        for yr, mo in periods_found:
                            if not existing_summary[
                                (existing_summary["Year"] == yr) &
                                (existing_summary["Month"] == mo)
                            ].empty:
                                overlap.append(f"{mo} {int(yr)}")
                    if overlap:
                        st.warning(
                            f"⚠️ The database already contains data for: **{', '.join(overlap[:6])}**"
                            + (" …" if len(overlap) > 6 else "")
                            + ". Confirming below will **replace** those records."
                        )

                    btn_label = (
                        f"💾 Add to SAP Database ({len(period_labels)} period(s))"
                        if len(period_labels) > 3
                        else f"💾 Add to SAP Database ({', '.join(period_labels)})"
                    )
                    if st.button(btn_label, type="primary"):
                        new_data_clean = new_data.drop(columns=["_yr", "_mo"], errors="ignore")
                        updated_db = existing_db.copy() if not existing_db.empty else pd.DataFrame()

                        for yr, mo in periods_found:
                            period_rows = new_data_clean[
                                (new_data_clean["date"].dt.year == yr) &
                                (new_data_clean["date"].dt.strftime("%B") == mo)
                            ].copy()
                            updated_db = upsert_sap_period(updated_db, period_rows, int(yr), mo)

                        save_sap_db(updated_db)
                        st.session_state["sap_df"]      = updated_db
                        st.session_state["sap_filename"] = sap_file.name
                        st.success(
                            f"✅ Database updated — "
                            f"{len(updated_db):,} total lines across all periods."
                        )
                        st.rerun()

        except Exception as exc:
            st.error(f"Could not parse file: {exc}")

    # ── Section 2: Current database ─────────────────────────────────────────
    st.divider()
    st.markdown("### 2 · Current SAP Database")

    current_db = st.session_state["sap_df"]
    if current_db is None or current_db.empty:
        st.info("No SAP data in the database yet. Upload a monthly file above.")
    else:
        summary = sap_period_summary(current_db)
        summary["Total Amount ($)"] = summary["Total Amount ($)"].apply(_fmt_dollar)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption(f"Database total: {len(current_db):,} posting lines | "
                   f"{_fmt_dollar(current_db['amount'].sum())} cumulative")

        # Delete a period
        with st.expander("🗑️ Delete a Period from the Database"):
            period_options = [
                f"{row['Month']} {row['Year']}"
                for _, row in sap_period_summary(current_db).iterrows()
            ]
            if period_options:
                del_choice = st.selectbox("Select period to delete", period_options)
                if st.button("Delete selected period", type="secondary"):
                    parts = del_choice.split(" ", 1)
                    del_mo, del_yr = parts[0], int(parts[1])
                    db_copy = current_db.copy()
                    db_copy["_yr"] = db_copy["date"].dt.year
                    db_copy["_mo"] = db_copy["date"].dt.strftime("%B")
                    db_copy = db_copy[
                        ~((db_copy["_yr"] == del_yr) & (db_copy["_mo"] == del_mo))
                    ].drop(columns=["_yr", "_mo"])
                    save_sap_db(db_copy)
                    st.session_state["sap_df"] = db_copy
                    st.success(f"Deleted {del_choice} from the database.")
                    st.rerun()

    # ── Section 3: Sync to Contract Tracker ─────────────────────────────────
    st.divider()
    st.markdown("### 3 · Sync Amount Spent → Contract Tracker")
    st.markdown(
        "Match PO/PR numbers from the SAP database against your Contract Tracker "
        "and update the **Amount Spent** column automatically."
    )

    sap_df = st.session_state.get("sap_df")
    if sap_df is None or sap_df.empty:
        st.info("Load SAP data first (Section 1 above).")
    elif st.button("🔄 Sync Actuals → Contract Tracker"):
        contracts_df = contracts().copy()
        if contracts_df.empty:
            st.warning("Contract Tracker is empty. Add contracts first.")
        else:
            sap_agg = sap_df.copy()
            sap_agg["key"] = sap_agg["po_number"].where(
                sap_agg["po_number"] != "", sap_agg["pr_number"]
            )
            sap_lookup = (
                sap_agg[sap_agg["key"] != ""]
                .groupby("key")["amount"].sum()
                .to_dict()
            )

            updated = 0
            for idx, row in contracts_df.iterrows():
                key = row["po_number"] if str(row["po_number"]).strip() else row["pr_number"]
                if str(key).strip() and key in sap_lookup:
                    contracts_df.at[idx, "amount_spent"] = round(sap_lookup[key], 2)
                    contracts_df.at[idx, "sap_synced"]   = True
                    updated += 1

            save_contracts(contracts_df)
            st.session_state["contracts"] = contracts_df
            st.success(
                f"✅ {updated} contract row(s) updated from SAP database. "
                f"{len(contracts_df) - updated} row(s) had no matching PO/PR."
            )
            st.rerun()


# =============================================================================
# PAGE: 📤 Export
# =============================================================================
elif page == "📤 Export":
    st.title("📤 Export")
    st.markdown("Download the reconciled contract matrix or raw tracker tables.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Reconciled Contract Matrix (Excel)")
        st.markdown("Full contract register with all calculated fields, formatted and ready to share.")
        if st.button("🔨 Generate Excel Export"):
            c_calc = _add_contract_calcs(contracts())
            if c_calc.empty:
                st.warning("No contract data to export.")
            else:
                # Map to expected column names for exporter
                export_input = c_calc.rename(columns={
                    "original_budget":  "baseline_value",
                    "amount_spent":     "sap_posted_amount",
                    "amount_left":      "remaining_budget",
                })
                export_input["change_orders"]          = 0.0
                export_input["true_contract_value"]    = export_input["baseline_value"]
                export_input["outstanding_obligation"] = export_input["remaining_budget"].clip(lower=0)
                export_input["total_committed"]        = export_input["sap_posted_amount"]
                export_input["phased_budget"]          = export_input["baseline_value"]
                export_input["utilisation_pct"]        = np.where(
                    export_input["baseline_value"] > 0,
                    export_input["sap_posted_amount"] / export_input["baseline_value"] * 100,
                    0.0,
                )
                xlsx_bytes = export_reconciled_matrix(export_input)
                fname = "MineDept_Contract_Matrix.xlsx"
                st.download_button("⬇️ Download Excel", data=xlsx_bytes,
                                   file_name=fname,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col2:
        st.markdown("### ⛏️ TMM Tracker (CSV)")
        st.markdown("All TMM entries with calculated Cost per Ton.")
        if st.button("🔨 Generate TMM Export"):
            t_calc = _add_tmm_calcs(tmm())
            if t_calc.empty:
                st.warning("No TMM data to export.")
            else:
                t_calc["year"] = t_calc["year"].astype(int)
                csv = t_calc.to_csv(index=False)
                st.download_button("⬇️ Download CSV", data=csv,
                                   file_name="MineDept_TMM_Tracker.csv",
                                   mime="text/csv")

    st.divider()
    st.markdown("### 💾 Backup / Restore Data")
    st.markdown(
        "Data is saved automatically to `data/contracts.json` and `data/tmm.json` "
        "in the app folder. Copy those files to back up your data."
    )
    bc1, bc2 = st.columns(2)
    with bc1:
        import json as _json
        if st.button("⬇️ Download contracts.json"):
            st.download_button(
                "Save contracts.json",
                data=_json.dumps(contracts().to_dict(orient="records"), indent=2),
                file_name="contracts.json", mime="application/json",
            )
    with bc2:
        if st.button("⬇️ Download tmm.json"):
            st.download_button(
                "Save tmm.json",
                data=_json.dumps(tmm().to_dict(orient="records"), indent=2),
                file_name="tmm.json", mime="application/json",
            )
