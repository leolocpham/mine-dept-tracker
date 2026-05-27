"""
MineDept Cost & Contract Tracker
=================================
Production-ready Streamlit application for reconciling SAP financial actuals
against an Operational Contract Tracker, computing financial KPIs, and
surfacing budget risk through an interactive executive dashboard.

Run:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local modules are importable
sys.path.insert(0, str(Path(__file__).parent))

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.cleaner    import read_file, clean_sap, clean_ops
from engine.reconciler import reconcile
from engine.metrics    import (
    compute_metrics, compute_summary, compute_subdept_summary,
    get_stuck_prs, compute_burn_flags,
)
from utils.exporter import export_reconciled_matrix

# ===========================================================================
# Page configuration
# ===========================================================================
st.set_page_config(
    page_title="MineDept Cost & Contract Tracker",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# Brand CSS
# ===========================================================================
st.markdown("""
<style>
:root {
    --navy:  #1B3A5C;
    --amber: #C9872A;
    --light: #F4F6F9;
}
[data-testid="stSidebar"]   { background-color: var(--navy); }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
h1, h2, h3 { font-family: Arial, sans-serif; color: var(--navy); }
.stButton > button {
    background-color: var(--amber);
    color: white; border: none;
    font-family: Arial, sans-serif; font-weight: bold;
}
.stButton > button:hover { background-color: #a56e1c; }
div[data-testid="stMetric"] {
    background: white;
    border-radius: 8px;
    padding: 12px 16px;
    border-left: 4px solid var(--amber);
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.flag-red    { color: #C62828; font-weight: bold; }
.flag-yellow { color: #F57C00; font-weight: bold; }
.flag-green  { color: #2E7D32; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# Session state initialisation
# ===========================================================================
_DEFAULTS: dict = {
    # Raw uploaded bytes (so data survives tab switches)
    "sap_bytes":        None,
    "ops_bytes":        None,
    "sap_filename":     "",
    "ops_filename":     "",
    # Cleaned frames
    "sap_clean":        None,
    "ops_clean":        None,
    # Reconciled + metrics frame (with editable notes column)
    "reconciled":       None,
    # Warning log
    "mismatch_log":     None,
    # Processing warnings from column detection
    "parse_warnings":   [],
    # Flag: pipeline has run at least once
    "data_loaded":      False,
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ===========================================================================
# Data-pipeline helper
# ===========================================================================

def run_pipeline() -> None:
    """
    Execute the full ETL → reconcile → metrics pipeline using whatever
    bytes are stored in session state.  Results are written back to session
    state so they survive page navigation.
    """
    warnings: list[str] = []

    # -- SAP --
    sap_raw = read_file(st.session_state["sap_bytes"], st.session_state["sap_filename"])
    sap_clean, sap_warns = clean_sap(sap_raw)
    warnings.extend(sap_warns)

    # -- Ops Tracker --
    ops_raw = read_file(st.session_state["ops_bytes"], st.session_state["ops_filename"])
    ops_clean, ops_warns = clean_ops(ops_raw)
    warnings.extend(ops_warns)

    # -- Reconcile --
    merged, mismatch_log = reconcile(sap_clean, ops_clean)

    # -- Metrics --
    final = compute_metrics(merged)

    # Add notes column if not already present
    if "notes" not in final.columns:
        final["notes"] = ""

    # -- Persist --
    st.session_state["sap_clean"]      = sap_clean
    st.session_state["ops_clean"]      = ops_clean
    st.session_state["reconciled"]     = final
    st.session_state["mismatch_log"]   = mismatch_log
    st.session_state["parse_warnings"] = warnings
    st.session_state["data_loaded"]    = True


# ===========================================================================
# Sidebar
# ===========================================================================

with st.sidebar:
    st.markdown(
        "<h2 style='color:white; font-family:Arial; margin-bottom:2px;'>⛏️ MineDept Tracker</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#C9872A; font-size:12px; font-family:Arial; margin-top:0;'>"
        "Cost &amp; Contract Reconciliation</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ---- SAP upload ----
    st.markdown("**1 · SAP Financial Actuals**")
    sap_file = st.file_uploader(
        "Upload SAP export (.csv / .xlsx / .xls)",
        type=["csv", "xlsx", "xls"],
        key="sap_uploader",
        label_visibility="collapsed",
    )
    if sap_file and sap_file.name != st.session_state["sap_filename"]:
        st.session_state["sap_bytes"]    = sap_file.read()
        st.session_state["sap_filename"] = sap_file.name

    if st.session_state["sap_filename"]:
        st.caption(f"✅ {st.session_state['sap_filename']}")
    else:
        st.caption("No file uploaded")

    st.markdown("**2 · Operational Contract Tracker**")
    ops_file = st.file_uploader(
        "Upload Ops Tracker (.xlsx / .xls)",
        type=["xlsx", "xls"],
        key="ops_uploader",
        label_visibility="collapsed",
    )
    if ops_file and ops_file.name != st.session_state["ops_filename"]:
        st.session_state["ops_bytes"]    = ops_file.read()
        st.session_state["ops_filename"] = ops_file.name

    if st.session_state["ops_filename"]:
        st.caption(f"✅ {st.session_state['ops_filename']}")
    else:
        st.caption("No file uploaded")

    st.divider()

    # ---- Process button ----
    both_ready = (
        st.session_state["sap_bytes"] is not None
        and st.session_state["ops_bytes"] is not None
    )
    if st.button("⚙️ Process & Reconcile", disabled=not both_ready):
        with st.spinner("Running reconciliation pipeline…"):
            try:
                run_pipeline()
                st.success("Pipeline complete.")
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")

    # ---- Demo data ----
    st.divider()
    st.markdown("**Demo / Test Data**")
    if st.button("🧪 Load Sample Dataset"):
        from utils.demo import (
            generate_ops_tracker, generate_sap_actuals,
            ops_to_excel_bytes, sap_to_excel_bytes,
        )
        ops_demo = generate_ops_tracker()
        sap_demo = generate_sap_actuals(ops_demo)

        st.session_state["ops_bytes"]    = ops_to_excel_bytes(ops_demo)
        st.session_state["ops_filename"] = "demo_ops_tracker.xlsx"
        st.session_state["sap_bytes"]    = sap_to_excel_bytes(sap_demo)
        st.session_state["sap_filename"] = "demo_sap_actuals.xlsx"

        with st.spinner("Running reconciliation pipeline…"):
            run_pipeline()
        st.success("Sample data loaded.")
        st.rerun()

    # ---- Status panel ----
    if st.session_state["data_loaded"]:
        df = st.session_state["reconciled"]
        st.divider()
        st.markdown(f"**Contracts:** {len(df):,}")
        st.markdown(f"**Vendors:** {df['vendor'].nunique():,}")
        st.markdown(f"**Cost Centres:** {df['cost_center'].nunique():,}")
        mm = st.session_state["mismatch_log"]
        if mm is not None and not mm.empty:
            st.markdown(f"⚠️ **Mismatches:** {len(mm):,}")


# ===========================================================================
# Main content
# ===========================================================================

st.markdown(
    "<h1 style='margin-bottom:4px;'>⛏️ MineDept Cost &amp; Contract Tracker</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#666; font-size:13px; margin-top:0;'>"
    "SAP Actuals · Contract Obligations · Budget Burn · Reconciliation</p>",
    unsafe_allow_html=True,
)

# ---- No data state ----
if not st.session_state["data_loaded"]:
    st.info(
        "👈 Upload your **SAP Financial Actuals** and **Ops Tracker** files "
        "in the sidebar, then click **Process & Reconcile** — or use "
        "**Load Sample Dataset** to explore immediately."
    )
    st.markdown("""
### Expected File Formats

| File | Required columns (flexible naming) |
|---|---|
| **SAP Actuals** | Cost Center · PO Number · PR Number · Vendor · Amount · Posting Date |
| **Ops Tracker** | Cost Center · PO/PR Number · Vendor · Project Task · Sub-Department · Baseline Value · Change Orders · Phased Budget · PR Date |

Column names are matched flexibly — common SAP transaction variants (KSB1, ME2N, ZFICO) are recognised automatically.
""")
    st.stop()


# ---- Parse warnings ----
warnings = st.session_state.get("parse_warnings", [])
if warnings:
    with st.expander(f"⚠️ Column Detection Warnings ({len(warnings)})", expanded=False):
        for w in warnings:
            st.warning(w)

# ---- Data integrity / mismatch log ----
mm_log = st.session_state["mismatch_log"]
if mm_log is not None and not mm_log.empty:
    with st.expander(
        f"🔍 Data Integrity / Mismatch Warnings — {len(mm_log)} row(s) flagged",
        expanded=False,
    ):
        st.markdown(
            "These rows could not be cleanly matched between SAP and the Ops Tracker. "
            "Review and correct source files as needed."
        )
        st.dataframe(mm_log, use_container_width=True, hide_index=True)


df  = st.session_state["reconciled"]
kpi = compute_summary(df)

# ===========================================================================
# Tabs
# ===========================================================================
tab_exec, tab_ops = st.tabs(["📊 Executive Summary", "📋 Operational Bullpen"])


# ---------------------------------------------------------------------------
# TAB 1 — Executive Summary
# ---------------------------------------------------------------------------
with tab_exec:

    # ---- KPI cards (row 1) ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Mine Dept Budget",  f"${kpi['total_phased_budget']:,.0f}")
    c2.metric("Total Posted Actuals",    f"${kpi['total_posted_actuals']:,.0f}",
              delta=f"{kpi['total_posted_actuals']/kpi['total_phased_budget']*100:.1f}% of budget"
              if kpi["total_phased_budget"] else None)
    c3.metric("Open Contract Commitments", f"${kpi['total_open_commitments']:,.0f}")
    c4.metric("Remaining Free Cash",
              f"${kpi['remaining_free_cash']:,.0f}",
              delta=f"{kpi['remaining_free_cash']/kpi['total_phased_budget']*100:.1f}%"
              if kpi["total_phased_budget"] else None,
              delta_color="normal" if kpi["remaining_free_cash"] >= 0 else "inverse")

    # ---- KPI cards (row 2) ----
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("True Contract Value",     f"${kpi['total_true_contract_value']:,.0f}")
    c6.metric("Total Approved FCOs",     f"${kpi['total_change_orders']:,.0f}")
    c7.metric("Active Contracts",        f"{kpi['contract_count']:,}")
    c8.metric("Unique Vendors",          f"{kpi['vendor_count']:,}")

    st.divider()

    # ---- Budget vs Actuals vs Commitments — horizontal bar chart ----
    subdept = compute_subdept_summary(df)
    st.markdown("### Budget vs Actuals vs Commitments by Sub-Department")

    if subdept.empty:
        st.info("Sub-department breakdown unavailable — add a 'Sub-Department' column to your Ops Tracker.")
    else:
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Phased Budget",
            y=subdept["sub_dept"],
            x=subdept["phased_budget"],
            orientation="h",
            marker_color="#1B3A5C",
            opacity=0.85,
            text=subdept["phased_budget"].apply(lambda v: f"${v:,.0f}"),
            textposition="inside",
            insidetextanchor="middle",
        ))
        fig.add_trace(go.Bar(
            name="Posted Actuals",
            y=subdept["sub_dept"],
            x=subdept["posted_actuals"],
            orientation="h",
            marker_color="#C9872A",
            text=subdept["posted_actuals"].apply(lambda v: f"${v:,.0f}"),
            textposition="inside",
            insidetextanchor="middle",
        ))
        fig.add_trace(go.Bar(
            name="Open Commitments",
            y=subdept["sub_dept"],
            x=subdept["open_commitments"],
            orientation="h",
            marker_color="#4CAF50",
            text=subdept["open_commitments"].apply(lambda v: f"${v:,.0f}"),
            textposition="inside",
            insidetextanchor="middle",
        ))

        fig.update_layout(
            barmode="group",
            height=max(280, len(subdept) * 90),
            margin=dict(l=0, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(
                tickprefix="$",
                tickformat=",.0f",
                showgrid=True,
                gridcolor="#EEEEEE",
            ),
            yaxis=dict(automargin=True),
            font=dict(family="Arial", size=12),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---- Budget burn flags ----
    col_burn, col_stuck = st.columns([1, 1])

    with col_burn:
        st.markdown("### 🚦 Budget Burn Rate by Cost Centre")
        burn = compute_burn_flags(df)
        if burn.empty:
            st.info("No cost-centre data available.")
        else:
            # Colour-mapped dataframe display
            def _colour_flag(val: str) -> str:
                if "Red" in val:
                    return "color: #C62828; font-weight: bold;"
                if "Yellow" in val:
                    return "color: #F57C00; font-weight: bold;"
                return "color: #2E7D32; font-weight: bold;"

            display_burn = burn.rename(columns={
                "cost_center":    "Cost Center",
                "phased_budget":  "Budget",
                "total_committed":"Committed",
                "posted_actuals": "Posted",
                "remaining_budget":"Remaining",
                "burn_rate_pct":  "Burn %",
                "status":         "Status",
            })
            # Format money cols
            for col in ("Budget", "Committed", "Posted", "Remaining"):
                if col in display_burn.columns:
                    display_burn[col] = display_burn[col].apply(lambda v: f"${v:,.0f}")
            if "Burn %" in display_burn.columns:
                display_burn["Burn %"] = display_burn["Burn %"].apply(lambda v: f"{v:.1f}%")

            st.dataframe(
                display_burn.style.applymap(_colour_flag, subset=["Status"]),
                use_container_width=True,
                hide_index=True,
            )

    # ---- Stuck PR warning list ----
    with col_stuck:
        st.markdown("### ⏰ Stuck PR Warning List (> 14 days)")
        stuck = get_stuck_prs(df, days=14)
        if stuck.empty:
            st.success("No stuck PRs detected. All requisitions are within the 14-day SLA.")
        else:
            st.warning(f"{len(stuck)} purchase requisition(s) have exceeded 14 days without a PO.")

            def _highlight_days(val):
                try:
                    d = int(val)
                    if d >= 30:
                        return "color: #C62828; font-weight: bold;"
                    if d >= 14:
                        return "color: #F57C00;"
                except (TypeError, ValueError):
                    pass
                return ""

            display_stuck = stuck.rename(columns={
                "pr_number":     "PR Number",
                "cost_center":   "Cost Center",
                "sub_dept":      "Sub-Dept",
                "vendor":        "Vendor",
                "task":          "Task",
                "baseline_value":"Value",
                "pr_date":       "PR Date",
                "days_open":     "Days Open",
            })
            if "Value" in display_stuck.columns:
                display_stuck["Value"] = display_stuck["Value"].apply(lambda v: f"${v:,.0f}")
            if "PR Date" in display_stuck.columns:
                display_stuck["PR Date"] = pd.to_datetime(
                    display_stuck["PR Date"], errors="coerce"
                ).dt.strftime("%d %b %Y")

            st.dataframe(
                display_stuck.style.applymap(_highlight_days, subset=["Days Open"]),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# TAB 2 — Operational Bullpen
# ---------------------------------------------------------------------------
with tab_ops:

    st.markdown("### Operational Bullpen — Contract-by-Contract View")

    # ---- Cost Centre filter ----
    all_ccs = sorted(df["cost_center"].dropna().unique().tolist())
    all_ccs = [cc for cc in all_ccs if cc.strip()]

    col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 4])
    with col_filter1:
        selected_cc = st.selectbox(
            "Filter by Cost Centre",
            options=["All Cost Centres"] + all_ccs,
            key="cc_filter",
        )
    with col_filter2:
        # Sub-dept filter
        all_depts = sorted(df["sub_dept"].dropna().unique().tolist())
        all_depts = [d for d in all_depts if d.strip()]
        selected_dept = st.selectbox(
            "Filter by Sub-Department",
            options=["All Sub-Departments"] + all_depts,
            key="dept_filter",
        )
    with col_filter3:
        vendor_search = st.text_input("🔍 Search Vendor / Task", key="vendor_search")

    # Apply filters
    view = df.copy()
    if selected_cc != "All Cost Centres":
        view = view[view["cost_center"] == selected_cc]
    if selected_dept != "All Sub-Departments":
        view = view[view["sub_dept"] == selected_dept]
    if vendor_search:
        mask = (
            view["vendor"].str.contains(vendor_search, case=False, na=False) |
            view["task"].str.contains(vendor_search, case=False, na=False)
        )
        view = view[mask]

    st.caption(f"Showing {len(view):,} of {len(df):,} contract rows")

    # ---- Display columns ----
    DISPLAY_COLS = {
        "cost_center":          "Cost Center",
        "sub_dept":             "Sub-Dept",
        "vendor":               "Vendor",
        "task":                 "Project Task",
        "po_number":            "PO",
        "pr_number":            "PR",
        "baseline_value":       "Baseline",
        "change_orders":        "FCOs",
        "true_contract_value":  "Contract Value",
        "sap_posted_amount":    "Posted Actuals",
        "outstanding_obligation": "Outstanding",
        "remaining_budget":     "Remaining Budget",
        "utilisation_pct":      "Util %",
        "notes":                "Operational Notes",
    }
    cols_present = [c for c in DISPLAY_COLS if c in view.columns]
    view_display = view[cols_present].rename(columns=DISPLAY_COLS).reset_index(drop=True)

    # ---- Editable data table with notes ----
    st.markdown("**Click any cell in _Operational Notes_ to add or edit justification notes.**")

    # Column config: all read-only except Notes
    money_display = ["Baseline", "FCOs", "Contract Value", "Posted Actuals",
                     "Outstanding", "Remaining Budget"]
    col_config = {
        "Operational Notes": st.column_config.TextColumn(
            "Operational Notes", width="large",
            help="Type justification, status updates, or action items here.",
        ),
        "Util %": st.column_config.ProgressColumn(
            "Util %", min_value=0, max_value=100, format="%.1f%%",
        ),
    }
    for mc in money_display:
        if mc in view_display.columns:
            col_config[mc] = st.column_config.NumberColumn(mc, format="$%.0f")

    # Disabled columns = everything except Notes
    editable_col = "Operational Notes"
    disabled_cols = [c for c in view_display.columns if c != editable_col]

    edited = st.data_editor(
        view_display,
        column_config=col_config,
        disabled=disabled_cols,
        use_container_width=True,
        hide_index=True,
        key="bullpen_editor",
        num_rows="fixed",
    )

    # Write notes back into the master reconciled dataframe
    if "Operational Notes" in edited.columns and "notes" in df.columns:
        note_map = edited["Operational Notes"].to_dict()
        # Map filtered view indices back to master df indices
        view_indices = view.index.tolist()
        for local_idx, master_idx in enumerate(view_indices):
            if local_idx in note_map:
                st.session_state["reconciled"].at[master_idx, "notes"] = note_map[local_idx]

    st.divider()

    # ---- Mini bar chart for selected cost centre ----
    if selected_cc != "All Cost Centres" and not view.empty:
        st.markdown(f"### 📊 Spend Profile — {selected_cc}")
        vend_grp = (
            view.groupby("vendor", as_index=False)
            .agg(posted=("sap_posted_amount", "sum"), committed=("outstanding_obligation", "sum"))
            .sort_values("posted", ascending=True)
            .tail(10)
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Posted Actuals",
            y=vend_grp["vendor"],
            x=vend_grp["posted"],
            orientation="h",
            marker_color="#C9872A",
        ))
        fig2.add_trace(go.Bar(
            name="Outstanding Commitment",
            y=vend_grp["vendor"],
            x=vend_grp["committed"],
            orientation="h",
            marker_color="#1B3A5C",
        ))
        fig2.update_layout(
            barmode="stack",
            height=max(250, len(vend_grp) * 45),
            margin=dict(l=0, r=20, t=10, b=20),
            xaxis=dict(tickprefix="$", tickformat=",.0f"),
            legend=dict(orientation="h", y=1.05),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Arial", size=11),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ---- Export button ----
    st.markdown("### 📥 Export Reconciled Matrix")
    col_exp1, col_exp2 = st.columns([1, 3])
    with col_exp1:
        if st.button("🔨 Generate Export"):
            export_df = st.session_state["reconciled"]
            with st.spinner("Building Excel workbook…"):
                xlsx_bytes = export_reconciled_matrix(export_df)
            fname = (
                "MineDept_Reconciled_Matrix.xlsx"
            )
            st.download_button(
                label="⬇️ Download Excel",
                data=xlsx_bytes,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    with col_exp2:
        st.caption(
            "Downloads the full reconciled dataset including all KPIs, "
            "outstanding commitments, and any operational notes entered above."
        )
