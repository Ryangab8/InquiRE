import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm

# ---------------------------------------------------------------------
# 1) Wide Mode + Page Title
# ---------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="US Metro Alpha/Beta Analysis")

# ---------------------------------------------------------------------
# 2) Custom CSS
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Use a default serif font site-wide */
    html, body, [class*="css"]  {
        font-family: "Times New Roman", serif;
    }

    /* Centered, black main title */
    h1 {
        text-align: center;
        color: black !important;
    }

    /* Override Streamlit's default primary color (pills, some other elements) */
    :root {
        --primary-color: #93D0EC; /* Light blue brand color */
    }

    /* Specifically style the selected "pills" in st.multiselect to our brand color */
    div[data-baseweb="tag"] {
        background-color: #93D0EC !important;
        color: black !important; /* For contrast */
        border-radius: 4px;
    }

    /* Style the Streamlit buttons to our brand color */
    div.stButton > button {
        background-color: #93D0EC !important;
        color: black !important;
        border-radius: 4px;
        border: 1px solid #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# 3) Main Title
# ---------------------------------------------------------------------
st.title("US Metro Alpha/Beta Analysis")

# ---------------------------------------------------------------------
# 4) Expanders with Descriptive Text
# ---------------------------------------------------------------------
with st.expander("About the Data", expanded=False):
    st.markdown(
        """
        **Data Source**  
        - All data is publicly available from the Bureau of Labor Statistics (BLS).  
        - This tool currently features the top 50 U.S. MSAs by population, plus a “National” benchmark.

        **Alpha**  
        - Alpha measures an MSA’s growth rate relative to the national average.  
        - A positive alpha means the MSA outperforms the nation; a negative alpha means it underperforms.  
        - For instance, an alpha of +1.5 implies the MSA grew about 1.5% faster than the national rate over the chosen period.

        **Beta**  
        - Beta measures how sensitive (or volatile) an MSA is relative to the nation’s changes.  
        - If beta = 1, the MSA moves in sync with the national trend.  
        - A beta > 1 indicates greater volatility (e.g., 1.5 → 50% larger swings), while a beta < 1 indicates less volatility (e.g., 0.5 → half as volatile).
        """
    )

with st.expander("How To Use", expanded=False):
    st.markdown(
        """
        **Step 1:** Select the desired metric from the dropdown below (currently only one option).  

        **XY Chart (Alpha vs. Beta)**  
        1. Pick MSA(s) from the dropdown, or click “Select All” to include the top 50 MSAs.  
        2. Choose a start and end date range.  
        3. Click “Generate XY Chart.” Each MSA is plotted by (β, α).  
        4. You can zoom with the scroll wheel or pinch gesture and pan by dragging.

        **Time Series (Rolling Alpha/Beta Over the Selected Date Range)**  
        1. Select up to 5 MSAs (excluding “National,” which is the benchmark).  
        2. Pick Alpha or Beta to track, plus a separate date range.  
        3. Click “Compute Time Series.” See how alpha or beta evolves month by month.  
        4. Likewise, you can zoom/pan and optionally show/hide the underlying data table.
        """
    )

# ---------------------------------------------------------------------
# 5) Global Metric Selector
# ---------------------------------------------------------------------
metric_choice = st.selectbox(
    "Select a metric:",
    ["Total NonFarm Employment - Seasonally Adjusted"]
)

# ---------------------------------------------------------------------
# 6) Load CSV (Remove DB calls)
# ---------------------------------------------------------------------
CSV_FILE = "nonfarm_data.csv"  # Or "data/nonfarm_data.csv" if in subfolder

NATIONAL_SERIES_ID = "CES0000000001"

# (Optional) Mapping of MSA IDs -> friendly names
MSA_NAME_MAP = {
    "CES0000000001": "National",
    # Add more if desired, else fallback to raw ID
    # "SMS36356200000000001": "NYC Metro",
    # "SMS06310800000000001": "LA Metro",
    # ...
}

@st.cache_data
def load_data():
    """Load CSV data with columns [date, series_id, value]."""
    df = pd.read_csv(CSV_FILE)
    # Ensure 'date' is datetime
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["series_id","date"], inplace=True)
    return df

df_raw_csv = load_data()

# ---------------------------------------------------------------------
# 7) XY Chart (Alpha/Beta)
# ---------------------------------------------------------------------
def compute_alpha_beta_XY(chosen_msas, start_ym, end_ym):
    # 1) Pivot data => columns=series_id, index=date, values=value
    df_pivot = df_raw_csv.pivot(index="date", columns="series_id", values="value")

    # 2) Filter date range
    start_date = pd.to_datetime(f"{start_ym}-01")
    end_date   = pd.to_datetime(f"{end_ym}-01")
    df_pivot = df_pivot.loc[(df_pivot.index >= start_date) & (df_pivot.index <= end_date)]

    # 3) Compute MoM % change
    df_growth = df_pivot.pct_change(1)*100
    df_growth.dropna(inplace=True)

    if NATIONAL_SERIES_ID not in df_growth.columns:
        st.error(f"National series '{NATIONAL_SERIES_ID}' not found in CSV. Check your data.")
        return pd.DataFrame()

    results = []
    for msa_id in chosen_msas:
        if msa_id == NATIONAL_SERIES_ID:
            continue
        if msa_id not in df_growth.columns:
            # Possibly not in this date range
            continue
        merged = df_growth[[NATIONAL_SERIES_ID, msa_id]].dropna()
        if merged.empty:
            continue
        X = sm.add_constant(merged[NATIONAL_SERIES_ID])
        y = merged[msa_id]
        model = sm.OLS(y, X).fit()
        alpha = model.params["const"]
        beta  = model.params[NATIONAL_SERIES_ID]
        results.append({
            "msa_id": msa_id,
            "Metro": MSA_NAME_MAP.get(msa_id, msa_id),
            "Alpha": alpha,
            "Beta": beta
        })
    return pd.DataFrame(results)

def select_all():
    unique_ids = sorted(df_raw_csv["series_id"].unique())
    st.session_state["msa_selection"] = unique_ids

def clear_all():
    st.session_state["msa_selection"] = []

if "msa_selection" not in st.session_state:
    st.session_state["msa_selection"] = []

if "df_ab" not in st.session_state:
    st.session_state["df_ab"] = None

if "fig" not in st.session_state:
    st.session_state["fig"] = None

st.markdown("### XY Chart (Alpha vs. Beta)")

# All unique IDs in CSV
all_unique_msas = sorted(df_raw_csv["series_id"].unique())

st.multiselect(
    "Pick MSA(s):",
    options=all_unique_msas,
    key="msa_selection"
)

col1, col2 = st.columns(2)
col1.button("Select All", on_click=select_all)
col2.button("Clear", on_click=clear_all)

months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
years_xy = list(range(1990, 2025))
default_start_year = 2019
default_end_year   = 2024

# Locate the indexes for default years
default_start_index = years_xy.index(default_start_year)
default_end_index   = years_xy.index(default_end_year)

st.write("#### Date Range for XY Chart")
col_s1, col_s2 = st.columns(2)
with col_s1:
    xy_start_month = st.selectbox("Start Month (XY)", months, index=0)
    xy_start_year = st.selectbox(
        "Start Year (XY)",
        years_xy,
        index=default_start_index
    )
with col_s2:
    xy_end_month = st.selectbox("End Month (XY)", months, index=11)
    xy_end_year  = st.selectbox(
        "End Year (XY)",
        years_xy,
        index=default_end_index
    )

xy_smonth_num = months.index(xy_start_month) + 1
xy_emonth_num = months.index(xy_end_month) + 1
xy_start_ym = f"{xy_start_year:04d}-{xy_smonth_num:02d}"
xy_end_ym   = f"{xy_end_year:04d}-{xy_emonth_num:02d}"

if st.button("Generate XY Chart"):
    if not st.session_state["msa_selection"]:
        st.warning("No MSAs selected!")
    else:
        df_ab = compute_alpha_beta_XY(st.session_state["msa_selection"], xy_start_ym, xy_end_ym)
        if df_ab.empty:
            st.error("No alpha/beta results. Possibly no overlapping data with national.")
        else:
            fig_xy = px.scatter(
                df_ab,
                x="Beta",
                y="Alpha",
                text="Metro",
                title=f"Alpha vs. Beta ({xy_start_ym} to {xy_end_ym})",
                labels={"Beta": "Beta", "Alpha": "Alpha"}
            )
            fig_xy.update_traces(textposition='top center', textfont_size=14)
            fig_xy.update_layout(dragmode='pan', title_x=0.5, title_xanchor='center')

            st.session_state["df_ab"] = df_ab
            st.session_state["fig"]   = fig_xy

if st.session_state["fig"] is not None:
    st.plotly_chart(
        st.session_state["fig"],
        use_container_width=True,
        config={"scrollZoom": True}
    )
    if st.checkbox("View alpha/beta table for XY chart"):
        st.dataframe(st.session_state["df_ab"])

# ---------------------------------------------------------------------
# 8) TIME SERIES
# ---------------------------------------------------------------------
st.markdown("### Time Series (Rolling Alpha/Beta)")

if "df_ts" not in st.session_state:
    st.session_state["df_ts"] = None

if "fig_ts" not in st.session_state:
    st.session_state["fig_ts"] = None

all_msas_no_national = [m for m in all_unique_msas if m != NATIONAL_SERIES_ID]

selected_time_msas = st.multiselect(
    "Pick up to 5 MSAs for Time Series:",
    options=all_msas_no_national,
    max_selections=5
)

ab_choice = st.selectbox("Which metric to graph in the Time Series?", ["alpha", "beta"])

ts_default_start_index = years_xy.index(2019)
ts_default_end_index   = years_xy.index(2024)

st.write("#### Date Range for Time Series")
col_t1, col_t2 = st.columns(2)
with col_t1:
    ts_start_month = st.selectbox("Start Month (Time Series)", months, index=0)
    ts_start_year = st.selectbox(
        "Start Year (Time Series)",
        years_xy,
        index=ts_default_start_index
    )
with col_t2:
    ts_end_month = st.selectbox("End Month (Time Series)", months, index=11)
    ts_end_year  = st.selectbox(
        "End Year (Time Series)",
        years_xy,
        index=ts_default_end_index
    )

ts_smonth_num = months.index(ts_start_month) + 1
ts_emonth_num = months.index(ts_end_month) + 1
ts_start_ym = f"{ts_start_year:04d}-{ts_smonth_num:02d}"
ts_end_ym   = f"{ts_end_year:04d}-{ts_emonth_num:02d}"

def compute_alpha_beta_subset(df_window, nat_col, msa_col):
    """OLS of MSA vs. National in that rolling subset."""
    if len(df_window.dropna()) < 2:
        return None, None
    X = sm.add_constant(df_window[nat_col])
    y = df_window[msa_col]
    model = sm.OLS(y, X).fit()
    alpha = model.params["const"]
    beta  = model.params[nat_col]
    return alpha, beta

def compute_time_series_rolling_alpha_beta(msa_ids, start_ym_ts, end_ym_ts):
    """Compute rolling alpha/beta for each monthly date up to that month."""
    df_pivot = df_raw_csv.pivot(index="date", columns="series_id", values="value")
    # Filter date range
    start_dt = pd.to_datetime(f"{start_ym_ts}-01")
    end_dt   = pd.to_datetime(f"{end_ym_ts}-01")
    df_pivot = df_pivot.loc[(df_pivot.index >= start_dt) & (df_pivot.index <= end_dt)]

    # MoM growth
    df_growth = df_pivot.pct_change(1)*100
    df_growth.dropna(inplace=True)
    if NATIONAL_SERIES_ID not in df_growth.columns:
        return pd.DataFrame()

    unique_months = sorted(df_growth.index.unique())
    results = []

    for current_month in unique_months:
        # subset from earliest date up to current_month
        df_window = df_growth.loc[df_growth.index <= current_month]
        for msa in msa_ids:
            if msa == NATIONAL_SERIES_ID or msa not in df_growth.columns:
                continue
            alpha_val, beta_val = compute_alpha_beta_subset(df_window[[NATIONAL_SERIES_ID, msa]], NATIONAL_SERIES_ID, msa)
            if alpha_val is not None and beta_val is not None:
                results.append({
                    "Date": current_month,
                    "Metro": MSA_NAME_MAP.get(msa, msa),
                    "alpha": alpha_val,
                    "beta": beta_val
                })
    return pd.DataFrame(results)

if st.button("Compute Time Series"):
    if not selected_time_msas:
        st.warning("Pick at least 1 MSA (up to 5).")
    else:
        df_ts_result = compute_time_series_rolling_alpha_beta(selected_time_msas, ts_start_ym, ts_end_ym)
        if df_ts_result.empty:
            st.warning("Could not compute time-series alpha/beta. Possibly insufficient overlapping data.")
        else:
            # Reorder columns: Date, Metro, alpha, beta
            df_ts_result = df_ts_result[["Date","Metro","alpha","beta"]]
            st.session_state["df_ts"] = df_ts_result

            # For plotting, pick the user-chosen metric (alpha or beta)
            df_plot = df_ts_result.copy()
            df_plot["AB_Chosen"] = df_plot[ab_choice]

            title_ts = f"Time Series of {ab_choice.title()} ({ts_start_ym} to {ts_end_ym})"
            fig_ts = px.line(
                df_plot,
                x="Date",
                y="AB_Chosen",
                color="Metro",
                title=title_ts
            )
            fig_ts.update_layout(
                dragmode='pan',
                xaxis_title="Date",
                yaxis_title=ab_choice.title(),
                title_x=0.5,
                title_xanchor="center"
            )
            st.session_state["fig_ts"] = fig_ts

if st.session_state["fig_ts"] is not None:
    st.plotly_chart(
        st.session_state["fig_ts"],
        use_container_width=True,
        config={"scrollZoom": True}
    )
    if st.checkbox("Show time-series data table"):
        st.dataframe(st.session_state["df_ts"])
