import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import datetime

# ---------------------------------------------------------------------
# 1) Wide Mode + Page Title
# ---------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="US Metro Alpha/Beta Analysis")

# ---------------------------------------------------------------------
# 2) Custom CSS
# ---------------------------------------------------------------------
CUSTOM_CSS = """
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
    color: black !important;
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
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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

All data is publicly available from the Bureau of Labor Statistics (BLS).
This tool currently features the top 50 U.S. MSAs by population, plus a “National” benchmark.

**Alpha**

Alpha measures an MSA’s growth rate relative to the national average.
A positive alpha means the MSA outperforms the nation; a negative alpha means it underperforms.
For instance, an alpha of +1.5 implies the MSA grew about 1.5% faster than the national rate over the chosen period.

**Beta**

Beta measures how sensitive (or volatile) an MSA is relative to the nation’s changes.
If beta = 1, the MSA moves in sync with the national trend.
A beta > 1 indicates greater volatility (e.g., 1.5 → 50% larger swings), while a beta < 1 indicates less volatility (e.g., 0.5 → half as volatile).
        """
    )

with st.expander("How To Use", expanded=False):
    st.markdown(
        """
**Step 1:** Select the desired metric from the dropdown (currently only one).  

**XY Chart (Alpha vs. Beta)**  
1. Pick MSA(s) from the dropdown, or click “Select All.”  
2. Choose a start and end date range.  
3. Click “Generate XY Chart.” Each MSA is plotted by (β, α).  

**Time Series (Rolling Alpha/Beta)**  
1. Select up to 5 MSAs (excluding “National,” which is the benchmark).  
2. Pick Alpha or Beta to track, plus a separate date range.  
3. Click “Compute Time Series.” See how alpha or beta evolves month by month.  
4. Optionally view the underlying data table.
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
# 6) Load CSV, Parse Dates, Check for Duplicates
# ---------------------------------------------------------------------
st.write("Loading CSV: data/raw_nonfarm_jobs.csv ...")

# IMPORTANT: parse dates with format='%m/%d/%Y' to match e.g. "12/1/2009"
df_full = pd.read_csv(
    "data/raw_nonfarm_jobs.csv",
    dtype={"series_id": str, "value": float},  # optional type hints
)
df_full["obs_date"] = pd.to_datetime(
    df_full["obs_date"],
    format="%m/%d/%Y",   # matches "12/1/2009"
    errors="coerce"
)

# Check for NaT after parsing
if df_full["obs_date"].isna().any():
    st.warning("Some rows have invalid obs_date after parsing (NaT). Check CSV format or bad rows.")
    bad_date_rows = df_full[df_full["obs_date"].isna()]
    st.write("Rows with bad dates:", bad_date_rows.head(20))

# Check for duplicates of (series_id, obs_date)
duplicated_rows = df_full[df_full.duplicated(subset=["series_id", "obs_date"], keep=False)]
if not duplicated_rows.empty:
    st.warning("WARNING: Found duplicates for (series_id, obs_date). May cause pivot collisions.")
    st.write(duplicated_rows.head(30))

# The "National" series ID
NATIONAL_SERIES_ID = "CES0000000001"

MSA_NAME_MAP = {
    "CES0000000001": "National",
    "SMS36356200000000001": "NYC Metro",
    "SMS06310800000000001": "LA Metro",
    "SMS13120600000000001": "Atlanta",
    "SMS17169800000000001": "Chicago",
    "SMS04380600000000001": "Phoenix",
    "SMS48191000000000001": "Dallas–Fort Worth",
    "SMS25716540000000001": "Boston",
    "SMS48264200000000001": "Houston",
    "SMS06418840000000001": "San Francisco–Oakland",
    "SMS06401400000000001": "Riverside–San Bernardino",
    "SMS11479000000000001": "Washington DC",
    "SMS26198200000000001": "Detroit",
    "SMS42979610000000001": "Philadelphia",
    "SMS53426600000000001": "Seattle",
    "SMS12331000000000001": "Miami",
    "SMS27334600000000001": "Minneapolis–St. Paul",
    "SMS06417400000000001": "San Diego",
    "SMS12453000000000001": "Tampa",
    "SMS08197400000000001": "Denver",
    "SMS29411800000000001": "St. Louis",
    "SMS24925810000000001": "Baltimore",
    "SMS37167400000000001": "Charlotte",
    "SMS12367400000000001": "Orlando",
    "SMS48417000000000001": "San Antonio",
    "SMS41389000000000001": "Portland",
    "SMS42383000000000001": "Pittsburgh",
    "SMS06409000000000001": "Sacramento",
    "SMS32298200000000001": "Las Vegas",
    "SMS39171400000000001": "Cincinnati",
    "SMS20928120000000001": "Kansas City",
    "SMS18180200000000001": "Columbus",
    "SMS18269000000000001": "Indianapolis",
    "SMS39174600000000001": "Cleveland",
    "SMS06419400000000001": "San Jose",
    "SMS47349800000000001": "Nashville",
    "SMS51472600000000001": "Virginia Beach–Norfolk",
    "SMS44772000000000001": "Providence–Warwick",
    "SMS55333400000000001": "Milwaukee",
    "SMS12272600000000001": "Jacksonville",
    "SMS47328200000000001": "Memphis",
    "SMS51400600000000001": "Richmond",
    "SMS40364200000000001": "Oklahoma City",
    "SMU04380600000000001": "Hartford",
    "SMS22353800000000001": "New Orleans",
    "SMS36153800000000001": "Buffalo–Cheektowaga",
    "SMS37395800000000001": "Raleigh",
    "SMS01138200000000001": "Birmingham–Hoover",
    "SMS49416200000000001": "Salt Lake City",
    "SMS36403800000000001": "Rochester",
    "SMS21311400000000001": "Louisville"
}
INVERTED_MAP = {v: k for k, v in MSA_NAME_MAP.items()}

# ---------------------------------------------------------------------
# Shared Lists for Month/Year
# ---------------------------------------------------------------------
months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
years_xy = list(range(1990, 2030))

# ---------------------------------------------------------------------
# fetch_raw_data_multiple
# ---------------------------------------------------------------------
def fetch_raw_data_multiple(msa_ids, start_ym, end_ym):
    """
    Filter df_full for the chosen MSA IDs and date range.
    Returns subset with obs_date, value, etc.
    """
    start_year, start_month = map(int, start_ym.split("-"))
    end_year, end_month     = map(int, end_ym.split("-"))
    start_dt = datetime.datetime(start_year, start_month, 1)
    end_dt   = datetime.datetime(end_year, end_month, 1)

    df_filtered = df_full[
        (df_full["series_id"].isin(msa_ids)) &
        (df_full["obs_date"] >= start_dt) &
        (df_full["obs_date"] <= end_dt)
    ].copy()
    return df_filtered

# ---------------------------------------------------------------------
# XY Chart: compute_multi_alpha_beta
# ---------------------------------------------------------------------
def compute_multi_alpha_beta(df_raw):
    """
    Pivot monthly data -> monthly growth -> regress vs. National.
    """
    df_raw["value"] = pd.to_numeric(df_raw["value"], errors="coerce")
    df_pivot = df_raw.pivot(index="obs_date", columns="series_id", values="value")

    # monthly % growth
    df_growth = df_pivot.pct_change(1) * 100
    df_growth.dropna(inplace=True)

    if NATIONAL_SERIES_ID not in df_growth.columns:
        return pd.DataFrame()

    results = []
    for col in df_growth.columns:
        if col == NATIONAL_SERIES_ID:
            continue
        merged = df_growth[[NATIONAL_SERIES_ID, col]].dropna()
        if len(merged) < 2:
            continue
        X = sm.add_constant(merged[NATIONAL_SERIES_ID])
        y = merged[col]
        model = sm.OLS(y, X).fit()
        alpha_val = model.params["const"]
        beta_val  = model.params[NATIONAL_SERIES_ID]
        r_sq      = model.rsquared

        results.append({
            "series_id": col,
            "Alpha": alpha_val,
            "Beta": beta_val,
            "R-Squared": r_sq
        })

    df_ab = pd.DataFrame(results)
    if df_ab.empty:
        return df_ab

    df_ab["Metro"] = df_ab["series_id"].apply(lambda sid: MSA_NAME_MAP.get(sid, sid))
    df_ab.drop(columns=["series_id"], inplace=True)
    return df_ab[["Metro", "Alpha", "Beta", "R-Squared"]]

# ---------------------------------------------------------------------
# XY Chart UI
# ---------------------------------------------------------------------
def select_all():
    st.session_state["msa_selection"] = sorted(INVERTED_MAP.keys())

def clear_all():
    st.session_state["msa_selection"] = []

if "msa_selection" not in st.session_state:
    st.session_state["msa_selection"] = []
if "df_ab" not in st.session_state:
    st.session_state["df_ab"] = None
if "fig_xy" not in st.session_state:
    st.session_state["fig_xy"] = None

st.markdown("### XY Chart (Alpha vs. Beta)")

all_msa_names = sorted(INVERTED_MAP.keys())
st.multiselect("Pick MSA(s):", options=all_msa_names, key="msa_selection")

col1, col2 = st.columns(2)
col1.button("Select All", on_click=select_all)
col2.button("Clear", on_click=clear_all)

default_start_year = 2019
default_end_year   = 2024

st.write("#### Date Range for XY Chart")
col_s1, col_s2 = st.columns(2)
with col_s1:
    xy_start_month = st.selectbox("Start Month (XY)", months, index=0)
    xy_start_year  = st.selectbox("Start Year (XY)", years_xy, index=years_xy.index(default_start_year))
with col_s2:
    xy_end_month = st.selectbox("End Month (XY)", months, index=11)
    xy_end_year  = st.selectbox("End Year (XY)", years_xy, index=years_xy.index(default_end_year))

xy_smonth_num = months.index(xy_start_month) + 1
xy_emonth_num = months.index(xy_end_month) + 1
xy_start_ym   = f"{xy_start_year:04d}-{xy_smonth_num:02d}"
xy_end_ym     = f"{xy_end_year:04d}-{xy_emonth_num:02d}"

if st.button("Generate XY Chart"):
    if not st.session_state["msa_selection"]:
        st.warning("No MSAs selected!")
    else:
        chosen_ids = [INVERTED_MAP[m] for m in st.session_state["msa_selection"]]
        df_raw = fetch_raw_data_multiple(chosen_ids, xy_start_ym, xy_end_ym)
        if df_raw.empty:
            st.warning("No data found. Check your CSV or date range.")
        else:
            df_ab = compute_multi_alpha_beta(df_raw)
            if df_ab.empty:
                st.error("Could not compute alpha/beta. Possibly not enough overlapping data with National.")
            else:
                fig_xy = px.scatter(
                    df_ab,
                    x="Beta",
                    y="Alpha",
                    text="Metro",
                    title=f"Alpha vs. Beta ({xy_start_ym} to {xy_end_ym})",
                    render_mode="webgl"
                )
                fig_xy.update_traces(textposition='top center', textfont_size=14)
                fig_xy.update_layout(dragmode='pan', title_x=0.5, title_xanchor='center')
                fig_xy.add_hline(
                    y=0, line_width=2, line_color="black", line_dash="dot",
                    annotation_text="Alpha = 0", annotation_position="top left"
                )
                fig_xy.add_vline(
                    x=1, line_width=2, line_color="black", line_dash="dot",
                    annotation_text="Beta = 1", annotation_position="bottom right"
                )
                st.session_state["df_ab"] = df_ab
                st.session_state["fig_xy"] = fig_xy

if st.session_state["fig_xy"] is not None:
    st.plotly_chart(
        st.session_state["fig_xy"],
        use_container_width=True,
        config={"scrollZoom": True}
    )
    if st.checkbox("View alpha/beta table for XY chart"):
        st.dataframe(st.session_state["df_ab"])

# ---------------------------------------------------------------------
# 7) TIME SERIES (Rolling Alpha/Beta) w/ Debug
# ---------------------------------------------------------------------
st.markdown("### Time Series (Rolling Alpha/Beta)")

if "df_ts" not in st.session_state:
    st.session_state["df_ts"] = None
if "fig_ts" not in st.session_state:
    st.session_state["fig_ts"] = None

all_msa_names_no_nat = [m for m in all_msa_names if m != "National"]
selected_time_msas = st.multiselect(
    "Pick up to 5 MSAs for Time Series:",
    options=all_msa_names_no_nat,
    max_selections=5
)

ab_choice = st.selectbox("Which metric to graph in the Time Series?", ["alpha", "beta"])

ts_default_start = 2019
ts_default_end   = 2024
ts_default_start_index = years_xy.index(ts_default_start)
ts_default_end_index   = years_xy.index(ts_default_end)

st.write("#### Date Range for Time Series")
col_t1, col_t2 = st.columns(2)
with col_t1:
    ts_start_month = st.selectbox("Start Month (Time Series)", months, index=0)
    ts_start_year  = st.selectbox("Start Year (Time Series)", years_xy, index=ts_default_start_index)
with col_t2:
    ts_end_month = st.selectbox("End Month (Time Series)", months, index=11)
    ts_end_year  = st.selectbox("End Year (Time Series)", years_xy, index=ts_default_end_index)

ts_smonth_num = months.index(ts_start_month) + 1
ts_emonth_num = months.index(ts_end_month) + 1
ts_start_ym   = f"{ts_start_year:04d}-{ts_smonth_num:02d}"
ts_end_ym     = f"{ts_end_year:04d}-{ts_emonth_num:02d}"

def compute_alpha_beta_subset(df_subset, nat_col, msa_col):
    if len(df_subset.dropna()) < 2:
        return None, None
    X = sm.add_constant(df_subset[nat_col])
    y = df_subset[msa_col]
    model = sm.OLS(y, X).fit()
    return model.params["const"], model.params[nat_col]

def compute_alpha_beta_time_series(df_raw_ts, start_ym_ts, end_ym_ts):
    """
    Rolling approach: For each month in [start, end], run OLS on all data up to that month.
    Debug statements included for pivot and monthly growth checks.
    """
    # 1) Pivot
    df_pivot = df_raw_ts.pivot(index="obs_date", columns="series_id", values="value")

    st.write("DEBUG: Pivot table (first 10 rows):")
    st.dataframe(df_pivot.head(10))

    # 2) monthly % change
    df_growth = df_pivot.pct_change(1) * 100
    df_growth.dropna(how="all", inplace=True)  # drop rows entirely NaN
    st.write("DEBUG: After pct_change, 'df_growth' (first 10 rows):")
    st.dataframe(df_growth.head(10))

    # Must have National
    if NATIONAL_SERIES_ID not in df_growth.columns:
        st.write("DEBUG: National series not in columns. Columns are:", df_growth.columns.tolist())
        return pd.DataFrame()

    # 3) Filter growth to the user’s date range
    start_dt = pd.to_datetime(f"{start_ym_ts}-01")
    end_dt   = pd.to_datetime(f"{end_ym_ts}-01")
    df_growth = df_growth.loc[(df_growth.index >= start_dt) & (df_growth.index <= end_dt)]

    unique_months_ts = sorted(df_growth.index.unique())
    results_ts = []

    for current_month in unique_months_ts:
        df_window = df_growth.loc[df_growth.index <= current_month]
        for msa_col in df_growth.columns:
            if msa_col == NATIONAL_SERIES_ID:
                continue
            data_subset = df_window[[NATIONAL_SERIES_ID, msa_col]].dropna()
            alpha_val, beta_val = compute_alpha_beta_subset(data_subset, NATIONAL_SERIES_ID, msa_col)
            if alpha_val is not None and beta_val is not None:
                results_ts.append({
                    "obs_date": current_month,
                    "series_id": msa_col,
                    "alpha": alpha_val,
                    "beta": beta_val
                })

    df_ts = pd.DataFrame(results_ts)
    return df_ts

if st.button("Compute Time Series"):
    if not selected_time_msas:
        st.warning("Pick at least 1 MSA (up to 5).")
    else:
        chosen_time_ids = [INVERTED_MAP[n] for n in selected_time_msas]
        df_raw_ts = fetch_raw_data_multiple(chosen_time_ids, ts_start_ym, ts_end_ym)
        if df_raw_ts.empty:
            st.warning("No data found for that range. Check your CSV or input months/years.")
        else:
            df_ts_result = compute_alpha_beta_time_series(df_raw_ts, ts_start_ym, ts_end_ym)
            if df_ts_result.empty:
                st.warning("Could not compute time-series alpha/beta. Possibly insufficient overlapping data.")
            else:
                # "Metro" column
                df_ts_result["Metro"] = df_ts_result["series_id"].apply(lambda sid: MSA_NAME_MAP.get(sid, sid))
                df_ts_result.drop(columns=["series_id"], inplace=True)
                df_ts_result.rename(columns={"obs_date": "Date"}, inplace=True)
                df_ts_result = df_ts_result[["Date", "Metro", "alpha", "beta"]]

                st.session_state["df_ts"] = df_ts_result

                # Plot
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
