import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import datetime

# ---------------------------------------------------------------------
# 1) Wide Mode + Page Title
# ---------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="US Metro Alpha/Beta Analysis (Rolling Window)")

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
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# 3) Main Title
# ---------------------------------------------------------------------
st.title("US Metro Alpha/Beta Analysis (Rolling Window)")

# ---------------------------------------------------------------------
# 4) Expanders with Descriptive Text
# ---------------------------------------------------------------------
with st.expander("About the Data", expanded=False):
    st.markdown(
        """
        **Data Source**  
        - All data is publicly available from the Bureau of Labor Statistics (BLS).  
        - This tool currently features the top 50 U.S. MSAs (Metropolitan Statistical Area) by population, plus a “National” benchmark.

        **Alpha**  
        - Alpha measures a MSA’s growth rate relative to the national average.  
        - A positive alpha means the MSA outperforms the nation; a negative alpha means it underperforms.  
        - For instance, an alpha of +1.5 implies the MSA grew about 1.5% faster than the national rate over the chosen period.

        **Beta**  
        - Beta measures how sensitive (or volatile) an MSA is relative to the nation’s changes.  
        - If beta = 1, the MSA moves in sync with the national trend.  
        - A beta > 1 indicates greater volatility (e.g., 1.5 → 50% larger swings), while a beta < 1 indicates less volatility (e.g., 0.5 → half as volatile).

        **Note on Rolling Time Series**  
        - This app now uses a 12-month rolling window for time-series alpha/beta.  
        - Each monthly alpha/beta is calculated from that month and the previous 11 months only, making the series more responsive to newer data.
        """
    )

with st.expander("How To Use", expanded=False):
    st.markdown(
        """
        **Select the desired metric from the dropdown below (currently only one option).**  

        **XY Chart (Alpha vs. Beta)**  
        1. Pick MSA(s) from the dropdown, or click “Select All” to include the top 50 MSAs.  
        2. Choose a start and end date range.  
        3. Click “Generate XY Chart.” Each MSA is plotted by (β, α).  

        **Time Series (Rolling Alpha/Beta Over the Selected Date Range)**  
        1. Select up to 5 MSAs (excluding “National,” which is the benchmark).  
        2. Pick Alpha or Beta to track, plus a separate date range.  
        3. Click “Compute Time Series.”  
        4. Each month’s alpha/beta is calculated from that month and the prior 11 months only, forming a 12-month rolling window.

        **Scenario Analysis**  
        - Enter a 5-year (or longer) historical date range so we have enough data to compute alpha/beta.  
        - Input up to 3 hypothetical national growth scenarios.  
        - See the resulting MSA growth predictions for each scenario in a new XY chart, plus a confidence note using R-squared.
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
# 6) Load CSV from GitHub (or local DB)
# ---------------------------------------------------------------------
github_csv_url = "https://raw.githubusercontent.com/Ryangab8/InquiRE/main/raw_nonfarm_jobs.csv"
df_full = pd.read_csv(github_csv_url)

# Convert obs_date to datetime
df_full["obs_date"] = pd.to_datetime(df_full["obs_date"], errors="coerce")

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
# Utility Functions
# ---------------------------------------------------------------------
def fetch_raw_data_multiple(msa_ids, start_ym, end_ym):
    """
    Fetch filtered data for the specified MSA IDs and date range.
    """
    if NATIONAL_SERIES_ID not in msa_ids:
        msa_ids.append(NATIONAL_SERIES_ID)

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

def compute_multi_alpha_beta(df_raw):
    """
    Computes alpha and beta for each MSA vs. the national benchmark
    over the entire date range of df_raw (expanding approach).
    """
    df_raw["value"] = pd.to_numeric(df_raw["value"], errors="coerce")
    df_pivot = df_raw.pivot(index="obs_date", columns="series_id", values="value")
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
    df_ab = df_ab[["Metro", "Alpha", "Beta", "R-Squared"]]
    return df_ab

def compute_alpha_beta_subset(df_subset, nat_col, msa_col):
    """
    Helper function to compute alpha/beta for a given subset.
    Returns (alpha, beta) or (None, None) if insufficient data.
    """
    if len(df_subset.dropna()) < 2:
        return None, None
    X = sm.add_constant(df_subset[nat_col])
    y = df_subset[msa_col]
    model = sm.OLS(y, X).fit()
    return model.params["const"], model.params[nat_col]

# Rolling time-series function (12-month lookback)
ROLLING_WINDOW_MONTHS = 12

def compute_rolling_alpha_beta_time_series(df_raw_ts, start_ym_ts, end_ym_ts):
    """
    1) Pivot MSA data -> monthly values
    2) Compute month-to-month % changes
    3) For each month in [start_ym_ts, end_ym_ts], take the *prior 11 months + current month*
       (12-month window) and run an OLS regression vs. National growth.
    4) Return a DataFrame of (obs_date, series_id, alpha, beta).
    """
    df_pivot = df_raw_ts.pivot(index="obs_date", columns="series_id", values="value")
    df_growth = df_pivot.pct_change(1) * 100
    df_growth.dropna(inplace=True)

    if NATIONAL_SERIES_ID not in df_growth.columns:
        return pd.DataFrame()

    start_dt = pd.to_datetime(f"{start_ym_ts}-01")
    end_dt   = pd.to_datetime(f"{end_ym_ts}-01")
    unique_months_ts = sorted(df_growth.index.unique())

    results_ts = []
    for current_month in unique_months_ts:
        if current_month < start_dt or current_month > end_dt:
            continue
        rolling_start = current_month - pd.DateOffset(months=ROLLING_WINDOW_MONTHS - 1)
        df_window = df_growth.loc[(df_growth.index >= rolling_start) & (df_growth.index <= current_month)]
        if len(df_window) < 2:
            continue

        for msa_col in df_growth.columns:
            if msa_col == NATIONAL_SERIES_ID:
                continue
            subset = df_window[[NATIONAL_SERIES_ID, msa_col]].dropna()
            alpha_val, beta_val = compute_alpha_beta_subset(subset, NATIONAL_SERIES_ID, msa_col)
            if alpha_val is not None and beta_val is not None:
                results_ts.append({
                    "obs_date": current_month,
                    "series_id": msa_col,
                    "alpha": alpha_val,
                    "beta": beta_val
                })
    return pd.DataFrame(results_ts)

# ---------------------------------------------------------------------
# 7) XY Chart (Alpha vs. Beta)
# ---------------------------------------------------------------------
def select_all():
    st.session_state["msa_selection"] = sorted(INVERTED_MAP.keys())

def clear_all():
    st.session_state["msa_selection"] = []

if "msa_selection" not in st.session_state:
    st.session_state["msa_selection"] = []
if "df_ab" not in st.session_state:
    st.session_state["df_ab"] = None
if "fig" not in st.session_state:
    st.session_state["fig"] = None

st.markdown("### XY Chart (Alpha vs. Beta)")

all_msa_names = sorted(INVERTED_MAP.keys())

st.multiselect(
    "Pick MSA(s):",
    options=all_msa_names,
    key="msa_selection"
)

col1, col2 = st.columns(2)
col1.button("Select All", on_click=select_all)
col2.button("Clear", on_click=clear_all)

months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
years_xy = list(range(1990, 2026))
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
            st.warning("No data found for that range. Check your CSV or chosen date range.")
        else:
            df_ab = compute_multi_alpha_beta(df_raw)
            if df_ab.empty:
                st.error("Could not compute alpha/beta. Possibly insufficient data overlap.")
            else:
                title_xy = f"Alpha vs. Beta ({xy_start_ym} to {xy_end_ym})"
                fig_xy = px.scatter(
                    df_ab,
                    x="Beta",
                    y="Alpha",
                    text="Metro",
                    title=title_xy,
                    render_mode="webgl"
                )
                fig_xy.update_traces(textposition='top center', textfont_size=14)
                fig_xy.update_layout(dragmode='pan', title_x=0.5, title_xanchor='center')
                fig_xy.add_hline(
                    y=0, line_width=3, line_color="black", line_dash="dot",
                    annotation_text="Alpha = 0", annotation_position="top left"
                )
                fig_xy.add_vline(
                    x=1, line_width=3, line_color="black", line_dash="dot",
                    annotation_text="Beta = 1", annotation_position="bottom right"
                )
                st.session_state["df_ab"] = df_ab
                st.session_state["fig"] = fig_xy

if st.session_state["fig"] is not None:
    st.plotly_chart(
        st.session_state["fig"],
        use_container_width=True,
        config={"scrollZoom": True}
    )
    if st.checkbox("View alpha/beta table for XY chart"):
        st.dataframe(st.session_state["df_ab"])

# ---------------------------------------------------------------------
# 8) TIME SERIES (Rolling Alpha/Beta)
# ---------------------------------------------------------------------
st.markdown("### Time Series (Rolling 12-Month Alpha/Beta)")

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

ts_default_start_year = 2019
ts_default_end_year   = 2024

st.write("#### Date Range for Time Series (Rolling Window)")
col_t1, col_t2 = st.columns(2)
with col_t1:
    ts_start_month = st.selectbox("Start Month (Time Series)", months, index=0)
    ts_start_year  = st.selectbox("Start Year (Time Series)", years_xy, index=years_xy.index(ts_default_start_year))
with col_t2:
    ts_end_month = st.selectbox("End Month (Time Series)", months, index=11)
    ts_end_year  = st.selectbox("End Year (Time Series)", years_xy, index=years_xy.index(ts_default_end_year))

ts_smonth_num = months.index(ts_start_month) + 1
ts_emonth_num = months.index(ts_end_month) + 1
ts_start_ym   = f"{ts_start_year:04d}-{ts_smonth_num:02d}"
ts_end_ym     = f"{ts_end_year:04d}-{ts_emonth_num:02d}"

if st.button("Compute Time Series"):
    if not selected_time_msas:
        st.warning("Pick at least 1 MSA (up to 5).")
    else:
        chosen_time_ids = [INVERTED_MAP[n] for n in selected_time_msas]
        # We'll fetch from 1990-01 to ensure we have enough history for a 12-month rolling window:
        df_raw_ts = fetch_raw_data_multiple(chosen_time_ids, "1990-01", ts_end_ym)
        if df_raw_ts.empty:
            st.warning("No data found. Check your CSV or chosen date range.")
        else:
            df_ts_result = compute_rolling_alpha_beta_time_series(df_raw_ts, ts_start_ym, ts_end_ym)
            if df_ts_result.empty:
                st.warning("Could not compute time-series alpha/beta. Possibly insufficient data.")
            else:
                df_ts_result["Metro"] = df_ts_result["series_id"].apply(lambda sid: MSA_NAME_MAP.get(sid, sid))
                df_ts_result.drop(columns=["series_id"], inplace=True)
                df_ts_result.rename(columns={"obs_date": "Date"}, inplace=True)
                df_ts_result = df_ts_result[["Date", "Metro", "alpha", "beta"]]

                st.session_state["df_ts"] = df_ts_result

                df_plot = df_ts_result.copy()
                df_plot["AB_Chosen"] = df_plot[ab_choice]
                title_ts = f"Time Series of {ab_choice.title()} (Rolling 12-Month) {ts_start_ym}–{ts_end_ym}"
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

# ---------------------------------------------------------------------
# 9) SCENARIO ANALYSIS
# ---------------------------------------------------------------------
st.markdown("### Scenario Analysis (Alpha/Beta Forecast)")

st.write("Enter a date range **of at least 5 years** to compute alpha/beta for each MSA, then input up to 3 national growth scenarios.")

# MSA selection for scenario analysis
scenario_msa = st.multiselect(
    "Pick MSA(s) for Scenario Analysis",
    options=all_msa_names,
    help="Pick one or many MSAs to see how they might perform under custom national growth scenarios."
)

# Date range selection (must be >= 5 years)
scenario_years = list(range(1990, 2030))
default_scenario_start = 2018
default_scenario_end   = 2023

col_sa1, col_sa2 = st.columns(2)
with col_sa1:
    sc_start_month = st.selectbox("Scenario Start Month", months, index=0)
    sc_start_year  = st.selectbox("Scenario Start Year", scenario_years, index=scenario_years.index(default_scenario_start))
with col_sa2:
    sc_end_month = st.selectbox("Scenario End Month", months, index=11)
    sc_end_year  = st.selectbox("Scenario End Year", scenario_years, index=scenario_years.index(default_scenario_end))

sc_smonth_num = months.index(sc_start_month) + 1
sc_emonth_num = months.index(sc_end_month) + 1
sc_start_ym   = f"{sc_start_year:04d}-{sc_smonth_num:02d}"
sc_end_ym     = f"{sc_end_year:04d}-{sc_emonth_num:02d}"

# Text inputs for up to 3 national growth scenarios
st.write("#### Enter up to 3 hypothetical national growth rates (in %):")
scenario_1 = st.text_input("Scenario #1 (e.g., 1.0)", value="1.0")
scenario_2 = st.text_input("Scenario #2 (optional)", value="2.5")
scenario_3 = st.text_input("Scenario #3 (optional)", value="4.0")

if st.button("Run Scenario Analysis"):
    # Validate date range
    start_dt = datetime.datetime(sc_start_year, sc_smonth_num, 1)
    end_dt   = datetime.datetime(sc_end_year, sc_emonth_num, 1)
    total_months = (end_dt.year - start_dt.year)*12 + (end_dt.month - start_dt.month)
    if total_months < 60:  # roughly 5 years in months
        st.error("Please select at least a 5-year range for scenario analysis.")
    elif not scenario_msa:
        st.warning("No MSAs selected for scenario analysis.")
    else:
        # Convert scenario inputs to floats (ignore empty fields)
        def parse_float_safe(x):
            try:
                return float(x)
            except:
                return None
        
        s1 = parse_float_safe(scenario_1)
        s2 = parse_float_safe(scenario_2)
        s3 = parse_float_safe(scenario_3)
        scenario_vals = [(f"Scenario #1 ({s1}%)", s1) if s1 is not None else None,
                         (f"Scenario #2 ({s2}%)", s2) if s2 is not None else None,
                         (f"Scenario #3 ({s3}%)", s3) if s3 is not None else None]
        scenario_vals = [sv for sv in scenario_vals if sv and sv[1] is not None]

        if not scenario_vals:
            st.error("Please enter at least one valid scenario growth rate (numeric).")
        else:
            chosen_scenario_ids = [INVERTED_MAP[m] for m in scenario_msa]
            df_raw_scenario = fetch_raw_data_multiple(chosen_scenario_ids, sc_start_ym, sc_end_ym)
            if df_raw_scenario.empty:
                st.warning("No data found for that scenario range. Check your CSV or chosen date range.")
            else:
                df_ab_scenario = compute_multi_alpha_beta(df_raw_scenario)
                if df_ab_scenario.empty:
                    st.error("Could not compute alpha/beta for scenarios. Possibly insufficient data.")
                else:
                    # Build scenario data for plotting
                    scenario_plot_data = []
                    for scenario_label, sc_val in scenario_vals:
                        # Predicted MSA growth = alpha + beta * sc_val
                        df_ab_scenario[scenario_label] = df_ab_scenario["Alpha"] + df_ab_scenario["Beta"] * sc_val

                    # We will "melt" the data to plot each scenario as a separate trace
                    melt_cols = [sv[0] for sv in scenario_vals]  # scenario labels
                    df_melt = df_ab_scenario.melt(
                        id_vars=["Metro", "Alpha", "Beta", "R-Squared"],
                        value_vars=melt_cols,
                        var_name="Scenario",
                        value_name="Predicted Growth"
                    )

                    # XY chart: x=Beta, y=Predicted Growth, color=Scenario, text=Metro
                    fig_scen = px.scatter(
                        df_melt,
                        x="Beta",
                        y="Predicted Growth",
                        color="Scenario",
                        text="Metro",
                        title="Scenario Analysis: Beta vs. Predicted Growth",
                        render_mode="webgl"
                    )
                    fig_scen.update_traces(textposition='top center', textfont_size=12)
                    fig_scen.add_vline(
                        x=1, line_width=2, line_color="black", line_dash="dot",
                        annotation_text="Beta=1", annotation_position="bottom right"
                    )
                    fig_scen.update_layout(dragmode='pan', title_x=0.5, title_xanchor='center')

                    st.plotly_chart(fig_scen, use_container_width=True, config={"scrollZoom": True})

                    # Show a data table with Alpha, Beta, R-Sq, plus each scenario
                    st.write("#### Detailed Results")
                    st.dataframe(df_ab_scenario)

                    # Simple confidence note based on R-squared
                    st.markdown(
                        """
                        **Confidence Note**  
                        - R-Squared indicates how much of the MSA's job growth variation is explained by national trends.  
                        - Higher R-Squared (e.g., 0.70 or above) → **More reliable** predictions using alpha/beta.  
                        - Lower R-Squared → The MSA has more local or idiosyncratic factors, so scenario forecasts may be **less reliable**.
                        """
                    )

