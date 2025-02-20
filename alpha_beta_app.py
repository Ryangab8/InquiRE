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

        **Year-over-Year Bar Chart**  
        1. Pick a single MSA and define a historical range of years.  
        2. We compute January-to-January YOY % changes for National vs. your MSA.  
        3. We regress MSA YOY on National YOY to get alpha & beta.  
        4. Enter a next-year National forecast, and we'll show the implied MSA growth in the bar chart.
        """
    )

# ---------------------------------------------------------------------
# 5) Global Metric Selector (currently only 1 option)
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
    if len(df_subset.dropna()) < 2:
        return None, None
    X = sm.add_constant(df_subset[nat_col])
    y = df_subset[msa_col]
    model = sm.OLS(y, X).fit()
    return model.params["const"], model.params[nat_col]

ROLLING_WINDOW_MONTHS = 12
def compute_rolling_alpha_beta_time_series(df_raw_ts, start_ym_ts, end_ym_ts):
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
        df_window = df_growth.loc[(df_growth.index>=rolling_start) & (df_growth.index<=current_month)]
        if len(df_window)<2:
            continue

        for col in df_growth.columns:
            if col == NATIONAL_SERIES_ID:
                continue
            subset = df_window[[NATIONAL_SERIES_ID,col]].dropna()
            alpha_val, beta_val = compute_alpha_beta_subset(subset, NATIONAL_SERIES_ID, col)
            if alpha_val is not None and beta_val is not None:
                results_ts.append({
                    "obs_date": current_month,
                    "series_id": col,
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
# 8) TIME SERIES (Rolling 12-Month Alpha/Beta)
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
# 9) YEAR-OVER-YEAR BAR CHART (NEW)
# ---------------------------------------------------------------------
st.markdown("### Year-over-Year Bar Chart: National vs. Single MSA + Forecast")

st.write("""
We use each January's Total Nonfarm value to compute year-over-year % changes for both the chosen MSA 
and the national benchmark. Then, we regress the MSA's YoY on the national YoY to find alpha/beta. 
Finally, you can enter a future national YoY forecast, and we'll show the implied MSA YoY as well.
""")

# 9a) User picks single MSA
msa_choice = st.selectbox("Pick an MSA for YOY Chart:", sorted(INVERTED_MAP.keys()))
msa_id = INVERTED_MAP[msa_choice]

# 9b) Historical years selection
years_list = list(range(1990, 2031))
c1, c2 = st.columns(2)
with c1:
    yoy_start = st.selectbox("Start Year (Jan)", years_list, index=years_list.index(2015))
with c2:
    yoy_end   = st.selectbox("End Year (Jan)", years_list, index=years_list.index(2024))

# 9c) Next-year forecast for National
nat_forecast = st.number_input("Projected Next-Year National YoY Growth (%)", value=1.0, step=0.1, format="%.1f")

if st.button("Generate YoY Bar Chart"):
    if yoy_end < yoy_start:
        st.error("End year cannot be earlier than start year.")
        st.stop()

    # Filter to relevant series
    sel_ids = [msa_id, NATIONAL_SERIES_ID]
    df_filtered = df_full[df_full["series_id"].isin(sel_ids)].copy()

    # Only january data
    df_filtered["year"] = df_filtered["obs_date"].dt.year
    df_filtered["month"] = df_filtered["obs_date"].dt.month
    df_jan = df_filtered[df_filtered["month"]==1].copy()
    df_jan = df_jan[(df_jan["year"]>=yoy_start) & (df_jan["year"]<=yoy_end)]

    if df_jan.empty:
        st.warning("No January data found in that range.")
        st.stop()

    # Pivot => row=year, columns=series_id, value=jobs
    pivot_jan = df_jan.pivot(index="year", columns="series_id", values="value")
    pivot_jan.dropna(inplace=True)
    if NATIONAL_SERIES_ID not in pivot_jan.columns or msa_id not in pivot_jan.columns:
        st.warning("Missing data for MSA or National in january pivot.")
        st.stop()

    # Build yoy
    yoy_records = []
    sorted_years = sorted(pivot_jan.index)
    for y in sorted_years:
        prev_y = y - 1
        if prev_y in pivot_jan.index:
            nat_t   = pivot_jan.loc[y, NATIONAL_SERIES_ID]
            nat_tm1 = pivot_jan.loc[prev_y, NATIONAL_SERIES_ID]
            yoy_nat = 100*(nat_t - nat_tm1)/nat_tm1

            msa_t   = pivot_jan.loc[y, msa_id]
            msa_tm1 = pivot_jan.loc[prev_y, msa_id]
            yoy_msa = 100*(msa_t - msa_tm1)/msa_tm1

            yoy_records.append({
                "Year": y,
                "Nat_Growth": yoy_nat,
                "MSA_Growth": yoy_msa
            })

    df_yoy = pd.DataFrame(yoy_records)
    if df_yoy.empty:
        st.warning("Not enough consecutive january data to compute yoy.")
        st.stop()

    # OLS yoy_msa = alpha + beta * yoy_nat
    X = sm.add_constant(df_yoy["Nat_Growth"])
    y = df_yoy["MSA_Growth"]
    model = sm.OLS(y, X).fit()
    alpha_val = model.params["const"]
    beta_val  = model.params["Nat_Growth"]
    r_sq      = model.rsquared

    # Next year's implied MSA yoy
    yoy_msa_fore = alpha_val + beta_val * nat_forecast

    # We'll label the forecast as "Forecast" in the chart
    forecast_rec = {
        "Year": "Forecast",
        "Nat_Growth": nat_forecast,
        "MSA_Growth": yoy_msa_fore
    }

    df_plot = df_yoy.copy()
    df_plot.loc[df_plot.shape[0]] = forecast_rec  # add forecast row

    # Melt for side-by-side bars
    bar_data = []
    for i, row_ in df_plot.iterrows():
        bar_data.append({
            "Year": row_["Year"],
            "Series": "National",
            "Growth": row_["Nat_Growth"]
        })
        bar_data.append({
            "Year": row_["Year"],
            "Series": msa_choice,
            "Growth": row_["MSA_Growth"]
        })

    df_bar = pd.DataFrame(bar_data)

    fig_bar = px.bar(
        df_bar,
        x="Year",
        y="Growth",
        color="Series",
        barmode="group",
        title=f"Year-over-Year Growth (Jan) — National vs {msa_choice}"
    )
    fig_bar.update_layout(xaxis_type='category')

    st.plotly_chart(fig_bar, use_container_width=True)

    # Show results in a small table
    st.markdown("#### OLS Regression (YoY) Results")
    df_result = pd.DataFrame([{
        "MSA": msa_choice,
        "Alpha": alpha_val,
        "Beta": beta_val,
        "R-Squared": r_sq
    }])
    st.dataframe(df_result)

    st.markdown(f"""
    **Interpretation**  
    - Years: {sorted_years[0]} - {sorted_years[-1]}  
    - alpha = {alpha_val:.3f}, beta = {beta_val:.3f}, R² = {r_sq:.2f}  
    - With your forecast of National = {nat_forecast:.1f}%, 
      implied MSA yoy = {yoy_msa_fore:.1f}%.
    """)


