import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import datetime

# ---------------------------------------------------------------------
# 1) Wide Mode + Page Title
# ---------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="US Metro Alpha/Beta Analysis - Line Scenarios")

# ---------------------------------------------------------------------
# 2) Custom CSS (optional)
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: "Times New Roman", serif;
    }
    h1 {
        text-align: center;
        color: black !important;
    }
    :root {
        --primary-color: #93D0EC; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# 3) Title & Intro
# ---------------------------------------------------------------------
st.title("US Metro Alpha/Beta Analysis")

with st.expander("About the Data", expanded=False):
    st.markdown(
        """
        **Data Source**  
        - All data is from the Bureau of Labor Statistics (BLS).  

        **Alpha / Beta**  
        - **Alpha**: The intercept (baseline MSA growth if national growth is 0%).  
        - **Beta**: How strongly the MSA’s growth responds to national growth changes (β>1 → amplifies, β<1 → more stable).  

        **Line Chart Scenarios**  
        - We'll show historical monthly job growth (solid lines) plus forecasted lines (dotted) if you provide future national growth assumptions.  
        """
    )

# ---------------------------------------------------------------------
# 4) Load CSV from GitHub (or local DB)
# ---------------------------------------------------------------------
github_csv_url = "https://raw.githubusercontent.com/Ryangab8/InquiRE/main/raw_nonfarm_jobs.csv"
df_full = pd.read_csv(github_csv_url)

# Convert obs_date to datetime
df_full["obs_date"] = pd.to_datetime(df_full["obs_date"], errors="coerce")

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
# Utility function for data fetch
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

# ---------------------------------------------------------------------
# 5) (Optional) XY CHART (Alpha vs. Beta)
# ---------------------------------------------------------------------
st.markdown("### XY Chart (Alpha vs. Beta) [Optional]")

all_msa_names = sorted(INVERTED_MAP.keys())

xy_selection = st.multiselect(
    "Pick MSA(s) for XY chart:",
    options=all_msa_names
)

months = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
years_xy = list(range(1990, 2026))
xy_start_year = st.selectbox("XY Start Year", years_xy, index=5)
xy_start_month = st.selectbox("XY Start Month", months, index=0)
xy_end_year   = st.selectbox("XY End Year", years_xy, index=25)
xy_end_month  = st.selectbox("XY End Month", months, index=11)

xy_smonth_num = months.index(xy_start_month)+1
xy_emonth_num = months.index(xy_end_month)+1
xy_start_ym = f"{xy_start_year:04d}-{xy_smonth_num:02d}"
xy_end_ym   = f"{xy_end_year:04d}-{xy_emonth_num:02d}"

def compute_multi_alpha_beta(df_raw):
    df_raw["value"] = pd.to_numeric(df_raw["value"], errors="coerce")
    df_pivot = df_raw.pivot(index="obs_date", columns="series_id", values="value")
    df_growth = df_pivot.pct_change(1)*100
    df_growth.dropna(inplace=True)

    if NATIONAL_SERIES_ID not in df_growth.columns:
        return pd.DataFrame()

    results = []
    for col in df_growth.columns:
        if col == NATIONAL_SERIES_ID:
            continue
        merged = df_growth[[NATIONAL_SERIES_ID, col]].dropna()
        if len(merged)<2:
            continue
        X = sm.add_constant(merged[NATIONAL_SERIES_ID])
        y = merged[col]
        model = sm.OLS(y, X).fit()
        alpha_val = model.params["const"]
        beta_val  = model.params[NATIONAL_SERIES_ID]
        r_sq      = model.rsquared

        results.append({
            "series_id": col,
            "Metro": MSA_NAME_MAP.get(col,col),
            "Alpha": alpha_val,
            "Beta": beta_val,
            "R-Squared": r_sq
        })
    return pd.DataFrame(results)

if st.button("Generate XY Chart"):
    if not xy_selection:
        st.warning("No MSAs selected.")
    else:
        chosen_ids = [INVERTED_MAP[m] for m in xy_selection]
        df_xy = fetch_raw_data_multiple(chosen_ids, xy_start_ym, xy_end_ym)
        if df_xy.empty:
            st.warning("No data found.")
        else:
            df_xy_ab = compute_multi_alpha_beta(df_xy)
            if df_xy_ab.empty:
                st.error("No alpha/beta computed. Possibly insufficient data.")
            else:
                fig_xy = px.scatter(
                    df_xy_ab,
                    x="Beta",
                    y="Alpha",
                    text="Metro",
                    title=f"Alpha vs. Beta ({xy_start_ym} to {xy_end_ym})"
                )
                fig_xy.update_traces(textposition='top center')
                fig_xy.add_vline(x=1, line_color='black', line_dash='dot')
                fig_xy.add_hline(y=0, line_color='black', line_dash='dot')
                st.plotly_chart(fig_xy, use_container_width=True)
                st.dataframe(df_xy_ab)

# ---------------------------------------------------------------------
# 6) TIME SERIES (Rolling Alpha/Beta) [Optional]
# ---------------------------------------------------------------------
st.markdown("### Time Series (Rolling 12-Month Alpha/Beta) [Optional]")

ROLLING_WINDOW_MONTHS = 12
def compute_rolling_alpha_beta_time_series(df_raw_ts, start_ym_ts, end_ym_ts):
    df_pivot = df_raw_ts.pivot(index="obs_date", columns="series_id", values="value")
    df_growth = df_pivot.pct_change(1)*100
    df_growth.dropna(inplace=True)

    if NATIONAL_SERIES_ID not in df_growth.columns:
        return pd.DataFrame()

    start_dt = pd.to_datetime(f"{start_ym_ts}-01")
    end_dt   = pd.to_datetime(f"{end_ym_ts}-01")
    unique_months_ts = sorted(df_growth.index.unique())

    results_ts = []
    for current_month in unique_months_ts:
        if current_month<start_dt or current_month>end_dt:
            continue
        rolling_start = current_month - pd.DateOffset(months=ROLLING_WINDOW_MONTHS-1)
        df_window = df_growth.loc[(df_growth.index>=rolling_start) & (df_growth.index<=current_month)]
        if len(df_window)<2:
            continue
        for col in df_growth.columns:
            if col == NATIONAL_SERIES_ID:
                continue
            subset = df_window[[NATIONAL_SERIES_ID,col]].dropna()
            if len(subset)<2:
                continue
            X = sm.add_constant(subset[NATIONAL_SERIES_ID])
            y = subset[col]
            model = sm.OLS(y,X).fit()
            alpha_val = model.params["const"]
            beta_val  = model.params[NATIONAL_SERIES_ID]
            results_ts.append({
                "obs_date": current_month,
                "series_id": col,
                "alpha": alpha_val,
                "beta": beta_val
            })
    return pd.DataFrame(results_ts)

all_msa_names_no_nat = [m for m in all_msa_names if m!="National"]
selected_time_msas = st.multiselect("Pick up to 5 MSAs:", options=all_msa_names_no_nat, max_selections=5)
ab_choice = st.selectbox("Which metric to graph in Time Series?", ["alpha","beta"])

ts_years = list(range(1990,2030))
ts_start_year = st.selectbox("TS Start Year", ts_years, index=5)
ts_start_month = st.selectbox("TS Start Month", months, index=0)
ts_end_year   = st.selectbox("TS End Year", ts_years, index=25)
ts_end_month  = st.selectbox("TS End Month", months, index=11)

ts_smonth_num = months.index(ts_start_month)+1
ts_emonth_num = months.index(ts_end_month)+1
ts_start_ym = f"{ts_start_year:04d}-{ts_smonth_num:02d}"
ts_end_ym   = f"{ts_end_year:04d}-{ts_emonth_num:02d}"

if st.button("Compute Rolling Time Series"):
    if not selected_time_msas:
        st.warning("No MSAs selected.")
    else:
        chosen_time_ids = [INVERTED_MAP[m] for m in selected_time_msas]
        df_ts_raw = fetch_raw_data_multiple(chosen_time_ids, "1990-01", ts_end_ym)
        if df_ts_raw.empty:
            st.warning("No data found.")
        else:
            df_ts_res = compute_rolling_alpha_beta_time_series(df_ts_raw, ts_start_ym, ts_end_ym)
            if df_ts_res.empty:
                st.warning("No rolling alpha/beta computed.")
            else:
                df_ts_res["Metro"] = df_ts_res["series_id"].map(lambda x: MSA_NAME_MAP.get(x,x))
                df_ts_res.drop(columns=["series_id"], inplace=True)
                df_ts_res.rename(columns={"obs_date":"Date"}, inplace=True)
                df_ts_res = df_ts_res[["Date","Metro","alpha","beta"]]

                df_ts_res["AB_Chosen"] = df_ts_res[ab_choice]
                fig_ts = px.line(
                    df_ts_res,
                    x="Date",
                    y="AB_Chosen",
                    color="Metro",
                    title=f"Rolling {ab_choice.title()} {ts_start_ym}–{ts_end_ym}"
                )
                st.plotly_chart(fig_ts, use_container_width=True)
                st.dataframe(df_ts_res)

# ---------------------------------------------------------------------
# 7) LINE CHART: HISTORICAL + FORECASTED JOB GROWTH
# ---------------------------------------------------------------------
st.markdown("### Historical + Forecasted Job Growth (Line Chart)")

st.write("""
1. Select a **historical** date range of at least 5 years to estimate alpha/beta.  
2. Pick MSA(s).  
3. Input up to 3 future **monthly** growth scenarios for the **national** rate (e.g., "0.2" for +0.2% per month).  
4. Pick how many months forward to forecast.  
We'll plot **solid** lines for historical data and **dotted** lines for the forecast portion.
""")

hist_years = range(1990,2030)
lc_col1, lc_col2 = st.columns(2)
with lc_col1:
    hist_start_year = st.selectbox("Historical Start Year", hist_years, index=5)
    hist_start_month = st.selectbox("Historical Start Month", months, index=0)
with lc_col2:
    hist_end_year = st.selectbox("Historical End Year", hist_years, index=25)
    hist_end_month = st.selectbox("Historical End Month", months, index=11)

hist_smonth_num = months.index(hist_start_month)+1
hist_emonth_num = months.index(hist_end_month)+1
hist_start_ym = f"{hist_start_year:04d}-{hist_smonth_num:02d}"
hist_end_ym   = f"{hist_end_year:04d}-{hist_emonth_num:02d}"

line_msas = st.multiselect(
    "Pick MSA(s) to compare vs. National:",
    options=sorted(INVERTED_MAP.keys()),
    help="We'll plot each MSA's historical job growth + forecast lines if scenarios are provided."
)

st.write("#### Enter up to 3 monthly growth scenarios (%) for the forecast:")
sc_line1 = st.text_input("Scenario #1", value="0.2")
sc_line2 = st.text_input("Scenario #2 (optional)", value="0.5")
sc_line3 = st.text_input("Scenario #3 (optional)", value="1.0")

forecast_months = st.number_input("Months forward to forecast:", min_value=1, max_value=60, value=12)

def parse_scenario(sval):
    try:
        return float(sval)
    except:
        return None

if st.button("Generate Growth Chart"):
    # Validate 5-year
    start_dt = datetime.datetime(hist_start_year, hist_smonth_num, 1)
    end_dt   = datetime.datetime(hist_end_year, hist_emonth_num, 1)
    months_apart = (end_dt.year - start_dt.year)*12 + (end_dt.month - start_dt.month)
    if months_apart<60:
        st.error("Please select at least a 5-year historical range.")
    elif not line_msas:
        st.warning("No MSAs selected.")
    else:
        s_vals = []
        s1 = parse_scenario(sc_line1)
        s2 = parse_scenario(sc_line2)
        s3 = parse_scenario(sc_line3)
        if s1 is not None: s_vals.append(("Scenario #1", s1))
        if s2 is not None: s_vals.append(("Scenario #2", s2))
        if s3 is not None: s_vals.append(("Scenario #3", s3))

        chosen_ids_line = [INVERTED_MAP[m] for m in line_msas if m!="National"]
        chosen_ids_line.append(NATIONAL_SERIES_ID)
        chosen_ids_line = list(set(chosen_ids_line))

        df_raw_line = fetch_raw_data_multiple(chosen_ids_line, hist_start_ym, hist_end_ym)
        if df_raw_line.empty:
            st.warning("No data found. Check date range or CSV.")
        else:
            # Convert to monthly % growth
            df_raw_line["value"] = pd.to_numeric(df_raw_line["value"], errors="coerce")
            pivot_line = df_raw_line.pivot(index="obs_date", columns="series_id", values="value")
            pivot_line.sort_index(inplace=True)
            growth_line = pivot_line.pct_change(1)*100
            growth_line.dropna(inplace=True)

            if NATIONAL_SERIES_ID not in growth_line.columns:
                st.error("National series not found. Check data.")
            else:
                # alpha/beta each MSA
                results_list = []
                for sid in growth_line.columns:
                    if sid==NATIONAL_SERIES_ID:
                        continue
                    merged = growth_line[[NATIONAL_SERIES_ID,sid]].dropna()
                    if len(merged)<2:
                        continue
                    X = sm.add_constant(merged[NATIONAL_SERIES_ID])
                    y = merged[sid]
                    model = sm.OLS(y,X).fit()
                    alpha_val = model.params["const"]
                    beta_val  = model.params[NATIONAL_SERIES_ID]
                    r_sq      = model.rsquared
                    results_list.append({
                        "series_id": sid,
                        "Metro": MSA_NAME_MAP.get(sid,sid),
                        "Alpha": alpha_val,
                        "Beta": beta_val,
                        "R-Squared": r_sq
                    })
                df_ab_line = pd.DataFrame(results_list)
                st.write("#### Alpha/Beta Confidence Table")
                if df_ab_line.empty:
                    st.error("Insufficient data to compute alpha/beta.")
                else:
                    st.dataframe(df_ab_line[["Metro","Alpha","Beta","R-Squared"]])

                # Build historical portion
                line_records = []
                for dt_idx in growth_line.index:
                    for col_sid in growth_line.columns:
                        if col_sid==NATIONAL_SERIES_ID:
                            metro_name = "National"
                        else:
                            metro_name = MSA_NAME_MAP.get(col_sid,col_sid)
                        if (metro_name in line_msas) or (metro_name=="National"):
                            val = growth_line.loc[dt_idx,col_sid]
                            line_records.append({
                                "Date": dt_idx,
                                "Growth": val,
                                "Metro": metro_name,
                                "Scenario": "Historical",
                                "LineStyle": "solid"
                            })

                # Forecast portion
                last_dt = growth_line.index.max()
                forecast_dates = []
                for i in range(1, forecast_months+1):
                    future_date = last_dt + pd.DateOffset(months=i)
                    forecast_dates.append(future_date)

                for (sc_label,sc_val) in s_vals:
                    # National line
                    for fdt in forecast_dates:
                        line_records.append({
                            "Date": fdt,
                            "Growth": sc_val,
                            "Metro": "National",
                            "Scenario": sc_label,
                            "LineStyle": "dot"
                        })
                    # MSAs
                    for idx, rowm in df_ab_line.iterrows():
                        metro = rowm["Metro"]
                        alpha_v = rowm["Alpha"]
                        beta_v  = rowm["Beta"]
                        for fdt in forecast_dates:
                            forecast_msa_g = alpha_v + beta_v*sc_val
                            line_records.append({
                                "Date": fdt,
                                "Growth": forecast_msa_g,
                                "Metro": metro,
                                "Scenario": sc_label,
                                "LineStyle": "dot"
                            })

                df_line_final = pd.DataFrame(line_records)
                fig_line = px.line(
                    df_line_final,
                    x="Date",
                    y="Growth",
                    color="Metro",
                    line_dash="Scenario",
                    title="Job Growth: Historical (solid) + Forecast (dotted)"
                )
                # Force national line black if you want:
                # custom map or let plotly choose
                fig_line.update_traces(connectgaps=True)
                st.plotly_chart(fig_line, use_container_width=True)

                st.markdown("""
                **Notes:**  
                - Solid lines = Historical monthly growth  
                - Dotted lines = Forecast  
                - National forecast growth is whatever scenario you typed (e.g., 0.2% monthly)  
                - MSA forecast = alpha + beta * scenario  
                - "R-Squared" indicates how strongly MSA growth correlated with national growth in the historical window.
                """)



