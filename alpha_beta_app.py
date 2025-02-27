import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import datetime
import re

# ---------------------------------------------------------------------
# 0) Page Config Must Be FIRST
# ---------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="US Metro Analysis - Full MSA Map")

# ---------------------------------------------------------------------
# 1) Custom CSS
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
    div[data-baseweb="tag"] {
        background-color: #93D0EC !important;
        color: black !important;
        border-radius: 4px;
    }
    div.stButton > button {
        background-color: #93D0EC !important;
        color: black !important;
        border-radius: 4px;
        border: 1px solid #333333;
    }
    /* Pin the first column in st.dataframe */
    .stDataFrame table tbody tr td:nth-of-type(1),
    .stDataFrame table thead tr th:nth-of-type(1) {
        position: sticky;
        left: 0px;
        background-color: white;
        z-index: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# 2) Main Title
# ---------------------------------------------------------------------
st.title("US Metro Analysis")

# ---------------------------------------------------------------------
# 3) About the Data & How To Use
# ---------------------------------------------------------------------
with st.expander("About the Data", expanded=False):
    st.markdown(
        """
        **Data Source**  
        - All data is publicly available from the Bureau of Labor Statistics (BLS).  
        - This tool features the top 50 U.S. MSAs (Metro Statistical Area) by population, plus a “National” benchmark.

        **Alpha & Beta**  
        - **Alpha**: MSA’s baseline performance vs. national within the chosen date range.  
        - **Beta**: MSA’s sensitivity to national changes.  
        - **R-Squared**: Fit quality of alpha/beta model.

        **Rolling 12-Month Time Series**  
        - Each monthly alpha/beta is derived from that month + previous 11 months (12 total).  
        - More responsive to new data than a single full-period OLS.
        """
    )

with st.expander("How To Use", expanded=False):
    st.markdown(
        """
        1. **Select a metric** (currently one).  
        2. **XY Chart (Alpha vs Beta)**: pick MSAs + date range.  
        3. **Time Series (Rolling 12-Month)**: up to 5 MSAs + date range → monthly alpha/beta lines.  
        4. **Historical Growth & Forecasts (All MSAs)**: yoy data, plus scenario forecasts.  
        5. **Single MSA**: yoy vs. national, plus forecast scenarios.
        """
    )

# ---------------------------------------------------------------------
# 4) Load CSV & Force 'value' to be Numeric
# ---------------------------------------------------------------------
CSV_URL = "https://raw.githubusercontent.com/Ryangab8/InquiRE/main/raw_nonfarm_jobs.csv"
df_full = pd.read_csv(CSV_URL)

df_full["obs_date"] = pd.to_datetime(df_full["obs_date"], errors="coerce")

def strip_non_digits(val):
    # remove commas or other characters
    return re.sub(r"[^0-9.\-eE]", "", str(val))

df_full["value"] = df_full["value"].apply(strip_non_digits)
df_full["value"] = pd.to_numeric(df_full["value"], errors="coerce")

NATIONAL_SERIES_ID = "CES0000000001"

# ---------------------------------------------------------------------
# Full MSA Map
# ---------------------------------------------------------------------
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
# 5) Metric Selector
# ---------------------------------------------------------------------
st.markdown("#### Select a Metric")
metric_choice = st.selectbox(
    "Pick a metric:",
    ["Total NonFarm Employment - Seasonally Adjusted"]
)
st.write(f"You selected: **{metric_choice}**")

# ---------------------------------------------------------------------
# 6) Utility Functions
# ---------------------------------------------------------------------
def fetch_raw_data_multiple(msa_ids, start_ym, end_ym):
    if NATIONAL_SERIES_ID not in msa_ids:
        msa_ids.append(NATIONAL_SERIES_ID)
    syear, smonth = map(int, start_ym.split("-"))
    eyear, emonth = map(int, end_ym.split("-"))
    start_dt = datetime.datetime(syear, smonth, 1)
    end_dt   = datetime.datetime(eyear, emonth, 1)
    return df_full[
        (df_full["series_id"].isin(msa_ids)) &
        (df_full["obs_date"] >= start_dt) &
        (df_full["obs_date"] <= end_dt)
    ].copy()

def compute_multi_alpha_beta(df_raw):
    df_raw["value"] = pd.to_numeric(df_raw["value"], errors="coerce")
    pivoted = df_raw.pivot(index="obs_date", columns="series_id", values="value")
    growth = pivoted.pct_change(1)*100
    growth.dropna(inplace=True)
    if NATIONAL_SERIES_ID not in growth.columns:
        return pd.DataFrame()
    rows = []
    for col in growth.columns:
        if col == NATIONAL_SERIES_ID:
            continue
        sub = growth[[NATIONAL_SERIES_ID, col]].dropna()
        if len(sub) < 2:
            continue
        X = sm.add_constant(sub[NATIONAL_SERIES_ID])
        y = sub[col]
        model = sm.OLS(y, X).fit()
        alpha_v = model.params.get("const", None)
        slope_keys = [k for k in model.params if k != "const"]
        beta_v = model.params[slope_keys[0]] if len(slope_keys) == 1 else None
        rsq_val = model.rsquared
        rows.append({
            "Metro": MSA_NAME_MAP.get(col, col),
            "Alpha": alpha_v,
            "Beta": beta_v,
            "R-Squared": rsq_val
        })
    return pd.DataFrame(rows)

def compute_alpha_beta_subset(df_subset, nat_col, msa_col):
    if len(df_subset.dropna()) < 2:
        return None, None
    X = sm.add_constant(df_subset[nat_col])
    y = df_subset[msa_col]
    model = sm.OLS(y, X).fit()
    alpha_v = model.params.get("const", None)
    slope_keys = [k for k in model.params if k != "const"]
    beta_v = model.params[slope_keys[0]] if len(slope_keys) == 1 else None
    return alpha_v, beta_v

def compute_rolling_alpha_beta_time_series(df_raw_ts, start_ym_ts, end_ym_ts):
    """
    Rolling 12-month alpha/beta (no debug prints, includes sort_index).
    """
    df_pivot = df_raw_ts.pivot(index="obs_date", columns="series_id", values="value")
    df_growth = df_pivot.pct_change(1)*100
    df_growth.dropna(how="all", inplace=True)
    df_growth.sort_index(inplace=True)  # ensure chronological order

    if NATIONAL_SERIES_ID not in df_growth.columns:
        return pd.DataFrame()

    start_dt = pd.to_datetime(f"{start_ym_ts}-01")
    end_dt   = pd.to_datetime(f"{end_ym_ts}-01")

    months_sorted = sorted(df_growth.index.unique())
    final_rows = []

    for this_month in months_sorted:
        if this_month < start_dt or this_month > end_dt:
            continue
        rolling_start = this_month - pd.DateOffset(months=11)
        window_df = df_growth.loc[(df_growth.index >= rolling_start) & (df_growth.index <= this_month)]
        if len(window_df) < 2:
            continue

        for col in df_growth.columns:
            if col == NATIONAL_SERIES_ID:
                continue
            sub_df = window_df[[NATIONAL_SERIES_ID, col]].dropna()
            alpha_v, beta_v = compute_alpha_beta_subset(sub_df, NATIONAL_SERIES_ID, col)
            if alpha_v is not None and beta_v is not None:
                final_rows.append({
                    "obs_date": this_month,
                    "series_id": col,
                    "alpha": alpha_v,
                    "beta": beta_v
                })

    return pd.DataFrame(final_rows)

# ---------------------------------------------------------------------
# 7) XY Chart (Alpha vs. Beta)
# ---------------------------------------------------------------------
st.markdown("### XY Chart (Alpha vs Beta)")

if "xy_msas" not in st.session_state:
    st.session_state["xy_msas"] = []
if "xy_df" not in st.session_state:
    st.session_state["xy_df"] = None

def select_all():
    st.session_state["xy_msas"] = sorted(INVERTED_MAP.keys())

def clear_all():
    st.session_state["xy_msas"] = []

xy_all_msas = sorted(INVERTED_MAP.keys())
st.multiselect("Pick MSA(s):", options=xy_all_msas, key="xy_msas")

colXY1, colXY2 = st.columns(2)
colXY1.button("Select All", on_click=select_all)
colXY2.button("Clear", on_click=clear_all)

months_list = ["January","February","March","April","May","June","July","August","September","October","November","December"]
years_list = list(range(1990,2025))
xy_start_dyear = 2019
xy_end_dyear   = 2024

st.write("#### Date Range for XY Chart")
colA, colB = st.columns(2)
with colA:
    xy_start_month = st.selectbox("Start Month (XY)", months_list, index=0)
    xy_start_year  = st.selectbox("Start Year (XY)", years_list, index=years_list.index(xy_start_dyear))
with colB:
    xy_end_month = st.selectbox("End Month (XY)", months_list, index=11)
    xy_end_year  = st.selectbox("End Year (XY)", years_list, index=years_list.index(xy_end_dyear))

xy_sm = months_list.index(xy_start_month)+1
xy_em = months_list.index(xy_end_month)+1
xy_start_ym = f"{xy_start_year:04d}-{xy_sm:02d}"
xy_end_ym   = f"{xy_end_year:04d}-{xy_em:02d}"

if st.button("Generate XY Chart"):
    if not st.session_state["xy_msas"]:
        st.warning("No MSAs selected!")
    else:
        chosen_xy_ids = [INVERTED_MAP[m] for m in st.session_state["xy_msas"]]
        df_xy = fetch_raw_data_multiple(chosen_xy_ids, xy_start_ym, xy_end_ym)
        if df_xy.empty:
            st.warning("No data found in that range.")
        else:
            ab_df = compute_multi_alpha_beta(df_xy)
            if ab_df.empty:
                st.error("Could not compute alpha/beta.")
            else:
                st.session_state["xy_df"] = ab_df
                title_xy = f"Alpha vs Beta ({xy_start_ym} to {xy_end_ym}) - {metric_choice}"
                fig_xy = px.scatter(
                    ab_df, x="Beta", y="Alpha", text="Metro",
                    title=title_xy, render_mode="webgl"
                )
                fig_xy.update_traces(textposition='top center')
                fig_xy.update_layout(
                    dragmode='pan',
                    title_x=0.5,
                    title_xanchor='center',
                    xaxis=dict(fixedrange=False),
                    yaxis=dict(fixedrange=False)
                )
                fig_xy.add_hline(y=0, line_width=2, line_color="black", line_dash="dot")
                fig_xy.add_vline(x=1, line_width=2, line_color="black", line_dash="dot")
                st.plotly_chart(fig_xy, use_container_width=True, config={"scrollZoom": True})

if st.session_state["xy_df"] is not None:
    if st.checkbox("View alpha/beta table for XY chart"):
        st.dataframe(st.session_state["xy_df"])

# ---------------------------------------------------------------------
# 8) TIME SERIES (Rolling 12-Month Alpha/Beta)
# ---------------------------------------------------------------------
st.markdown("### Time Series (Rolling 12-Month Alpha/Beta)")

if "df_ts" not in st.session_state:
    st.session_state["df_ts"] = None
if "fig_ts" not in st.session_state:
    st.session_state["fig_ts"] = None

all_msas_no_nat = [m for m in xy_all_msas if m != "National"]
sel_time_msas = st.multiselect(
    "Pick up to 5 MSAs for Time Series:",
    options=all_msas_no_nat,
    max_selections=5
)

ab_pick = st.selectbox("Which metric to graph in the Time Series?", ["alpha", "beta"])

ts_def_start = 2019
ts_def_end   = 2024

st.write("#### Date Range for Time Series (Rolling Window)")
cT1, cT2 = st.columns(2)
with cT1:
    ts_start_month = st.selectbox("Start Month (Time Series)", months_list, index=0)
    ts_start_year  = st.selectbox("Start Year (Time Series)", years_list, index=years_list.index(ts_def_start))
with cT2:
    ts_end_month = st.selectbox("End Month (Time Series)", months_list, index=11)
    ts_end_year  = st.selectbox("End Year (Time Series)", years_list, index=years_list.index(ts_def_end))

ts_smon = months_list.index(ts_start_month)+1
ts_emon = months_list.index(ts_end_month)+1
ts_start_ym = f"{ts_start_year:04d}-{ts_smon:02d}"
ts_end_ym   = f"{ts_end_year:04d}-{ts_emon:02d}"

if st.button("Compute Time Series"):
    if not sel_time_msas:
        st.warning("Pick at least 1 MSA.")
    else:
        chosen_ts_ids = [INVERTED_MAP[m] for m in sel_time_msas]
        df_raw_ts = fetch_raw_data_multiple(chosen_ts_ids, "1990-01", ts_end_ym)
        if df_raw_ts.empty:
            st.warning("No data found. Check your CSV or chosen date range.")
        else:
            df_ts_res = compute_rolling_alpha_beta_time_series(df_raw_ts, ts_start_ym, ts_end_ym)
            if df_ts_res.empty:
                st.warning("No rolling alpha/beta computed.")
            else:
                df_ts_res["Metro"] = df_ts_res["series_id"].apply(lambda sid: MSA_NAME_MAP.get(sid, sid))
                df_ts_res.drop(columns=["series_id"], inplace=True)
                df_ts_res.rename(columns={"obs_date": "Date"}, inplace=True)
                df_ts_res = df_ts_res[["Date","Metro","alpha","beta"]]

                st.session_state["df_ts"] = df_ts_res

                plot_ts = df_ts_res.copy()
                plot_ts["AB_Chosen"] = plot_ts[ab_pick]
                title_ts = f"Time Series of {ab_pick.title()} (Rolling 12-Month) {ts_start_ym}–{ts_end_ym}"
                fig_ts = px.line(
                    plot_ts, x="Date", y="AB_Chosen", color="Metro",
                    title=title_ts
                )
                fig_ts.update_layout(
                    dragmode='pan',
                    xaxis_title="Date",
                    yaxis_title=ab_pick.title(),
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
# 9) HISTORICAL GROWTH AND FORECASTS (All MSAs)
# ---------------------------------------------------------------------
st.markdown("### Historical Growth and Forecasts")

st.write("""
View yoy comparisons for all MSAs vs. national, plus scenario forecasts using alpha/beta.
""")

if "all_msa_df" not in st.session_state:
    st.session_state["all_msa_df"] = None

hist_yrs = list(range(1990,2025))
hcA, hcB = st.columns(2)
with hcA:
    yoy_table_start = st.selectbox("All-MSA Start Year", hist_yrs, index=hist_yrs.index(2015), key="allmsa_startyear2")
with hcB:
    yoy_table_end = st.selectbox("All-MSA End Year", hist_yrs, index=hist_yrs.index(2024), key="allmsa_endyear2")

if yoy_table_end < yoy_table_start + 5:
    st.warning("Please pick at least a 5-year window (end >= start+5).")

st.markdown("#### Optional – Forecast year over year National growth rate (%)")
tablef1 = st.text_input("Forecast #1", value="1.0")
tablef2 = st.text_input("Forecast #2", value="2.5")
tablef3 = st.text_input("Forecast #3", value="")

def parse_fore_val(v):
    try:
        return float(v)
    except:
        return None

scenaries = []
for lbl, val_ in [("Forecast #1",tablef1),("Forecast #2",tablef2),("Forecast #3",tablef3)]:
    tv = parse_fore_val(val_)
    if tv is not None:
        scenaries.append((lbl, tv))

diff_mode = st.checkbox("Show National vs Metro Variance (MSA minus National)?", value=False)

def get_alpha_beta_rsq_yoy(sid, ylist, yoy_map):
    xvals, yvals = [], []
    for y in ylist:
        nval = yoy_map[NATIONAL_SERIES_ID].get(y, None)
        mval = yoy_map[sid].get(y, None)
        if (nval is not None) and (mval is not None):
            xvals.append(nval)
            yvals.append(mval)
    if len(xvals) < 2:
        return (None, None, None)
    Xdf = pd.DataFrame({"nat": xvals})
    Xdf = sm.add_constant(Xdf, prepend=True)
    model = sm.OLS(yvals, Xdf).fit()
    a_val = model.params.get("const", None)
    slope_keys = [kk for kk in model.params if kk != "const"]
    b_val = model.params[slope_keys[0]] if len(slope_keys) == 1 else None
    rsq_ = model.rsquared
    return (a_val, b_val, rsq_)

if st.button("Generate Table"):
    if yoy_table_end < yoy_table_start + 5:
        st.error("End year must be >= start+5.")
        st.stop()

    dtmp = df_full.copy()
    dtmp["year"] = dtmp["obs_date"].dt.year
    dtmp["month"] = dtmp["obs_date"].dt.month
    d_jan = dtmp[dtmp["month"] == 1]
    d_jan = d_jan[(d_jan["year"] >= yoy_table_start) & (d_jan["year"] <= yoy_table_end)]
    if d_jan.empty:
        st.warning("No january data in that range.")
        st.stop()

    pivot_jan = d_jan.pivot(index="year", columns="series_id", values="value")
    pivot_jan.dropna(how="all", inplace=True)
    sorted_years = sorted(pivot_jan.index)
    all_sids = pivot_jan.columns.unique().tolist()

    yoy_map = {}
    for sid in all_sids:
        yoy_map[sid] = {}
        for yy in sorted_years:
            py = yy - 1
            if py in pivot_jan.index:
                val_curr = pivot_jan.loc[yy, sid]
                val_prev = pivot_jan.loc[py, sid]
                if pd.notnull(val_curr) and pd.notnull(val_prev) and val_prev != 0:
                    yoyval = 100*(val_curr - val_prev)/val_prev
                    yoy_map[sid][yy] = yoyval

    yoy_years_list = [y for y in range(yoy_table_start+1, yoy_table_end+1)]
    yoy_cols = [str(y) for y in yoy_years_list]
    fore_cols = [x[0] for x in scenaries]
    base_cols = ["Metro"] + yoy_cols + fore_cols + ["R-Squared"]

    rowdata = []
    natrow = {"Metro": "National", "R-Squared": None}
    for y in yoy_years_list:
        v_n = yoy_map[NATIONAL_SERIES_ID].get(y, None)
        natrow[str(y)] = v_n
    for (lbl, valf) in scenaries:
        if diff_mode:
            natrow[lbl] = 0.0
        else:
            natrow[lbl] = valf
    rowdata.append(natrow)

    all_msas_list = [k for k in MSA_NAME_MAP if k != NATIONAL_SERIES_ID]
    sorted_msas = sorted(all_msas_list, key=lambda s: MSA_NAME_MAP[s])
    for sid in sorted_msas:
        name = MSA_NAME_MAP[sid]
        alph, beta, rsq_ = get_alpha_beta_rsq_yoy(sid, yoy_years_list, yoy_map)
        rowdict = {"Metro": name, "R-Squared": rsq_}
        for y in yoy_years_list:
            yoy_nat = yoy_map[NATIONAL_SERIES_ID].get(y, None)
            yoy_msa = yoy_map[sid].get(y, None)
            if yoy_nat is None or yoy_msa is None:
                rowdict[str(y)] = None
            else:
                if diff_mode:
                    rowdict[str(y)] = yoy_msa - yoy_nat
                else:
                    rowdict[str(y)] = yoy_msa
        if alph is not None and beta is not None:
            for (lbl, valf) in scenaries:
                yoy_msa_fore = alph + beta*valf
                rowdict[lbl] = (yoy_msa_fore - valf) if diff_mode else yoy_msa_fore
        else:
            for (lbl, valf) in scenaries:
                rowdict[lbl] = None
        rowdata.append(rowdict)

    df_all_msa = pd.DataFrame(rowdata)
    df_all_msa = df_all_msa[base_cols]

    def color_func(val):
        if val is None:
            return ""
        if isinstance(val, (int, float)):
            if val > 0:
                return "background-color: rgba(76,175,80,0.4);"
            elif val < 0:
                return "background-color: rgba(255,0,0,0.3);"
        return ""

    style_cols = yoy_cols + fore_cols
    styled_tab = df_all_msa.style.format(
        subset=style_cols, formatter="{:.2f}"
    ).format(
        subset=["R-Squared"], formatter="{:.3f}"
    ).applymap(color_func, subset=style_cols)
    st.session_state["all_msa_df"] = styled_tab

if "all_msa_df" in st.session_state and st.session_state["all_msa_df"] is not None:
    st.dataframe(st.session_state["all_msa_df"], use_container_width=True)

# ---------------------------------------------------------------------
# 10) SINGLE MSA Comparative Year Over Year
# ---------------------------------------------------------------------
st.markdown("### Single MSA Comparative Year Over Year Growth")
st.write("""
Compare national vs MSA yoy growth. Enter forecast scenarios to see MSA’s yoy projection.
""")

if "single_msa_df" not in st.session_state:
    st.session_state["single_msa_df"] = None

single_yrs_list = list(range(1990,2025))
cc1, cc2 = st.columns(2)
with cc1:
    single_start_yr = st.selectbox("Single MSA Start Year", single_yrs_list, index=single_yrs_list.index(2015), key="singlemsastart")
with cc2:
    single_end_yr = st.selectbox("Single MSA End Year", single_yrs_list, index=single_yrs_list.index(2024), key="singlemsaend")

single_msa_pick = st.selectbox("Select an MSA:", sorted(INVERTED_MAP.keys()))

st.markdown("#### Optional – Forecast year over year National growth rate (%)")
sf1 = st.text_input("Scenario #1", value="1.0")
sf2 = st.text_input("Scenario #2", value="2.5")
sf3 = st.text_input("Scenario #3", value="")

def parse_sing_forecast(val):
    try:
        return float(val)
    except:
        return None

sing_scenarios = []
for lbl, vstr in [("Scenario #1", sf1), ("Scenario #2", sf2), ("Scenario #3", sf3)]:
    fv = parse_sing_forecast(vstr)
    if fv is not None:
        sing_scenarios.append((lbl, fv))

if st.button("Generate Single-MSA YOY Chart"):
    if single_end_yr < single_start_yr:
        st.error("End year < start year not valid.")
        st.stop()

    tmp_df = df_full.copy()
    tmp_df["year"] = tmp_df["obs_date"].dt.year
    tmp_df["month"] = tmp_df["obs_date"].dt.month
    tmp_jan = tmp_df[tmp_df["month"] == 1]
    tmp_jan = tmp_jan[(tmp_jan["year"] >= single_start_yr) & (tmp_jan["year"] <= single_end_yr)]
    if tmp_jan.empty:
        st.warning("No january data in that range.")
        st.stop()

    sid_msa = INVERTED_MAP[single_msa_pick]
    needed_ids = [sid_msa, NATIONAL_SERIES_ID]
    tmp_jan = tmp_jan[tmp_jan["series_id"].isin(needed_ids)]
    pivot_sing = tmp_jan.pivot(index="year", columns="series_id", values="value")
    pivot_sing.dropna(inplace=True)
    yoy_sing_list = []
    s_yrs_ = sorted(pivot_sing.index)
    for y_ in s_yrs_:
        py_ = y_ - 1
        if py_ in pivot_sing.index:
            nat_val_t  = pivot_sing.loc[y_, NATIONAL_SERIES_ID]
            nat_val_m1 = pivot_sing.loc[py_, NATIONAL_SERIES_ID]
            msa_val_t  = pivot_sing.loc[y_, sid_msa]
            msa_val_m1 = pivot_sing.loc[py_, sid_msa]
            if (pd.notnull(nat_val_t) and pd.notnull(nat_val_m1) and nat_val_m1 != 0
               and pd.notnull(msa_val_t) and pd.notnull(msa_val_m1) and msa_val_m1 != 0):
                yoy_nat = 100*(nat_val_t - nat_val_m1)/nat_val_m1
                yoy_msa = 100*(msa_val_t - msa_val_m1)/msa_val_m1
                yoy_sing_list.append({
                    "Year": y_,
                    "Nat_Growth": yoy_nat,
                    "MSA_Growth": yoy_msa
                })

    df_sing_yoy = pd.DataFrame(yoy_sing_list)
    if df_sing_yoy.empty:
        st.warning("Not enough consecutive january data for yoy.")
        st.stop()

    # OLS for alpha/beta
    Xdf = sm.add_constant(df_sing_yoy["Nat_Growth"], prepend=True)
    yvals = df_sing_yoy["MSA_Growth"]
    model = sm.OLS(yvals, Xdf).fit()
    alpha_v, beta_v, rsq_v = None, None, None
    try:
        alpha_v = model.params["const"]
        slope_keys = [p for p in model.params.index if p != "const"]
        if len(slope_keys) == 1:
            beta_v = model.params[slope_keys[0]]
        rsq_v = model.rsquared
    except:
        alpha_v, beta_v, rsq_v = None, None, None

    # Build forecast
    forecast_list = []
    df_plot_sing = df_sing_yoy.copy()
    for (lbl, fval) in sing_scenarios:
        yoy_msa_fore = alpha_v + beta_v*fval if (alpha_v is not None and beta_v is not None) else None
        forecast_list.append({
            "Year": lbl,
            "Nat_Growth": fval,
            "MSA_Growth": yoy_msa_fore
        })
    if forecast_list:
        df_plot_sing = pd.concat([df_sing_yoy, pd.DataFrame(forecast_list)], ignore_index=True)

    bar_data = []
    for _, row_ in df_plot_sing.iterrows():
        bar_data.append({"Year": row_["Year"], "Series": "National", "Growth": row_["Nat_Growth"]})
        bar_data.append({"Year": row_["Year"], "Series": single_msa_pick, "Growth": row_["MSA_Growth"]})
    df_bar_sing = pd.DataFrame(bar_data)
    title_sing = f"Year-over-Year Growth (Jan) — National vs {single_msa_pick}"
    fig_sing = px.bar(df_bar_sing, x="Year", y="Growth", color="Series", barmode="group", title=title_sing)
    fig_sing.update_layout(xaxis_type='category')
    fig_sing.update_layout(
        xaxis=dict(fixedrange=False),
        yaxis=dict(fixedrange=False)
    )
    st.plotly_chart(fig_sing, use_container_width=True, config={"scrollZoom": True})

    if "single_msa_df" not in st.session_state:
        st.session_state["single_msa_df"] = None
    st.session_state["single_msa_df"] = df_plot_sing

    # OLS summary
    st.markdown("#### Summary Statistics")
    df_ols_sing = pd.DataFrame([{
        "MSA": single_msa_pick,
        "Alpha": alpha_v,
        "Beta": beta_v,
        "R-Squared": rsq_v
    }])
    st.dataframe(df_ols_sing)

    st.markdown("#### Interpretation")
    if metric_choice.startswith("Total NonFarm Employment"):
        st.write("For historical YoY context, see FRED chart: [Historical NonFarm Employment YoY](https://fred.stlouisfed.org/graph/?g=1DRDw)")

    if alpha_v is not None and beta_v is not None:
        lines = []
        for (lbl, fval) in sing_scenarios:
            yoy_msa_calc = alpha_v + beta_v*fval
            lines.append(f"For {lbl} (National={fval:.1f}%), MSA yoy ~ {yoy_msa_calc:.1f}%.")
        if lines:
            st.write("Based on alpha/beta, scenario implications:")
            for l_ in lines:
                st.write("-", l_)
        else:
            st.write("No valid forecast entries.")
    else:
        st.write("Insufficient data for alpha/beta on this MSA.")

if "single_msa_df" in st.session_state and st.session_state["single_msa_df"] is not None:
    if st.checkbox("View data table"):
        st.dataframe(st.session_state["single_msa_df"])
