import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import datetime

# ---------------------------------------------------------------------
# 0) Page Config Must Be FIRST
# ---------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="All-in-One US Metro Alpha/Beta App")

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
        --primary-color: #93D0EC; /* Light blue brand color */
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
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# 2) Main Title & Intro
# ---------------------------------------------------------------------
st.title("US Metro Alpha/Beta Analysis - Unified App")

with st.expander("About the Data", expanded=False):
    st.markdown(
        """
        **Data Source**  
        - All data is publicly available from the Bureau of Labor Statistics (BLS).  
        - This tool features the top 50 U.S. MSAs by population, plus a “National” benchmark.

        **Alpha**  
        - Measures a MSA’s baseline growth if national growth=0. Positive means outperformance, negative means underperformance.

        **Beta**  
        - Measures MSA volatility relative to national changes. Beta>1 → amplifies national, Beta<1 → more stable.

        **Sections**  
        1. XY Chart (Alpha vs Beta)  
        2. Rolling 12-Month Alpha/Beta Time Series  
        3. All-MSA YOY Table (Difference or Raw), plus Forecasts  
        4. Year-over-Year Bar Chart (Single MSA)  
        """
    )

# ---------------------------------------------------------------------
# 3) Load CSV from GitHub
# ---------------------------------------------------------------------
GITHUB_CSV_URL = "https://raw.githubusercontent.com/Ryangab8/InquiRE/main/raw_nonfarm_jobs.csv"
df_full = pd.read_csv(GITHUB_CSV_URL)
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
INVERTED_MAP = {v: k for k,v in MSA_NAME_MAP.items()}

# ---------------------------------------------------------------------
# 4) Utility Functions
# ---------------------------------------------------------------------
import math

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
    df_growth = df_pivot.pct_change(1)*100
    df_growth.dropna(inplace=True)

    if NATIONAL_SERIES_ID not in df_growth.columns:
        return pd.DataFrame()

    results = []
    for col in df_growth.columns:
        if col==NATIONAL_SERIES_ID:
            continue
        merged = df_growth[[NATIONAL_SERIES_ID,col]].dropna()
        if len(merged)<2:
            continue
        X = sm.add_constant(merged[NATIONAL_SERIES_ID])
        y = merged[col]
        model = sm.OLS(y, X).fit()
        alpha_val = model.params["const"]
        beta_val  = model.params[X.columns[-1]]
        r_sq      = model.rsquared
        results.append({
            "series_id":col,
            "Alpha": alpha_val,
            "Beta": beta_val,
            "R-Squared": r_sq
        })
    df_ab = pd.DataFrame(results)
    if df_ab.empty:
        return df_ab
    df_ab["Metro"] = df_ab["series_id"].apply(lambda sid: MSA_NAME_MAP.get(sid,sid))
    df_ab.drop(columns=["series_id"], inplace=True)
    df_ab = df_ab[["Metro","Alpha","Beta","R-Squared"]]
    return df_ab

def compute_alpha_beta_subset(df_subset,nat_col,msa_col):
    if len(df_subset.dropna())<2:
        return None,None
    X = sm.add_constant(df_subset[nat_col])
    y = df_subset[msa_col]
    model = sm.OLS(y,X).fit()
    return model.params["const"], model.params[X.columns[-1]]

ROLLING_WINDOW_MONTHS=12
def compute_rolling_alpha_beta_time_series(df_raw_ts, start_ym_ts, end_ym_ts):
    df_pivot = df_raw_ts.pivot(index="obs_date", columns="series_id", values="value")
    df_growth = df_pivot.pct_change(1)*100
    df_growth.dropna(inplace=True)

    if NATIONAL_SERIES_ID not in df_growth.columns:
        return pd.DataFrame()

    start_dt = pd.to_datetime(f"{start_ym_ts}-01")
    end_dt   = pd.to_datetime(f"{end_ym_ts}-01")
    unique_months_ts = sorted(df_growth.index.unique())

    results_ts=[]
    for current_month in unique_months_ts:
        if current_month<start_dt or current_month>end_dt:
            continue
        rolling_start = current_month - pd.DateOffset(months=ROLLING_WINDOW_MONTHS-1)
        df_window = df_growth.loc[(df_growth.index>=rolling_start)&(df_growth.index<=current_month)]
        if len(df_window)<2:
            continue
        for col in df_window.columns:
            if col==NATIONAL_SERIES_ID:
                continue
            subset = df_window[[NATIONAL_SERIES_ID,col]].dropna()
            alpha_val,beta_val = compute_alpha_beta_subset(subset,NATIONAL_SERIES_ID,col)
            if alpha_val is not None and beta_val is not None:
                results_ts.append({
                    "obs_date":current_month,
                    "series_id":col,
                    "alpha":alpha_val,
                    "beta":beta_val
                })
    return pd.DataFrame(results_ts)


# ---------------------------------------------------------------------
# 5) XY CHART (Alpha vs. Beta)
# ---------------------------------------------------------------------
st.markdown("### XY Chart (Alpha vs. Beta)")

def select_all():
    st.session_state["msa_selection"]=sorted(INVERTED_MAP.keys())
def clear_all():
    st.session_state["msa_selection"]=[]
if "msa_selection" not in st.session_state:
    st.session_state["msa_selection"]=[]
if "df_ab" not in st.session_state:
    st.session_state["df_ab"]=None
if "fig" not in st.session_state:
    st.session_state["fig"]=None

all_msa_names = sorted(INVERTED_MAP.keys())
st.multiselect(
    "Pick MSA(s):",
    options=all_msa_names,
    key="msa_selection"
)

col_xy1, col_xy2=st.columns(2)
col_xy1.button("Select All",on_click=select_all)
col_xy2.button("Clear",on_click=clear_all)

months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
years_xy=list(range(1990,2026))
default_start_year=2019
default_end_year=2024

st.write("#### Date Range for XY Chart")
cx1,cx2=st.columns(2)
with cx1:
    xy_start_month=st.selectbox("Start Month (XY)",months,index=0)
    xy_start_year =st.selectbox("Start Year (XY)",years_xy,index=years_xy.index(default_start_year))
with cx2:
    xy_end_month=st.selectbox("End Month (XY)",months,index=11)
    xy_end_year =st.selectbox("End Year (XY)",years_xy,index=years_xy.index(default_end_year))

xy_smonth_num=months.index(xy_start_month)+1
xy_emonth_num=months.index(xy_end_month)+1
xy_start_ym   =f"{xy_start_year:04d}-{xy_smonth_num:02d}"
xy_end_ym     =f"{xy_end_year:04d}-{xy_emonth_num:02d}"

if st.button("Generate XY Chart"):
    if not st.session_state["msa_selection"]:
        st.warning("No MSAs selected!")
    else:
        chosen_ids=[INVERTED_MAP[m] for m in st.session_state["msa_selection"]]
        df_raw=fetch_raw_data_multiple(chosen_ids,xy_start_ym,xy_end_ym)
        if df_raw.empty:
            st.warning("No data found for that range.")
        else:
            df_ab=compute_multi_alpha_beta(df_raw)
            if df_ab.empty:
                st.error("No alpha/beta computed. Possibly insufficient overlap.")
            else:
                title_xy=f"Alpha vs. Beta ({xy_start_ym} to {xy_end_ym})"
                fig_xy=px.scatter(
                    df_ab,x="Beta",y="Alpha",text="Metro",title=title_xy,render_mode="webgl"
                )
                fig_xy.update_traces(textposition='top center',textfont_size=14)
                fig_xy.update_layout(dragmode='pan',title_x=0.5,title_xanchor='center')
                fig_xy.add_hline(
                    y=0,line_width=3,line_color="black",line_dash="dot",
                    annotation_text="Alpha = 0",annotation_position="top left"
                )
                fig_xy.add_vline(
                    x=1,line_width=3,line_color="black",line_dash="dot",
                    annotation_text="Beta = 1",annotation_position="bottom right"
                )
                st.session_state["df_ab"]=df_ab
                st.session_state["fig"]=fig_xy

if st.session_state["fig"] is not None:
    st.plotly_chart(st.session_state["fig"],use_container_width=True,config={"scrollZoom":True})
    if st.checkbox("View alpha/beta table for XY chart"):
        st.dataframe(st.session_state["df_ab"])

# ---------------------------------------------------------------------
# 6) TIME SERIES (Rolling 12-Month Alpha/Beta)
# ---------------------------------------------------------------------
st.markdown("### Time Series (Rolling 12-Month Alpha/Beta)")

if "df_ts" not in st.session_state:
    st.session_state["df_ts"]=None
if "fig_ts" not in st.session_state:
    st.session_state["fig_ts"]=None

all_msa_names_no_nat=[m for m in all_msa_names if m!="National"]
selected_time_msas=st.multiselect("Pick up to 5 MSAs:",options=all_msa_names_no_nat,max_selections=5)
ab_choice=st.selectbox("Which metric to plot?",["alpha","beta"])
ts_default_start_year=2019
ts_default_end_year=2024

st.write("#### Date Range for Time Series (Rolling Window)")
ct1,ct2=st.columns(2)
with ct1:
    ts_start_month=st.selectbox("Start Month (Time Series)",months,index=0)
    ts_start_year =st.selectbox("Start Year (Time Series)",years_xy,index=years_xy.index(ts_default_start_year))
with ct2:
    ts_end_month=st.selectbox("End Month (Time Series)",months,index=11)
    ts_end_year =st.selectbox("End Year (Time Series)",years_xy,index=years_xy.index(ts_default_end_year))

ts_smonth_num=months.index(ts_start_month)+1
ts_emonth_num=months.index(ts_end_month)+1
ts_start_ym   =f"{ts_start_year:04d}-{ts_smonth_num:02d}"
ts_end_ym     =f"{ts_end_year:04d}-{ts_emonth_num:02d}"

if st.button("Compute Rolling Time Series"):
    if not selected_time_msas:
        st.warning("Pick at least 1 MSA.")
    else:
        chosen_time_ids=[INVERTED_MAP[n] for n in selected_time_msas]
        df_raw_ts=fetch_raw_data_multiple(chosen_time_ids,"1990-01",ts_end_ym)
        if df_raw_ts.empty:
            st.warning("No data found. Check range or CSV.")
        else:
            df_ts_result=compute_rolling_alpha_beta_time_series(df_raw_ts,ts_start_ym,ts_end_ym)
            if df_ts_result.empty:
                st.warning("No rolling alpha/beta computed.")
            else:
                df_ts_result["Metro"]=df_ts_result["series_id"].apply(lambda sid: MSA_NAME_MAP.get(sid,sid))
                df_ts_result.drop(columns=["series_id"],inplace=True)
                df_ts_result.rename(columns={"obs_date":"Date"},inplace=True)
                df_ts_result=df_ts_result[["Date","Metro","alpha","beta"]]
                st.session_state["df_ts"]=df_ts_result
                df_plot=df_ts_result.copy()
                df_plot["AB_Chosen"]=df_plot[ab_choice]
                title_ts=f"Time Series of {ab_choice.title()} (Rolling 12-Month) {ts_start_ym}–{ts_end_ym}"
                fig_ts=px.line(df_plot,x="Date",y="AB_Chosen",color="Metro",title=title_ts)
                fig_ts.update_layout(
                    dragmode='pan',
                    xaxis_title="Date",
                    yaxis_title=ab_choice.title(),
                    title_x=0.5,
                    title_xanchor="center"
                )
                st.session_state["fig_ts"]=fig_ts

if st.session_state["fig_ts"] is not None:
    st.plotly_chart(st.session_state["fig_ts"],use_container_width=True,config={"scrollZoom":True})
    if st.checkbox("Show rolling data table"):
        st.dataframe(st.session_state["df_ts"])

# ---------------------------------------------------------------------
# 7) ALL-MSA YOY TABLE (Difference or Raw) + Forecasts
# ---------------------------------------------------------------------
st.markdown("### All-MSA YOY Table (Difference or Raw) + Up to 3 Forecasts")

st.write("""
Compute January-to-January YOY for **all** MSAs, then switch between 
**raw YOY** or **(MSA minus National)** difference.  
We can also add up to 3 forecast yoy values for national and see each MSA's implied yoy or yoy difference.  
(We handle the case where there's not enough data to compute alpha/beta by returning None.)
""")

years_list_all = list(range(1990,2031))
col_z1, col_z2 = st.columns(2)
with col_z1:
    yoy_all_start = st.selectbox("All-MSA Start Year (>=1990)", years_list_all, index=years_list_all.index(2015))
with col_z2:
    yoy_all_end   = st.selectbox("All-MSA End Year (<=2030)", years_list_all, index=years_list_all.index(2024))

if yoy_all_end < yoy_all_start+5:
    st.warning("Please pick at least a 5-year window (end >= start+5).")

st.markdown("#### Enter up to 3 forecast yoy growth rates (National, in %):")
f1 = st.text_input("Forecast #1", value="1.0")
f2 = st.text_input("Forecast #2", value="2.5")
f3 = st.text_input("Forecast #3 (optional)", value="")

def parse_forecast(val_str):
    try:
        return float(val_str)
    except:
        return None

scenarios=[]
for label,val_str in [("Forecast #1",f1),("Forecast #2",f2),("Forecast #3",f3)]:
    fval = parse_forecast(val_str)
    if fval is not None:
        scenarios.append((label,fval))

diff_mode = st.checkbox("Show (MSA minus National) difference instead of raw yoy",value=False)

def get_alpha_beta_rsq_yoy(sid, yoy_years_list, yoy_map):
    # gather yoy points for each year
    xvals,yvals=[],[]
    for y in yoy_years_list:
        nat_y = yoy_map[NATIONAL_SERIES_ID].get(y,None)
        msa_y = yoy_map[sid].get(y,None)
        if nat_y is not None and msa_y is not None:
            xvals.append(nat_y)
            yvals.append(msa_y)
    if len(xvals)<2:
        return (None,None,None)
    # Build a small DataFrame with "nat" col, add const
    import pandas as pd
    X_df = pd.DataFrame({"nat": xvals})
    X_df = sm.add_constant(X_df, prepend=True)
    model = sm.OLS(yvals, X_df).fit()
    # Some safety checks in case "const" not found
    try:
        alpha_v = model.params["const"]
    except KeyError:
        return (None,None,None)
    # slope might be "nat"
    try:
        beta_v  = model.params["nat"]
    except KeyError:
        return (None,None,None)
    rsq_v = model.rsquared
    return (alpha_v,beta_v,rsq_v)

if st.button("Generate All-MSA Table"):
    if yoy_all_end < yoy_all_start+5:
        st.error("End year must be at least (start+5).")
        st.stop()

    df_filtered2 = df_full.copy()
    df_filtered2["year"] = df_filtered2["obs_date"].dt.year
    df_filtered2["month"]= df_filtered2["obs_date"].dt.month
    df_jan2 = df_filtered2[df_filtered2["month"]==1].copy()
    df_jan2 = df_jan2[(df_jan2["year"]>=yoy_all_start)&(df_jan2["year"]<=yoy_all_end)]
    if df_jan2.empty:
        st.warning("No january data in that range.")
        st.stop()

    pivot_jan2 = df_jan2.pivot(index="year",columns="series_id",values="value")
    pivot_jan2.dropna(how="all",inplace=True)
    sorted_years_2 = sorted(pivot_jan2.index)
    all_series_ids = pivot_jan2.columns.unique().tolist()

    # yoy_map[sid][year] = yoy
    yoy_map={}
    for sid in all_series_ids:
        yoy_map[sid]={}
        for yy in sorted_years_2:
            prev_y = yy-1
            if prev_y in pivot_jan2.index and sid in pivot_jan2.columns:
                val_t = pivot_jan2.loc[yy,sid]
                val_tm1 = pivot_jan2.loc[prev_y,sid]
                if pd.notnull(val_t) and pd.notnull(val_tm1) and val_tm1!=0:
                    yoy_map[sid][yy] = 100*(val_t - val_tm1)/val_tm1

    yoy_year_list = [y for y in range(yoy_all_start+1, yoy_all_end+1)]
    # build final table
    yoy_cols=[str(y) for y in yoy_year_list]
    fore_cols=[x[0] for x in scenarios]
    base_cols=["Metro"]+yoy_cols+fore_cols+["R-Squared"]

    final_rows=[]
    # National row first
    nat_row={"Metro":"National","R-Squared":None}
    for yy in yoy_year_list:
        val_nat = yoy_map[NATIONAL_SERIES_ID].get(yy,None)
        nat_row[str(yy)] = val_nat
    for (lbl,fval) in scenarios:
        # if difference => national=0, else => raw yoy = fval
        if diff_mode:
            nat_row[lbl]=0.0
        else:
            nat_row[lbl]=fval
    final_rows.append(nat_row)

    # each MSA
    all_msas = [k for k in MSA_NAME_MAP if k!=NATIONAL_SERIES_ID]
    # sort them by name
    sorted_msas = sorted(all_msas, key=lambda sid: MSA_NAME_MAP[sid])
    for sid in sorted_msas:
        name = MSA_NAME_MAP[sid]
        # compute alpha,beta,rsq
        alpha_v,beta_v,rsq_v = get_alpha_beta_rsq_yoy(sid,yoy_year_list,yoy_map)
        rowdict={"Metro":name,"R-Squared":rsq_v}
        for yy in yoy_year_list:
            yoy_n = yoy_map[NATIONAL_SERIES_ID].get(yy,None)
            yoy_m = yoy_map[sid].get(yy,None)
            if yoy_n is None or yoy_m is None:
                rowdict[str(yy)] = None
            else:
                if diff_mode:
                    rowdict[str(yy)] = yoy_m - yoy_n
                else:
                    rowdict[str(yy)] = yoy_m
        # forecast
        if alpha_v is not None and beta_v is not None:
            for (lbl,fval) in scenarios:
                yoy_msa_fore = alpha_v + beta_v*fval
                if diff_mode:
                    rowdict[lbl] = yoy_msa_fore - fval
                else:
                    rowdict[lbl] = yoy_msa_fore
        else:
            for (lbl,fval) in scenarios:
                rowdict[lbl]=None
        final_rows.append(rowdict)

    df_final=pd.DataFrame(final_rows)
    df_final=df_final[base_cols]

    # color code yoy/diff
    def colorfunc(val):
        if val is None:
            return ""
        if isinstance(val,(int,float)):
            if val>0:
                return "background-color: rgba(76, 175, 80, 0.4);"  # green
            elif val<0:
                return "background-color: rgba(255, 0, 0, 0.3);"   # red
        return ""

    style_cols=yoy_cols+fore_cols
    styler=df_final.style.format(
        subset=style_cols, formatter="{:.2f}"
    ).format(
        subset=["R-Squared"], formatter="{:.3f}"
    ).applymap(
        colorfunc, subset=style_cols
    )

    st.dataframe(styler,use_container_width=True)

    st.markdown(f"""
    **Notes**  
    - "National" row is at the top for reference.  
    - If "Difference" mode is on, we do (MSA yoy - National yoy), else raw yoy.  
    - Forecast columns use alpha/beta from the entire {yoy_all_start}–{yoy_all_end} window for each MSA.  
    - R-Squared is from yoy_msa ~ alpha+beta*yoy_nat across that window.  
    - If there's not enough data to fit alpha/beta, that MSA's forecast columns and R-Squared will be None.
    """)

# ---------------------------------------------------------------------
# 8) YEAR-OVER-YEAR BAR CHART (Single MSA)
# ---------------------------------------------------------------------
st.markdown("### Year-over-Year Bar Chart (Single MSA)")

st.write("""
Here we do a January-to-January YOY for a single MSA vs. the Nation, 
then let you provide a single forecast for next year.  
We've placed this **below** the big table so you can see the full MSA table first, 
and then optionally pick a single MSA for a bar chart.
""")

years_list_single = list(range(1990,2031))
col_y1,col_y2=st.columns(2)
with col_y1:
    yoy_start_sing=st.selectbox("Start Year (>=1990)", years_list_single, index=years_list_single.index(2015))
with col_y2:
    yoy_end_sing  =st.selectbox("End Year (<=2030)", years_list_single, index=years_list_single.index(2024))

single_msa_choice=st.selectbox("Select 1 MSA:",sorted(INVERTED_MAP.keys()))
nat_forecast_val=st.number_input("Next-Year Nat. Growth Forecast (%)",value=1.0,step=0.1,format="%.1f")

if st.button("Generate Single-MSA YOY Chart"):
    if yoy_end_sing<yoy_start_sing:
        st.error("End year < start year not valid.")
        st.stop()

    # Filter january
    df_filtered_single = df_full.copy()
    df_filtered_single["year"]=df_filtered_single["obs_date"].dt.year
    df_filtered_single["month"]=df_filtered_single["obs_date"].dt.month
    df_jan_sing = df_filtered_single[df_filtered_single["month"]==1]
    df_jan_sing = df_jan_sing[(df_jan_sing["year"]>=yoy_start_sing)&(df_jan_sing["year"]<=yoy_end_sing)]
    if df_jan_sing.empty:
        st.warning("No january data in that range.")
        st.stop()

    sid_msa = INVERTED_MAP[single_msa_choice]
    sel_ids = [sid_msa,NATIONAL_SERIES_ID]
    df_jan_sing=df_jan_sing[df_jan_sing["series_id"].isin(sel_ids)]
    pivot_jan_sing=df_jan_sing.pivot(index="year",columns="series_id",values="value")
    pivot_jan_sing.dropna(inplace=True)
    yoy_list_sing=[]
    sorted_yrs_sing=sorted(pivot_jan_sing.index)
    for y in sorted_yrs_sing:
        prev_y=y-1
        if prev_y in pivot_jan_sing.index:
            if (sid_msa in pivot_jan_sing.columns) and (NATIONAL_SERIES_ID in pivot_jan_sing.columns):
                nat_t = pivot_jan_sing.loc[y,NATIONAL_SERIES_ID]
                nat_tm1=pivot_jan_sing.loc[prev_y,NATIONAL_SERIES_ID]
                yoy_nat=None
                yoy_msa=None
                if pd.notnull(nat_t) and pd.notnull(nat_tm1) and nat_tm1!=0:
                    yoy_nat=100*(nat_t-nat_tm1)/nat_tm1
                msa_t = pivot_jan_sing.loc[y,sid_msa]
                msa_tm1=pivot_jan_sing.loc[prev_y,sid_msa]
                if pd.notnull(msa_t) and pd.notnull(msa_tm1) and msa_tm1!=0:
                    yoy_msa=100*(msa_t-msa_tm1)/msa_tm1
                if yoy_nat is not None and yoy_msa is not None:
                    yoy_list_sing.append({
                        "Year":y,
                        "Nat_Growth":yoy_nat,
                        "MSA_Growth":yoy_msa
                    })
    df_yoy_sing=pd.DataFrame(yoy_list_sing)
    if df_yoy_sing.empty:
        st.warning("Not enough consecutive january data to build yoy for single MSA.")
        st.stop()

    # OLS yoy_msa=alpha+beta*yoy_nat
    Xdf=sm.add_constant(df_yoy_sing["Nat_Growth"],prepend=True)
    yvals=df_yoy_sing["MSA_Growth"]
    model=sm.OLS(yvals,Xdf).fit()
    alpha_v=None
    beta_v=None
    rsq_v=None
    try:
        alpha_v=model.params["const"]
        # The slope may be "Nat_Growth" or just Xdf columns[-1]
        colname = [c for c in model.params.index if c!="const"]
        if len(colname)==1:
            beta_v=model.params[colname[0]]
        rsq_v=model.rsquared
    except:
        alpha_v=None
        beta_v=None
        rsq_v=None

    yoy_msa_fore=None
    if alpha_v is not None and beta_v is not None:
        yoy_msa_fore=alpha_v + beta_v*nat_forecast_val

    # build bar chart
    df_plot_sing=df_yoy_sing.copy()
    df_plot_sing.loc[df_plot_sing.shape[0]]={
        "Year":"Forecast",
        "Nat_Growth":nat_forecast_val,
        "MSA_Growth":yoy_msa_fore
    }

    bar_rows=[]
    for i,row_ in df_plot_sing.iterrows():
        bar_rows.append({
            "Year":row_["Year"],
            "Series":"National",
            "Growth":row_["Nat_Growth"]
        })
        bar_rows.append({
            "Year":row_["Year"],
            "Series":single_msa_choice,
            "Growth":row_["MSA_Growth"]
        })

    df_bar_sing=pd.DataFrame(bar_rows)
    fig_bar_sing=px.bar(
        df_bar_sing,
        x="Year",y="Growth",
        color="Series",
        barmode="group",
        title=f"Year-over-Year Growth (Jan) — National vs {single_msa_choice}"
    )
    fig_bar_sing.update_layout(xaxis_type='category')
    st.plotly_chart(fig_bar_sing,use_container_width=True)

    st.markdown("#### OLS (YoY) Results for Single MSA")
    df_ols_sing = pd.DataFrame([{
        "MSA":single_msa_choice,
        "Alpha":alpha_v,
        "Beta":beta_v,
        "R-Squared":rsq_v
    }])
    st.dataframe(df_ols_sing)
