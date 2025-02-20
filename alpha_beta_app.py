import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import datetime

# ---------------------------------------------------------------------
# 0) Page Config Must Be FIRST
# ---------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="US Metro Analysis")

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
    /* Pin first column in dataframes */
    .css-1ikzb0t .dataframe tr th:first-of-type, 
    .css-1ikzb0t .dataframe td:first-of-type {
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
# 3) About the Data & separate How to Use
# ---------------------------------------------------------------------
with st.expander("About the Data", expanded=False):
    st.markdown(
        """
        **Data Source**  
        - All data is publicly available from the Bureau of Labor Statistics (BLS).  
        - This tool features the top 50 U.S. MSAs (Metro Statistical Area) by population, plus a “National” benchmark.

        **Alpha & Beta**  
        - **Alpha**: Indicates the MSA’s baseline performance relative to national growth for the *chosen* historical range.  
          - A **positive** alpha suggests the MSA tends to have higher growth even if national growth is zero within that range.  
          - A **negative** alpha implies underperformance relative to a zero-national-growth scenario.  
        - **Beta**: Reflects how strongly the MSA’s growth moves in *proportion* to national changes over the *selected* data window.  
          - Beta > 1 → The MSA “amplifies” or responds more strongly than the national average.  
          - Beta < 1 → The MSA is more stable, not swinging as much as national.  

        **R-Squared**  
        - Gauges how well alpha & beta describe the MSA’s relationship with national growth (on a 0–1 scale).  
        - ~0.70+ = High confidence, 0.50–0.70 = Medium, <0.50 = Low.  
        - **Higher R-Squared implies a more reliable model for forecasting MSA growth** based on national growth.
        """
    )

with st.expander("How To Use", expanded=False):
    st.markdown(
        """
        1. **Metric**  
           - Select which metric (e.g., Total NonFarm Employment) at the top; everything references that series.  
        2. **XY Chart (Alpha vs Beta)**  
           - Choose MSAs + date range, then generate a scatter of Beta (x-axis) vs Alpha (y-axis).  
        3. **Time Series (Rolling 12-Month Alpha/Beta)**  
           - Up to 5 MSAs, pick alpha or beta, define date range.  
           - Shows monthly rolling alpha/beta lines.  
        4. **Historical Growth and Forecasts (All MSAs)**  
           - Year-over-Year (January–January) data for every MSA vs national.  
           - Input up to 3 forecast % for national. We compute each MSA’s implied growth from alpha & beta (in the chosen window).  
        5. **Single MSA Comparative Year Over Year Growth**  
           - Pick 1 MSA + start/end years.  
           - See a bar chart of national vs MSA historical growth.  
           - Input up to 3 forecast % to see the MSA’s projected growth bars.
        """
    )

# ---------------------------------------------------------------------
# 4) Load CSV
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
INVERTED_MAP = {v:k for k,v in MSA_NAME_MAP.items()}

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
# 6) Common Utility Functions
# ---------------------------------------------------------------------
import math

def fetch_raw_data_multiple(msa_ids, start_ym, end_ym):
    if NATIONAL_SERIES_ID not in msa_ids:
        msa_ids.append(NATIONAL_SERIES_ID)
    start_year, start_month = map(int, start_ym.split("-"))
    end_year, end_month = map(int, end_ym.split("-"))
    start_dt = datetime.datetime(start_year, start_month,1)
    end_dt   = datetime.datetime(end_year, end_month,1)
    df_filtered = df_full[
        (df_full["series_id"].isin(msa_ids)) &
        (df_full["obs_date"] >= start_dt) &
        (df_full["obs_date"] <= end_dt)
    ].copy()
    return df_filtered

import statsmodels.api as sm

def compute_multi_alpha_beta(df_raw):
    df_raw["value"] = pd.to_numeric(df_raw["value"], errors="coerce")
    pivoted = df_raw.pivot(index="obs_date", columns="series_id", values="value")
    growth = pivoted.pct_change(1)*100
    growth.dropna(inplace=True)
    if NATIONAL_SERIES_ID not in growth.columns:
        return pd.DataFrame()
    results=[]
    for col in growth.columns:
        if col==NATIONAL_SERIES_ID:
            continue
        sub = growth[[NATIONAL_SERIES_ID,col]].dropna()
        if len(sub)<2:
            continue
        X = sm.add_constant(sub[NATIONAL_SERIES_ID])
        y = sub[col]
        model=sm.OLS(y,X).fit()
        alpha_v = model.params.get("const",None)
        slope_keys=[k for k in model.params.keys() if k!="const"]
        beta_v=None
        if len(slope_keys)==1:
            beta_v=model.params[slope_keys[0]]
        rsq_v=model.rsquared
        results.append({
            "Metro": MSA_NAME_MAP.get(col,col),
            "Alpha": alpha_v,
            "Beta": beta_v,
            "R-Squared": rsq_v
        })
    return pd.DataFrame(results)

def compute_alpha_beta_subset(df_subset, nat_col, msa_col):
    if len(df_subset.dropna())<2:
        return None,None
    X=sm.add_constant(df_subset[nat_col])
    y=df_subset[msa_col]
    model=sm.OLS(y,X).fit()
    alpha_v=model.params.get("const",None)
    slope=[k for k in model.params.keys() if k!="const"]
    beta_v=None
    if len(slope)>0:
        beta_v=model.params[slope[0]]
    return alpha_v,beta_v

def compute_rolling_alpha_beta_time_series(df_raw_ts, start_ym_ts, end_ym_ts):
    pivoted = df_raw_ts.pivot(index="obs_date", columns="series_id",values="value")
    growth = pivoted.pct_change(1)*100
    growth.dropna(inplace=True)
    if NATIONAL_SERIES_ID not in growth.columns:
        return pd.DataFrame()
    start_year, start_month= map(int, start_ym_ts.split("-"))
    end_year, end_month   = map(int, end_ym_ts.split("-"))
    start_dt = datetime.datetime(start_year,start_month,1)
    end_dt   = datetime.datetime(end_year,end_month,1)
    unique_months=sorted(growth.index.unique())
    ROLLING_WINDOW_MONTHS=12
    res=[]
    for current_m in unique_months:
        if current_m<start_dt or current_m>end_dt:
            continue
        rolling_start=current_m - pd.DateOffset(months=ROLLING_WINDOW_MONTHS-1)
        window = growth.loc[(growth.index>=rolling_start)&(growth.index<=current_m)]
        if len(window)<2:
            continue
        for col in window.columns:
            if col==NATIONAL_SERIES_ID:
                continue
            sub=window[[NATIONAL_SERIES_ID,col]].dropna()
            a_v,b_v=compute_alpha_beta_subset(sub,NATIONAL_SERIES_ID,col)
            if a_v is not None and b_v is not None:
                res.append({
                    "obs_date":current_m,
                    "series_id":col,
                    "alpha":a_v,
                    "beta":b_v
                })
    return pd.DataFrame(res)

# ---------------------------------------------------------------------
# 7) XY CHART
# ---------------------------------------------------------------------
st.markdown("### XY Chart (Alpha vs Beta)")

def select_all():
    st.session_state["xy_msas"] = sorted(INVERTED_MAP.keys())
def clear_all():
    st.session_state["xy_msas"]=[]

if "xy_msas" not in st.session_state:
    st.session_state["xy_msas"]=[]

xy_all_msas=sorted(INVERTED_MAP.keys())
st.multiselect("Pick MSA(s):", options=xy_all_msas, key="xy_msas")

col_xy1,col_xy2 = st.columns(2)
col_xy1.button("Select All", on_click=select_all)
col_xy2.button("Clear", on_click=clear_all)

months_list=["January","February","March","April","May","June","July","August","September","October","November","December"]
years_xy= list(range(1990,2025))  # up to 2024
default_xy_start=2019
default_xy_end=2024

st.write("#### Date Range for XY Chart")
cxy1,cxy2=st.columns(2)
with cxy1:
    xy_start_month=st.selectbox("Start Month (XY)", months_list, index=0)
    xy_start_year =st.selectbox("Start Year (XY)", years_xy, index=years_xy.index(default_xy_start))
with cxy2:
    xy_end_month=st.selectbox("End Month (XY)", months_list, index=11)
    xy_end_year =st.selectbox("End Year (XY)", years_xy, index=years_xy.index(default_xy_end))

xy_smonth= months_list.index(xy_start_month)+1
xy_emonth= months_list.index(xy_end_month)+1
xy_start_ym = f"{xy_start_year:04d}-{xy_smonth:02d}"
xy_end_ym   = f"{xy_end_year:04d}-{xy_emonth:02d}"

if st.button("Generate XY Chart"):
    if not st.session_state["xy_msas"]:
        st.warning("No MSAs selected!")
    else:
        chosen_ids=[INVERTED_MAP[m] for m in st.session_state["xy_msas"]]
        df_xy= fetch_raw_data_multiple(chosen_ids, xy_start_ym, xy_end_ym)
        if df_xy.empty:
            st.warning("No data found for that range.")
        else:
            df_ab= compute_multi_alpha_beta(df_xy)
            if df_ab.empty:
                st.error("Could not compute alpha/beta.")
            else:
                title_xy=f"Alpha vs Beta ({xy_start_ym} to {xy_end_ym}) - {metric_choice}"
                fig_xy=px.scatter(
                    df_ab,x="Beta", y="Alpha", text="Metro",
                    title=title_xy, render_mode="webgl"
                )
                fig_xy.update_traces(textposition='top center')
                fig_xy.update_layout(
                    dragmode='pan',
                    title_x=0.5,
                    title_xanchor='center',
                    xaxis=dict(scroller="True") 
                )
                # Re-enable scroll zoom:
                fig_xy.update_layout(
                    xaxis=dict(fixedrange=False),
                    yaxis=dict(fixedrange=False)
                )
                # Add lines
                fig_xy.add_hline(y=0,line_width=2,line_color="black",line_dash="dot")
                fig_xy.add_vline(x=1,line_width=2,line_color="black",line_dash="dot")
                st.plotly_chart(fig_xy, use_container_width=True, config={"scrollZoom":True})
                if st.checkbox("View alpha/beta table for XY chart"):
                    st.dataframe(df_ab)

# ---------------------------------------------------------------------
# 8) TIME SERIES (Rolling 12-Month Alpha/Beta)
# ---------------------------------------------------------------------
st.markdown("### Time Series (Rolling 12-Month Alpha/Beta)")
all_msa_names_no_nat=[m for m in xy_all_msas if m!="National"]
roll_selected_msas= st.multiselect(
    "Pick up to 5 MSAs:",
    options=all_msa_names_no_nat,
    max_selections=5
)
roll_ab_choice= st.selectbox("Which metric to plot?", ["alpha","beta"])
roll_years= list(range(1990,2025)) 
roll_default_start=2019
roll_default_end=2024

st.write("#### Date Range for Time Series (Rolling Window)")
ct1,ct2=st.columns(2)
with ct1:
    ts_start_month= st.selectbox("Start Month", months_list, index=0)
    ts_start_year = st.selectbox("Start Year", roll_years, index=roll_years.index(roll_default_start))
with ct2:
    ts_end_month= st.selectbox("End Month", months_list, index=11)
    ts_end_year = st.selectbox("End Year", roll_years, index=roll_years.index(roll_default_end))

ts_smonth_num = months_list.index(ts_start_month)+1
ts_emonth_num = months_list.index(ts_end_month)+1
ts_start_ym= f"{ts_start_year:04d}-{ts_smonth_num:02d}"
ts_end_ym  = f"{ts_end_year:04d}-{ts_emonth_num:02d}"

if st.button("Compute Rolling Time Series"):
    if not roll_selected_msas:
        st.warning("Pick at least 1 MSA.")
    else:
        chosen_ids=[INVERTED_MAP[n] for n in roll_selected_msas]
        df_roll_raw= fetch_raw_data_multiple(chosen_ids,"1990-01",ts_end_ym)
        if df_roll_raw.empty:
            st.warning("No data found.")
        else:
            df_roll_res= compute_rolling_alpha_beta_time_series(df_roll_raw, ts_start_ym, ts_end_ym)
            if df_roll_res.empty:
                st.warning("No rolling alpha/beta computed.")
            else:
                df_roll_res["Metro"]= df_roll_res["series_id"].apply(lambda sid: MSA_NAME_MAP.get(sid,sid))
                df_roll_res.drop(columns=["series_id"],inplace=True)
                df_roll_res.rename(columns={"obs_date":"Date"},inplace=True)
                df_roll_res= df_roll_res[["Date","Metro","alpha","beta"]]
                df_plot= df_roll_res.copy()
                df_plot["AB_Chosen"]= df_plot[roll_ab_choice]
                title_ts= f"Time Series of {roll_ab_choice.title()} (Rolling 12-Month) {ts_start_ym}–{ts_end_ym}"
                fig_ts = px.line(
                    df_plot,x="Date",y="AB_Chosen",
                    color="Metro",title=title_ts
                )
                fig_ts.update_layout(
                    dragmode='pan',
                    xaxis_title="Date",
                    yaxis_title=roll_ab_choice.title(),
                    title_x=0.5,
                    title_xanchor="center"
                )
                # Re-enable scroll zoom
                fig_ts.update_layout(
                    xaxis=dict(fixedrange=False),
                    yaxis=dict(fixedrange=False)
                )
                st.plotly_chart(fig_ts,use_container_width=True,config={"scrollZoom":True})
                if st.checkbox("Show rolling data table"):
                    st.dataframe(df_roll_res)

# ---------------------------------------------------------------------
# 9) HISTORICAL GROWTH AND FORECASTS (All MSAs)
# ---------------------------------------------------------------------
st.markdown("### Historical Growth and Forecasts")
st.write("""
View historical comparisons between national and metro growth. 
Enter national growth forecast scenarios to view projected impact at a metro level.
- Forecasts are based on alpha and beta **within the selected period** below.
""")

hist_years = list(range(1990,2025))
col_h1,col_h2 = st.columns(2)
with col_h1:
    yoy_table_start= st.selectbox("Start Year", hist_years, index=hist_years.index(2015))
with col_h2:
    yoy_table_end  = st.selectbox("End Year", hist_years, index=hist_years.index(2024))

if yoy_table_end < yoy_table_start+5:
    st.warning("Please pick at least a 5-year window (end >= start+5).")

st.markdown("#### Optional – Forecast year over year National growth rate (%)")
table_f1 = st.text_input("Forecast #1", value="1.0")
table_f2 = st.text_input("Forecast #2", value="2.5")
table_f3 = st.text_input("Forecast #3", value="")

def parse_forecast_val(val_str):
    try:
        return float(val_str)
    except:
        return None

scenario_vals=[]
for lbl,vstr in [("Forecast #1",table_f1),("Forecast #2",table_f2),("Forecast #3",table_f3)]:
    fval=parse_forecast_val(vstr)
    if fval is not None:
        scenario_vals.append((lbl,fval))

diff_mode = st.checkbox("Show National vs Metro Variance (MSA growth minus National)", value=False)

def get_alpha_beta_rsq_yoy(sid,ylist,yoy_map):
    xvals,yvals=[],[]
    for y in ylist:
        nat_val = yoy_map[NATIONAL_SERIES_ID].get(y,None)
        msa_val = yoy_map[sid].get(y,None)
        if nat_val is not None and msa_val is not None:
            xvals.append(nat_val)
            yvals.append(msa_val)
    if len(xvals)<2:
        return (None,None,None)
    Xdf = pd.DataFrame({"nat":xvals})
    Xdf = sm.add_constant(Xdf, prepend=True)
    model=sm.OLS(yvals,Xdf).fit()
    alpha_v=None
    beta_v=None
    rsq_v=None
    try:
        alpha_v= model.params["const"]
        beta_v = model.params["nat"]
        rsq_v  = model.rsquared
    except:
        alpha_v,beta_v,rsq_v = None,None,None
    return alpha_v,beta_v,rsq_v

if st.button("Generate Table"):
    if yoy_table_end<yoy_table_start+5:
        st.error("End year must be at least (start+5).")
        st.stop()

    df_tmp = df_full.copy()
    df_tmp["year"]= df_tmp["obs_date"].dt.year
    df_tmp["month"]=df_tmp["obs_date"].dt.month
    df_jan = df_tmp[df_tmp["month"]==1].copy()
    df_jan = df_jan[(df_jan["year"]>=yoy_table_start)&(df_jan["year"]<=yoy_table_end)]
    if df_jan.empty:
        st.warning("No january data in that range.")
        st.stop()

    pivot_jan = df_jan.pivot(index="year",columns="series_id",values="value")
    pivot_jan.dropna(how="all",inplace=True)
    sorted_yrs=sorted(pivot_jan.index)
    all_sids = pivot_jan.columns.unique().tolist()

    yoy_map={}
    for sid in all_sids:
        yoy_map[sid]={}
        for yy in sorted_yrs:
            prev_yy= yy-1
            if prev_yy in pivot_jan.index:
                val_t   = pivot_jan.loc[yy,sid]
                val_tm1 = pivot_jan.loc[prev_yy,sid]
                if pd.notnull(val_t) and pd.notnull(val_tm1) and val_tm1!=0:
                    yoy_val=100*(val_t-val_tm1)/val_tm1
                    yoy_map[sid][yy]= yoy_val

    yoy_years_list= [y for y in range(yoy_table_start+1,yoy_table_end+1)]
    yoy_cols = [str(y) for y in yoy_years_list]
    fore_cols= [x[0] for x in scenario_vals]
    base_cols=["Metro"]+yoy_cols+fore_cols+["R-Squared"]

    final_rows=[]
    # national row
    row_nat={"Metro":"National","R-Squared":None}
    for yy in yoy_years_list:
        v_n = yoy_map[NATIONAL_SERIES_ID].get(yy,None)
        row_nat[str(yy)] = v_n
    for (lbl,fval) in scenario_vals:
        if diff_mode:
            row_nat[lbl]=0.0
        else:
            row_nat[lbl]=fval
    final_rows.append(row_nat)

    # each MSA
    all_msas=[k for k in MSA_NAME_MAP if k!=NATIONAL_SERIES_ID]
    sorted_msas= sorted(all_msas, key=lambda sid: MSA_NAME_MAP[sid])
    for sid in sorted_msas:
        name= MSA_NAME_MAP[sid]
        alpha_v,beta_v,rsq_v= get_alpha_beta_rsq_yoy(sid,yoy_years_list,yoy_map)
        rowdict={"Metro":name,"R-Squared":rsq_v}
        for yy in yoy_years_list:
            yoy_nat= yoy_map[NATIONAL_SERIES_ID].get(yy,None)
            yoy_msa= yoy_map[sid].get(yy,None)
            if yoy_nat is None or yoy_msa is None:
                rowdict[str(yy)] = None
            else:
                if diff_mode:
                    rowdict[str(yy)] = yoy_msa - yoy_nat
                else:
                    rowdict[str(yy)] = yoy_msa
        if alpha_v is not None and beta_v is not None:
            for (lbl,fval) in scenario_vals:
                yoy_msa_fore= alpha_v + beta_v*fval
                if diff_mode:
                    rowdict[lbl] = yoy_msa_fore - fval
                else:
                    rowdict[lbl] = yoy_msa_fore
        else:
            for (lbl,fval) in scenario_vals:
                rowdict[lbl] = None
        final_rows.append(rowdict)

    df_table= pd.DataFrame(final_rows)
    df_table= df_table[base_cols]

    # color yoy or difference
    def color_func(val):
        if val is None:
            return ""
        if isinstance(val,(int,float)):
            if val>0:
                return "background-color: rgba(76, 175, 80, 0.4);"  # green
            elif val<0:
                return "background-color: rgba(255, 0, 0, 0.3);"   # red
        return ""

    style_cols=yoy_cols+fore_cols
    styler=df_table.style.format(
        subset=style_cols, formatter="{:.2f}"
    ).format(
        subset=["R-Squared"], formatter="{:.3f}"
    ).applymap(color_func, subset=style_cols)

    st.dataframe(styler,use_container_width=True)

# ---------------------------------------------------------------------
# 10) SINGLE MSA COMPARATIVE YEAR OVER YEAR GROWTH
# ---------------------------------------------------------------------
st.markdown("### Single MSA Comparative Year Over Year Growth")
st.write("""
Directly compare national and metro growth rates. Visualize metro growth projections.
""")

single_years_list=list(range(1990,2025))  # up to 2024
c_s1,c_s2= st.columns(2)
with c_s1:
    single_start_yr= st.selectbox("Start Year", single_years_list, index=single_years_list.index(2015))
with c_s2:
    single_end_yr  = st.selectbox("End Year", single_years_list, index=single_years_list.index(2024))

single_msa_pick= st.selectbox("Select an MSA:", sorted(INVERTED_MAP.keys()))

# 3 forecast yoy
st.markdown("#### Optional – Forecast year over year National growth rate (%)")
sing_f1= st.text_input("Scenario #1", value="1.0")
sing_f2= st.text_input("Scenario #2", value="2.5")
sing_f3= st.text_input("Scenario #3", value="")

def parse_sing_forecast(x):
    try:
        return float(x)
    except:
        return None

sing_scenarios=[]
for lbl,val in [("Scenario #1",sing_f1),("Scenario #2",sing_f2),("Scenario #3",sing_f3)]:
    fv=parse_sing_forecast(val)
    if fv is not None:
        sing_scenarios.append((lbl,fv))

if st.button("Generate Single-MSA YOY Chart"):
    if single_end_yr<single_start_yr:
        st.error("End year < start year not valid.")
        st.stop()

    # Filter january
    df_sing= df_full.copy()
    df_sing["year"]= df_sing["obs_date"].dt.year
    df_sing["month"]= df_sing["obs_date"].dt.month
    df_sing_jan= df_sing[df_sing["month"]==1]
    df_sing_jan= df_sing_jan[(df_sing_jan["year"]>=single_start_yr)&(df_sing_jan["year"]<=single_end_yr)]
    if df_sing_jan.empty:
        st.warning("No january data in that range.")
        st.stop()

    sid_msa= INVERTED_MAP[single_msa_pick]
    sel_ids=[sid_msa,NATIONAL_SERIES_ID]
    df_sing_jan= df_sing_jan[df_sing_jan["series_id"].isin(sel_ids)]
    pivot_sing= df_sing_jan.pivot(index="year", columns="series_id", values="value")
    pivot_sing.dropna(inplace=True)
    yoy_sing=[]
    sorted_sing_yrs=sorted(pivot_sing.index)
    for y in sorted_sing_yrs:
        py=y-1
        if py in pivot_sing.index:
            nat_val_t = pivot_sing.loc[y,NATIONAL_SERIES_ID]
            nat_val_m1= pivot_sing.loc[py,NATIONAL_SERIES_ID]
            msa_val_t = pivot_sing.loc[y,sid_msa]
            msa_val_m1= pivot_sing.loc[py,sid_msa]
            if pd.notnull(nat_val_t) and pd.notnull(nat_val_m1) and nat_val_m1!=0 \
               and pd.notnull(msa_val_t) and pd.notnull(msa_val_m1) and msa_val_m1!=0:
                yoy_nat=100*(nat_val_t-nat_val_m1)/nat_val_m1
                yoy_msa=100*(msa_val_t-msa_val_m1)/msa_val_m1
                yoy_sing.append({
                    "Year":y,
                    "Nat_Growth":yoy_nat,
                    "MSA_Growth":yoy_msa
                })
    df_sing_yoy= pd.DataFrame(yoy_sing)
    if df_sing_yoy.empty:
        st.warning("Not enough consecutive january data for yoy.")
        st.stop()

    Xdf= sm.add_constant(df_sing_yoy["Nat_Growth"], prepend=True)
    yvals= df_sing_yoy["MSA_Growth"]
    model= sm.OLS(yvals, Xdf).fit()
    alpha_val=None
    beta_val=None
    rsq_val=None
    try:
        alpha_val= model.params["const"]
        slope_keys = [c for c in model.params.index if c!="const"]
        if len(slope_keys)==1:
            beta_val= model.params[slope_keys[0]]
        rsq_val= model.rsquared
    except:
        alpha_val,beta_val,rsq_val= None,None,None

    # Build forecast row
    # yoy_msa_fore= alpha + beta*(nat_fore)
    forecast_rows=[]
    df_plot_sing= df_sing_yoy.copy()
    for (lbl,fval) in sing_scenarios:
        yoy_msa_fore=None
        if alpha_val is not None and beta_val is not None:
            yoy_msa_fore= alpha_val + beta_val*fval
        forecast_rows.append({
            "Year": lbl,
            "Nat_Growth": fval,
            "MSA_Growth": yoy_msa_fore
        })
    if forecast_rows:
        df_plot_sing= pd.concat([df_sing_yoy, pd.DataFrame(forecast_rows)], ignore_index=True)

    # Build bar chart
    bar_list=[]
    for i,row_ in df_plot_sing.iterrows():
        bar_list.append({
            "Year": row_["Year"],
            "Series":"National",
            "Growth": row_["Nat_Growth"]
        })
        bar_list.append({
            "Year": row_["Year"],
            "Series": single_msa_pick,
            "Growth": row_["MSA_Growth"]
        })
    df_bar_sing= pd.DataFrame(bar_list)
    title_sing= f"Year-over-Year Growth (Jan) — National vs {single_msa_pick}"
    fig_sing= px.bar(
        df_bar_sing,
        x="Year",
        y="Growth",
        color="Series",
        barmode="group",
        title=title_sing
    )
    fig_sing.update_layout(xaxis_type='category')
    # Re-enable scroll/pinch zoom
    fig_sing.update_layout(
        xaxis=dict(fixedrange=False),
        yaxis=dict(fixedrange=False)
    )
    st.plotly_chart(fig_sing,use_container_width=True,config={"scrollZoom":True})

    # Add "View data table" checkbox
    if st.checkbox("View data table"):
        st.dataframe(df_plot_sing)

    # Summary Stats
    st.markdown("#### Summary Statistics")
    df_ols_sing= pd.DataFrame([{
        "MSA": single_msa_pick,
        "Alpha": alpha_val,
        "Beta": beta_val,
        "R-Squared": rsq_val
    }])
    st.dataframe(df_ols_sing)

    # Interpretation
    st.markdown("#### Interpretation")
    # If "Total NonFarm Employment" is selected, add link to FRED
    if metric_choice.startswith("Total NonFarm Employment"):
        st.write("For historical YoY context, see FRED chart: [Historical NonFarm Employment YoY](https://fred.stlouisfed.org/graph/?g=1DRDw)")
    if alpha_val is not None and beta_val is not None:
        lines=[]
        for (lbl,fval) in sing_scenarios:
            yoy_msa=None
            yoy_msa= alpha_val + beta_val*fval
            lines.append(f"For **{lbl}** (National = {fval:.1f}%), the projected MSA growth = **{yoy_msa:.1f}%**.")
        if lines:
            st.write("Based on the above alpha/beta, here are scenario implications:")
            for l in lines:
                st.write("-", l)
        else:
            st.write("No valid forecast entries found.")
    else:
        st.write("Insufficient data to interpret alpha/beta for this MSA.")
