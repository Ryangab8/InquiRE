import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from datetime import datetime

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
        - This CSV (`data/nonfarm_data.csv`) has columns:
          `id, series_id, alpha, beta, r_squared, start_date, end_date, created_at`.
        - Each row is a final alpha/beta result for a given MSA and date range.

        **Alpha**  
        - Measures growth relative to national trends (positive = outperformance, negative = underperformance).

        **Beta**  
        - Measures volatility relative to national trends (1 = in sync, >1 = more volatile, <1 = less volatile).
        """
    )

with st.expander("How To Use", expanded=False):
    st.markdown(
        """
        **Step 1:** The metric dropdown is mostly for show now; we focus on alpha/beta in the CSV.  

        **XY Chart (Alpha vs. Beta)**  
        1. Pick MSA(s) from the dropdown, or click “Select All.”
        2. Choose a start/end date range (filters on `start_date`).
        3. Click “Generate XY Chart.” Each MSA is shown by (β, α).

        **Time Series (Alpha/Beta Over Start Date)**  
        1. Select up to 5 MSAs (excluding “National”).
        2. Pick alpha or beta, plus a date range.
        3. Click “Compute Time Series.” We line-plot your chosen metric vs. `start_date`.

        **MSA Map**  
        1. We assume a second file (`data/msa_map_coords.csv`) with lat/lon for each `series_id`.
        2. Select a date range, pick alpha or beta for coloring, and generate a scatter map.
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
# 6) Load the CSV with alpha/beta columns
# ---------------------------------------------------------------------
CSV_FILE = "data/nonfarm_data.csv"

@st.cache_data
def load_main_csv():
    df = pd.read_csv(CSV_FILE)
    needed = {"id","series_id","alpha","beta","r_squared","start_date","end_date","created_at"}
    missing = needed - set(df.columns)
    if missing:
        st.error(f"CSV is missing columns: {missing}")
        st.stop()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"]   = pd.to_datetime(df["end_date"],   errors="coerce")
    df.sort_values(["series_id","start_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df_raw = load_main_csv()

# For convenience, unique MSA list
all_series_ids = sorted(df_raw["series_id"].unique())
NATIONAL_SERIES_ID = "CES0000000001"

MSA_NAME_MAP = {
    "CES0000000001": "National",
    # add more if you want
}

def get_msa_name(sid):
    return MSA_NAME_MAP.get(sid, sid)

# ---------------------------------------------------------------------
# 7) XY Chart (Alpha vs. Beta)
# ---------------------------------------------------------------------
st.markdown("### XY Chart (Alpha vs. Beta)")

if "msa_selection" not in st.session_state:
    st.session_state["msa_selection"] = []

st.multiselect(
    "Pick MSA(s):",
    options=all_series_ids,
    key="msa_selection"
)

col1, col2 = st.columns(2)
col1.button("Select All", on_click=lambda: st.session_state.update({"msa_selection": all_series_ids}))
col2.button("Clear", on_click=lambda: st.session_state.update({"msa_selection": []}))

months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
years_xy = list(range(1990, 2025))

def year_index(val):
    return years_xy.index(val) if val in years_xy else 0

default_start_year = 2019
default_end_year   = 2024

st.write("#### Date Range for XY Chart (filters on `start_date`)")
col_s1, col_s2 = st.columns(2)
with col_s1:
    xy_start_month = st.selectbox("Start Month (XY)", months, index=0)
    xy_start_year  = st.selectbox("Start Year (XY)", years_xy, index=year_index(default_start_year))
with col_s2:
    xy_end_month   = st.selectbox("End Month (XY)", months, index=11)
    xy_end_year    = st.selectbox("End Year (XY)", years_xy, index=year_index(default_end_year))

xy_smonth_num = months.index(xy_start_month) + 1
xy_emonth_num = months.index(xy_end_month) + 1
xy_start_dt = datetime(xy_start_year, xy_smonth_num, 1)
xy_end_dt   = datetime(xy_end_year,   xy_emonth_num, 1)

if st.button("Generate XY Chart"):
    if not st.session_state["msa_selection"]:
        st.warning("No MSAs selected!")
    else:
        df_xy = df_raw[df_raw["series_id"].isin(st.session_state["msa_selection"])].copy()
        df_xy = df_xy[(df_xy["start_date"] >= xy_start_dt) & (df_xy["start_date"] <= xy_end_dt)]

        if df_xy.empty:
            st.warning("No rows found for that date range/MSAs.")
        else:
            df_xy["Metro"] = df_xy["series_id"].apply(get_msa_name)
            title_xy = f"Alpha vs. Beta ({xy_start_dt.strftime('%Y-%m')} to {xy_end_dt.strftime('%Y-%m')})"
            fig_xy = px.scatter(
                df_xy,
                x="beta",
                y="alpha",
                text="Metro",
                hover_data=["id","r_squared","start_date","end_date","created_at"],
                title=title_xy
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
            st.plotly_chart(fig_xy, use_container_width=True)

            if st.checkbox("View alpha/beta table for XY chart"):
                st.dataframe(df_xy[["id","Metro","alpha","beta","r_squared","start_date","end_date","created_at"]])

# ---------------------------------------------------------------------
# 8) TIME SERIES
# ---------------------------------------------------------------------
st.markdown("### Time Series (Rolling Alpha/Beta over start_date)")

if "df_ts" not in st.session_state:
    st.session_state["df_ts"] = None
if "fig_ts" not in st.session_state:
    st.session_state["fig_ts"] = None

time_msas = st.multiselect(
    "Pick up to 5 MSAs for Time Series:",
    options=[s for s in all_series_ids if s != NATIONAL_SERIES_ID],
    max_selections=5
)

ab_choice = st.selectbox("Which metric to graph in the Time Series?", ["alpha","beta"])

ts_default_start_year = 2019
ts_default_end_year   = 2024

st.write("#### Date Range for Time Series (filters on `start_date`)")
col_t1, col_t2 = st.columns(2)
with col_t1:
    ts_start_month = st.selectbox("Start Month (Time Series)", months, index=0)
    ts_start_year  = st.selectbox("Start Year (Time Series)", years_xy, index=year_index(ts_default_start_year))
with col_t2:
    ts_end_month   = st.selectbox("End Month (Time Series)", months, index=11)
    ts_end_year    = st.selectbox("End Year (Time Series)", years_xy, index=year_index(ts_default_end_year))

ts_smonth_num = months.index(ts_start_month) + 1
ts_emonth_num = months.index(ts_end_month) + 1
ts_start_dt   = datetime(ts_start_year, ts_smonth_num, 1)
ts_end_dt     = datetime(ts_end_year,   ts_emonth_num, 1)

if st.button("Compute Time Series"):
    if not time_msas:
        st.warning("Pick at least 1 MSA (up to 5).")
    else:
        df_ts = df_raw[df_raw["series_id"].isin(time_msas)].copy()
        df_ts = df_ts[(df_ts["start_date"] >= ts_start_dt) & (df_ts["start_date"] <= ts_end_dt)]
        if df_ts.empty:
            st.warning("No rows found for that range/MSAs.")
        else:
            # Pivot => index= start_date, columns= MSA name, values= alpha or beta
            df_ts["Metro"] = df_ts["series_id"].apply(get_msa_name)
            pivot_df = df_ts.pivot(index="start_date", columns="Metro", values=ab_choice)
            pivot_df.sort_index(inplace=True)

            if pivot_df.empty:
                st.warning("Pivot is empty. Possibly only 1 row per MSA or no coverage.")
            else:
                title_ts = f"Time Series of {ab_choice.title()} from {ts_start_dt.strftime('%Y-%m')} to {ts_end_dt.strftime('%Y-%m')}"
                fig_ts = px.line(
                    pivot_df,
                    x=pivot_df.index,
                    y=pivot_df.columns,
                    title=title_ts
                )
                fig_ts.update_layout(
                    dragmode='pan',
                    xaxis_title="start_date",
                    yaxis_title=ab_choice.title(),
                    title_x=0.5,
                    title_xanchor="center",
                    legend_title="Metro"
                )
                st.session_state["df_ts"] = pivot_df
                st.session_state["fig_ts"] = fig_ts

if st.session_state.get("fig_ts") is not None:
    st.plotly_chart(
        st.session_state["fig_ts"],
        use_container_width=True,
        config={"scrollZoom": True}
    )
    if st.checkbox("Show time-series data table"):
        st.dataframe(st.session_state["df_ts"])

# ---------------------------------------------------------------------
# 9) MSA MAP
# ---------------------------------------------------------------------
st.markdown("### MSA Map")

st.write("Select a date range, plus whether to color by alpha or beta. We'll do a scatter map of MSAs that fall in that range. We assume `data/msa_map_coords.csv` has `series_id, lat, lon`.")

MAP_CSV = "data/msa_map_coords.csv"

@st.cache_data
def load_map_coords():
    dfm = pd.read_csv(MAP_CSV)
    needed_map = {"series_id","lat","lon"}
    missing_map = needed_map - set(dfm.columns)
    if missing_map:
        st.error(f"Map coords CSV is missing columns: {missing_map}")
        st.stop()
    return dfm

try:
    df_mapcoords = load_map_coords()
except:
    df_mapcoords = pd.DataFrame()
    st.info("Could not load `data/msa_map_coords.csv`. Map won't work if file is missing.")


map_col_choice = st.selectbox("Which metric to color by on the map?", ["alpha","beta"])

map_default_start_year = 2019
map_default_end_year   = 2024
col_m1, col_m2 = st.columns(2)
with col_m1:
    map_start_month = st.selectbox("Start Month (Map)", months, index=0)
    map_start_year  = st.selectbox("Start Year (Map)", years_xy, index=year_index(map_default_start_year))
with col_m2:
    map_end_month   = st.selectbox("End Month (Map)", months, index=11)
    map_end_year    = st.selectbox("End Year (Map)", years_xy, index=year_index(map_default_end_year))

map_smonth_num = months.index(map_start_month) + 1
map_emonth_num = months.index(map_end_month) + 1
map_start_dt   = datetime(map_start_year, map_smonth_num, 1)
map_end_dt     = datetime(map_end_year,   map_emonth_num, 1)

if st.button("Generate MSA Map"):
    if df_mapcoords.empty:
        st.warning("No map coords loaded. Check `data/msa_map_coords.csv`.")
    else:
        # Filter main df by date
        df_map = df_raw[(df_raw["start_date"] >= map_start_dt) & (df_raw["start_date"] <= map_end_dt)].copy()
        if df_map.empty:
            st.warning("No data for that date range.")
        else:
            # We'll just pick the "latest" row per MSA or something. Or if multiple, we take the first?
            # For simplicity, let's group by series_id and pick the row with the max start_date
            # in the range. Adjust if you want a different logic.
            df_map.sort_values("start_date", inplace=True)
            df_map = df_map.groupby("series_id").tail(1)  # last row for each MSA

            # Merge in lat/lon
            df_map = pd.merge(df_map, df_mapcoords, on="series_id", how="inner")
            if df_map.empty:
                st.warning("None of the chosen MSAs have lat/lon in `msa_map_coords.csv`.")
            else:
                # We'll do px.scatter_mapbox or px.scatter_geo.
                # Let's do scatter_mapbox for a nicer look. Need a Mapbox token or use default styling.
                # If you don't have a token, we can do scatter_geo. Example:
                fig_map = px.scatter_geo(
                    df_map,
                    lat="lat",
                    lon="lon",
                    color=map_col_choice,  # "alpha" or "beta"
                    hover_data=["series_id","alpha","beta","r_squared","start_date","end_date","created_at"],
                    text="series_id",
                    projection="natural earth",
                    title=f"MSA Map colored by {map_col_choice} ({map_start_dt.strftime('%Y-%m')} to {map_end_dt.strftime('%Y-%m')})"
                )
                fig_map.update_traces(textposition='top center')
                fig_map.update_layout(
                    title_x=0.5,
                    title_xanchor='center'
                )
                st.plotly_chart(fig_map, use_container_width=True)


