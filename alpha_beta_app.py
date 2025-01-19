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
        - This CSV has columns: `id, series_id, alpha, beta, r_squared, start_date, end_date, created_at`.
        - Each row already contains a final alpha/beta for that MSA.

        **Alpha**  
        - Measures growth rate relative to a national benchmark. Positive = outperformance, Negative = underperformance.

        **Beta**  
        - Measures volatility or sensitivity relative to the national trend. 1 = in sync, >1 = more volatile, <1 = less volatile.
        """
    )

with st.expander("How To Use", expanded=False):
    st.markdown(
        """
        **Step 1:** We read from `data/nonfarm_data.csv` (adjust if needed).  
        **Step 2:** The XY Chart tab plots final alpha/beta values for user-chosen MSAs + date range.  
        **Step 3:** The Time Series tab attempts to plot alpha or beta over `start_date` if multiple rows per MSA exist.  
        **Step 4:** Optionally show the underlying data table for each chart.
        """
    )

# ---------------------------------------------------------------------
# 5) Global Metric Selector (not very relevant here, but kept for UI consistency)
# ---------------------------------------------------------------------
metric_choice = st.selectbox(
    "Select a metric:",
    ["Total NonFarm Employment - Seasonally Adjusted"]
)

# ---------------------------------------------------------------------
# 6) Load CSV with new columns
# ---------------------------------------------------------------------
CSV_FILE = "data/nonfarm_data.csv"  # Adjust if needed

@st.cache_data
def load_csv():
    df = pd.read_csv(CSV_FILE)
    # Confirm we have the needed columns
    needed = {"id","series_id","alpha","beta","r_squared","start_date","end_date","created_at"}
    missing = needed - set(df.columns)
    if missing:
        st.error(f"CSV missing columns: {missing}")
        st.stop()

    # Convert start_date, end_date to datetime
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"]   = pd.to_datetime(df["end_date"],   errors="coerce")

    # Just sort by start_date (optional)
    df.sort_values(["series_id","start_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df_raw = load_csv()

# Make a list of all unique series_ids
all_series_ids = sorted(df_raw["series_id"].unique())

# We'll keep a reference for "National" if you want, else can remove
NATIONAL_SERIES_ID = "CES0000000001"

# Mapping (optional) if you want friendlier names
MSA_NAME_MAP = {
    "CES0000000001": "National",
    # Add more if desired
}

# ---------------------------------------------------------------------
# 7) XY Chart (Alpha/Beta) with user-chosen date range filter
# ---------------------------------------------------------------------
st.markdown("### XY Chart (Alpha vs. Beta)")

if "msa_selection" not in st.session_state:
    st.session_state["msa_selection"] = []

st.multiselect(
    "Pick MSA(s):",
    options=all_series_ids,
    key="msa_selection"
)

colA, colB = st.columns(2)
colA.button("Select All", on_click=lambda: st.session_state.update({"msa_selection": all_series_ids}))
colB.button("Clear", on_click=lambda: st.session_state.update({"msa_selection": []}))

# The code originally had a year-month date range. We'll interpret them
# as filters on the CSV's "start_date" column.
months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
years_list = list(range(1990, 2031))
default_start_year = 2019
default_end_year   = 2024

def get_year_index(year):
    try:
        return years_list.index(year)
    except ValueError:
        return 0  # fallback

st.write("#### Date Range for XY Chart (based on start_date)")

col_s1, col_s2 = st.columns(2)
with col_s1:
    xy_start_month = st.selectbox("Start Month (XY)", months, index=0)
    xy_start_year  = st.selectbox("Start Year (XY)", years_list, index=get_year_index(default_start_year))
with col_s2:
    xy_end_month   = st.selectbox("End Month (XY)", months, index=11)
    xy_end_year    = st.selectbox("End Year (XY)", years_list, index=get_year_index(default_end_year))

# Build a datetime from user picks
xy_smonth_num = months.index(xy_start_month) + 1
xy_emonth_num = months.index(xy_end_month) + 1
xy_start_dt = datetime(xy_start_year, xy_smonth_num, 1)
xy_end_dt   = datetime(xy_end_year,   xy_emonth_num, 1)

if st.button("Generate XY Chart"):
    if not st.session_state["msa_selection"]:
        st.warning("No MSAs selected!")
    else:
        # Filter the CSV by the chosen MSAs + date range
        chosen_msas = st.session_state["msa_selection"]
        df_xy = df_raw[df_raw["series_id"].isin(chosen_msas)].copy()

        # Filter by start_date between xy_start_dt and xy_end_dt
        df_xy = df_xy[(df_xy["start_date"] >= xy_start_dt) & (df_xy["start_date"] <= xy_end_dt)]

        if df_xy.empty:
            st.error("No rows found for that range/MSAs.")
        else:
            # Plot alpha vs. beta
            # We'll label by series_id, hover w/ r_squared, start_date, end_date, created_at, id, etc.
            fig_xy = px.scatter(
                df_xy,
                x="beta",
                y="alpha",
                text="series_id",
                hover_data=["id","r_squared","start_date","end_date","created_at"],
                title=f"Alpha vs. Beta ({xy_start_dt.strftime('%Y-%m')} to {xy_end_dt.strftime('%Y-%m')})",
                labels={"beta": "Beta", "alpha": "Alpha"}
            )
            fig_xy.update_traces(textposition='top center', textfont_size=14)
            fig_xy.update_layout(dragmode='pan', title_x=0.5, title_xanchor='center')

            st.plotly_chart(fig_xy, use_container_width=True)

            if st.checkbox("View alpha/beta table for XY chart"):
                st.dataframe(df_xy)

# ---------------------------------------------------------------------
# 8) TIME SERIES (Rolling or Simple Over start_date)
# ---------------------------------------------------------------------
st.markdown("### Time Series (Alpha or Beta)")

if "df_ts" not in st.session_state:
    st.session_state["df_ts"] = None

if "fig_ts" not in st.session_state:
    st.session_state["fig_ts"] = None

time_msas = st.multiselect(
    "Pick up to 5 MSAs for Time Series:",
    options=[m for m in all_series_ids if m != NATIONAL_SERIES_ID],
    max_selections=5
)

ab_choice = st.selectbox("Which metric to graph in the Time Series?", ["alpha","beta"])

st.write("#### Date Range for Time Series (based on start_date)")
col_t3, col_t4 = st.columns(2)
with col_t3:
    ts_start_month = st.selectbox("Start Month (Time Series)", months, index=0)
    ts_start_year  = st.selectbox("Start Year (Time Series)", years_list, index=get_year_index(default_start_year))
with col_t4:
    ts_end_month   = st.selectbox("End Month (Time Series)", months, index=11)
    ts_end_year    = st.selectbox("End Year (Time Series)", years_list, index=get_year_index(default_end_year))

ts_smonth_num = months.index(ts_start_month) + 1
ts_emonth_num = months.index(ts_end_month) + 1
ts_start_dt   = datetime(ts_start_year, ts_smonth_num, 1)
ts_end_dt     = datetime(ts_end_year,   ts_emonth_num, 1)

def pivot_time_series(df_in, chosen_metric):
    """
    We pivot so index= start_date, columns= series_id, values= [alpha or beta].
    """
    if df_in.empty:
        return pd.DataFrame()
    return df_in.pivot(index="start_date", columns="series_id", values=chosen_metric)

if st.button("Compute Time Series"):
    if not time_msas:
        st.warning("Pick at least 1 MSA (up to 5).")
    else:
        df_ts = df_raw[df_raw["series_id"].isin(time_msas)].copy()
        # Filter by start_date
        df_ts = df_ts[(df_ts["start_date"] >= ts_start_dt) & (df_ts["start_date"] <= ts_end_dt)]

        if df_ts.empty:
            st.warning("No rows found for that range/MSAs.")
        else:
            # pivot => index= start_date, columns= series_id, values= alpha or beta
            df_ts_pivot = pivot_time_series(df_ts, ab_choice)
            if df_ts_pivot.empty:
                st.warning("Time series pivot is empty; maybe only 1 row per MSA.")
            else:
                # We'll do a line chart
                fig_ts = px.line(
                    df_ts_pivot,
                    x=df_ts_pivot.index,
                    y=df_ts_pivot.columns,
                    title=f"Time Series of {ab_choice.title()} from {ts_start_dt.strftime('%Y-%m')} to {ts_end_dt.strftime('%Y-%m')}"
                )
                fig_ts.update_layout(
                    xaxis_title="start_date",
                    yaxis_title=ab_choice.title(),
                    title_x=0.5,
                    title_xanchor="center",
                    legend_title="series_id"
                )
                st.session_state["df_ts"] = df_ts_pivot
                st.session_state["fig_ts"] = fig_ts

if st.session_state.get("fig_ts") is not None:
    st.plotly_chart(st.session_state["fig_ts"], use_container_width=True)
    if st.checkbox("Show time-series data table"):
        st.dataframe(st.session_state["df_ts"])
