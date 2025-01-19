import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

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

        **Beta**  
        - Beta measures how sensitive (or volatile) an MSA is relative to the nation’s changes.  
        - If beta > 1, the MSA is more volatile than the national baseline; if beta < 1, it is less volatile.
        """
    )

with st.expander("How To Use", expanded=False):
    st.markdown(
        """
        **Step 1:** Select the desired metric from the dropdown below (currently only one option).  

        **XY Chart (Alpha vs. Beta)**  
        1. Pick MSA(s) from the dropdown, or click “Select All.”  
        2. Choose a start and end date range.  
        3. Click “Generate XY Chart.” Each MSA is plotted by (β, α).  

        **Time Series (Rolling Alpha/Beta)**  
        1. Select up to 5 MSAs (excluding “National,” which is the benchmark).  
        2. Pick Alpha or Beta to track, plus a separate date range.  
        3. Click “Compute Time Series.” See how alpha or beta evolves month by month.  
        4. Likewise, you can zoom/pan in the chart and optionally show/hide the underlying data table.
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
# 6) Load CSV Instead of Using a Database
# ---------------------------------------------------------------------
# This CSV must have columns: id, series_id, alpha, beta, r_squared, start_date, end_date, created_at
# Adjust the path to wherever your CSV is located.
df_full = pd.read_csv("data/nonfarm_data.csv")

# Convert start_date / end_date columns to datetime
df_full["start_date"] = pd.to_datetime(df_full["start_date"])
df_full["end_date"]   = pd.to_datetime(df_full["end_date"])

# The "National" series ID and MSA name map are carried over from your original code
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
# 7) XY Chart
# ---------------------------------------------------------------------
def fetch_csv_data_multiple(msa_ids, start_ym, end_ym):
    """
    Filter the pre-loaded CSV (df_full) to only include rows for the given
    `msa_ids` whose date range intersects [start_ym, end_ym].
    """
    # Parse "YYYY-MM" -> datetime
    start_year, start_month = map(int, start_ym.split("-"))
    end_year, end_month     = map(int, end_ym.split("-"))
    start_dt = datetime.datetime(start_year, start_month, 1)
    end_dt   = datetime.datetime(end_year, end_month, 1)

    df_filtered = df_full[df_full["series_id"].isin(msa_ids)]
    # Simple filter: row's start_date >= start_dt and end_date <= end_dt
    df_filtered = df_filtered[
        (df_filtered["start_date"] >= start_dt) &
        (df_filtered["end_date"]   <= end_dt)
    ]
    return df_filtered

def create_xy_dataframe(df_subset):
    """
    Convert the filtered subset into an XY-friendly DataFrame.
    We assume each row has pre-computed alpha/beta/r_squared.
    """
    # Rename columns
    df_xy = df_subset.rename(
        columns={
            "alpha": "Alpha",
            "beta": "Beta",
            "r_squared": "R-Squared"
        }
    )
    # If multiple rows exist for the same MSA & date range, you might group or average. Example:
    df_xy = df_xy.groupby("series_id", as_index=False).agg(
        {"Alpha": "mean", "Beta": "mean", "R-Squared": "mean"}
    )

    # Add 'Metro' column from series_id
    df_xy["Metro"] = df_xy["series_id"].apply(lambda sid: MSA_NAME_MAP.get(sid, sid))
    return df_xy

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
years_xy = list(range(1990, 2025))
default_start_year = 2019
default_end_year   = 2024

st.write("#### Date Range for XY Chart")
col_s1, col_s2 = st.columns(2)
with col_s1:
    xy_start_month = st.selectbox("Start Month (XY)", months, index=0)
    xy_start_year  = st.selectbox(
        "Start Year (XY)",
        years_xy,
        index=years_xy.index(default_start_year)
    )
with col_s2:
    xy_end_month = st.selectbox("End Month (XY)", months, index=11)
    xy_end_year  = st.selectbox(
        "End Year (XY)",
        years_xy,
        index=years_xy.index(default_end_year)
    )

xy_smonth_num = months.index(xy_start_month) + 1
xy_emonth_num = months.index(xy_end_month) + 1
xy_start_ym = f"{xy_start_year:04d}-{xy_smonth_num:02d}"
xy_end_ym   = f"{xy_end_year:04d}-{xy_emonth_num:02d}"

if st.button("Generate XY Chart"):
    if not st.session_state["msa_selection"]:
        st.warning("No MSAs selected!")
    else:
        chosen_ids = [INVERTED_MAP[m] for m in st.session_state["msa_selection"]]
        df_subset = fetch_csv_data_multiple(chosen_ids, xy_start_ym, xy_end_ym)
        if df_subset.empty:
            st.warning("No data found for that range. Check inputs.")
        else:
            df_ab = create_xy_dataframe(df_subset)
            if df_ab.empty:
                st.error("No alpha/beta data to display.")
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
                fig_xy.update_layout(
                    dragmode='pan',
                    title_x=0.5,
                    title_xanchor='center'
                )
                # Add lines at Alpha=0 and Beta=1
                fig_xy.add_hline(
                    y=0, line_width=3, line_color="black", line_dash="dot",
                    annotation_text="Alpha = 0", annotation_position="top left"
                )
                fig_xy.add_vline(
                    x=1, line_width=3, line_color="black", line_dash="dot",
                    annotation_text="Beta = 1", annotation_position="bottom right"
                )

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

all_msa_names_no_nat = [m for m in all_msa_names if m != "National"]
selected_time_msas = st.multiselect(
    "Pick up to 5 MSAs for Time Series:",
    options=all_msa_names_no_nat,
    max_selections=5
)

ab_choice = st.selectbox("Which metric to graph in the Time Series?", ["alpha", "beta"])

ts_default_start_year = 2019
ts_default_end_year   = 2024

st.write("#### Date Range for Time Series")
col_t1, col_t2 = st.columns(2)
with col_t1:
    ts_start_month = st.selectbox("Start Month (Time Series)", months, index=0)
    ts_start_year  = st.selectbox(
        "Start Year (Time Series)",
        years_xy,
        index=years_xy.index(ts_default_start_year)
    )
with col_t2:
    ts_end_month = st.selectbox("End Month (Time Series)", months, index=11)
    ts_end_year  = st.selectbox(
        "End Year (Time Series)",
        years_xy,
        index=years_xy.index(ts_default_end_year)
    )

ts_smonth_num = months.index(ts_start_month) + 1
ts_emonth_num = months.index(ts_end_month) + 1
ts_start_ym = f"{ts_start_year:04d}-{ts_smonth_num:02d}"
ts_end_ym   = f"{ts_end_year:04d}-{ts_emonth_num:02d}"

def fetch_csv_data_time_series(msa_ids, start_ym, end_ym):
    """
    Similar to fetch_csv_data_multiple, but we won't group by MSA here.
    We'll preserve each row so we can see 'start_date' over time.
    """
    start_year, start_month = map(int, start_ym.split("-"))
    end_year, end_month     = map(int, end_ym.split("-"))
    start_dt = datetime.datetime(start_year, start_month, 1)
    end_dt   = datetime.datetime(end_year, end_month, 1)

    df_filtered = df_full[df_full["series_id"].isin(msa_ids)]
    df_filtered = df_filtered[
        (df_filtered["start_date"] >= start_dt) &
        (df_filtered["end_date"]   <= end_dt)
    ].copy()

    df_filtered.sort_values("start_date", inplace=True)
    return df_filtered

if st.button("Compute Time Series"):
    if not selected_time_msas:
        st.warning("Pick at least 1 MSA (up to 5).")
    else:
        chosen_time_ids = [INVERTED_MAP[n] for n in selected_time_msas]
        df_ts_subset = fetch_csv_data_time_series(chosen_time_ids, ts_start_ym, ts_end_ym)
        if df_ts_subset.empty:
            st.warning("No data found for that time range. Check inputs.")
        else:
            # Rename columns for clarity
            df_ts_subset["Metro"] = df_ts_subset["series_id"].apply(
                lambda sid: MSA_NAME_MAP.get(sid, sid)
            )
            df_ts_subset.rename(
                columns={
                    "alpha": "Alpha",
                    "beta": "Beta",
                    "start_date": "Date"  # We'll use start_date as the x-axis
                },
                inplace=True
            )
            # We'll plot whichever measure the user selected
            df_ts_subset["AB_Chosen"] = df_ts_subset[ab_choice.title()]

            st.session_state["df_ts"] = df_ts_subset

            title_ts = f"Time Series of {ab_choice.title()} ({ts_start_ym} to {ts_end_ym})"
            fig_ts = px.line(
                df_ts_subset,
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
