import streamlit as st
import pandas as pd
import plotly.express as px

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
        - Each row represents a final alpha/beta result for a given MSA or region.

        **Alpha**  
        - Measures growth rate relative to a national benchmark. Positive = outperformance, Negative = underperformance.

        **Beta**  
        - Measures volatility/sensitivity relative to the national trend. 1 = in sync, >1 = more volatile, <1 = less volatile.
        """
    )

with st.expander("How To Use", expanded=False):
    st.markdown(
        """
        **Step 1:** Ensure your CSV is named `data/nonfarm_data.csv` or adjust the code below.  
        **Step 2:** This app reads those final alpha/beta values directly.  
        **Step 3:** You can select one or more MSAs (by `series_id`) to visualize.  
        **Step 4:** Click “Generate XY Chart” to see Alpha vs. Beta.
        **Step 5:** Optionally view the underlying CSV data in a table.
        """
    )

# ---------------------------------------------------------------------
# 5) CSV Reference
# ---------------------------------------------------------------------
CSV_FILE = "data/nonfarm_data.csv"  # Ensure it has columns: id, series_id, alpha, beta, r_squared, start_date, end_date, created_at

@st.cache_data
def load_csv():
    df = pd.read_csv(CSV_FILE)
    return df

df = load_csv()

# Verify columns
required_cols = {"id","series_id","alpha","beta","r_squared","start_date","end_date","created_at"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"CSV is missing required columns: {missing}")
    st.stop()

# ---------------------------------------------------------------------
# 6) Let user pick MSA(s) by `series_id`
# ---------------------------------------------------------------------
all_series_ids = sorted(df["series_id"].unique())

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

# ---------------------------------------------------------------------
# 7) Generate XY Chart (Alpha vs. Beta)
# ---------------------------------------------------------------------
if st.button("Generate XY Chart"):
    if not st.session_state["msa_selection"]:
        st.warning("No MSAs selected!")
    else:
        # Filter the DataFrame to only those series_ids
        filtered_df = df[df["series_id"].isin(st.session_state["msa_selection"])]

        if filtered_df.empty:
            st.error("No rows found for the chosen MSA(s).")
        else:
            # Plot alpha vs. beta, labeling each point by series_id
            fig_xy = px.scatter(
                filtered_df,
                x="beta",
                y="alpha",
                text="series_id",
                hover_data=["id","r_squared","start_date","end_date","created_at"],
                title="Alpha vs. Beta (from CSV)",
                labels={"beta": "Beta", "alpha": "Alpha"}
            )
            fig_xy.update_traces(textposition='top center', textfont_size=14)
            fig_xy.update_layout(
                dragmode='pan',
                title_x=0.5,
                title_xanchor='center'
            )
            st.plotly_chart(fig_xy, use_container_width=True)

            if st.checkbox("View alpha/beta table"):
                st.dataframe(filtered_df)

# ---------------------------------------------------------------------
# 8) (Optional) More sections / placeholders
# ---------------------------------------------------------------------
# We've removed pivot/time-series logic since your CSV already has final alpha/beta,
# so there's no monthly or date-based OLS to do in this new structure.
