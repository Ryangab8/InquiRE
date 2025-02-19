# ---------------------------------------------------------------------
# 9) LINE CHART: HISTORICAL + FORECASTED JOB GROWTH
# ---------------------------------------------------------------------
import pandas as pd
import numpy as np
import plotly.express as px
import datetime

st.markdown("### Historical + Forecasted Job Growth (Line Chart)")

st.write("""
1. Select a **historical** date range of at least 5 years to estimate alpha/beta.  
2. Choose MSA(s).  
3. Input up to 3 future **monthly** growth scenarios for the **national** rate (e.g., "0.2" for +0.2% per month).  
4. Pick how many months forward to forecast.  
We'll plot **solid** lines for historical data and **dotted** lines for the forecast portion.
""")

# 1) Historical date range (>=5 years)
lc_col1, lc_col2 = st.columns(2)
with lc_col1:
    hist_start_year = st.selectbox("Historical Start Year", range(1990, 2030), index=5)
    hist_start_month = st.selectbox("Historical Start Month", months, index=0)
with lc_col2:
    hist_end_year = st.selectbox("Historical End Year", range(1990, 2030), index=25)
    hist_end_month = st.selectbox("Historical End Month", months, index=11)

hist_smonth_num = months.index(hist_start_month) + 1
hist_emonth_num = months.index(hist_end_month) + 1

hist_start_ym = f"{hist_start_year:04d}-{hist_smonth_num:02d}"
hist_end_ym   = f"{hist_end_year:04d}-{hist_emonth_num:02d}"

# 2) MSA selection
line_msas = st.multiselect(
    "Pick MSA(s) to compare vs. national:",
    options=sorted(INVERTED_MAP.keys()),
    help="We'll plot each MSA's historical job growth + forecast lines if scenarios are provided."
)

# 3) Up to 3 scenario inputs (monthly growth rates for national)
st.write("#### Enter up to 3 monthly growth scenarios (%) for the forecast:")
sc_line1 = st.text_input("Scenario #1", value="0.2")
sc_line2 = st.text_input("Scenario #2 (optional)", value="0.5")
sc_line3 = st.text_input("Scenario #3 (optional)", value="1.0")

# 4) Forecast horizon (months)
forecast_months = st.number_input(
    "How many months forward to forecast?",
    min_value=1, max_value=60, value=12
)

if st.button("Generate Growth Chart"):
    # Validate 5-year rule
    start_dt = datetime.datetime(hist_start_year, hist_smonth_num, 1)
    end_dt   = datetime.datetime(hist_end_year, hist_emonth_num, 1)
    months_apart = (end_dt.year - start_dt.year)*12 + (end_dt.month - start_dt.month)
    if months_apart < 60:
        st.error("Please select at least a 5-year historical range.")
    elif not line_msas:
        st.warning("No MSAs selected.")
    else:
        # Parse scenario floats
        def parse_scenario(sval):
            try:
                return float(sval)
            except:
                return None

        scenarios = []
        s1 = parse_scenario(sc_line1)
        s2 = parse_scenario(sc_line2)
        s3 = parse_scenario(sc_line3)
        if s1 is not None: scenarios.append(("Scenario #1", s1))
        if s2 is not None: scenarios.append(("Scenario #2", s2))
        if s3 is not None: scenarios.append(("Scenario #3", s3))

        # Historical data fetch
        # We'll always include the NATIONAL_SERIES_ID too
        chosen_ids = [INVERTED_MAP[m] for m in line_msas if m != "National"]
        chosen_ids.append(NATIONAL_SERIES_ID)
        chosen_ids = list(set(chosen_ids))  # remove duplicates

        df_raw_line = fetch_raw_data_multiple(chosen_ids, hist_start_ym, hist_end_ym)
        if df_raw_line.empty:
            st.warning("No data found. Check CSV or date range.")
        else:
            # Convert to monthly % growth
            df_raw_line["value"] = pd.to_numeric(df_raw_line["value"], errors="coerce")
            pivot_line = df_raw_line.pivot(index="obs_date", columns="series_id", values="value")
            pivot_line.sort_index(inplace=True)
            growth_line = pivot_line.pct_change(1) * 100
            growth_line.dropna(inplace=True)

            if NATIONAL_SERIES_ID not in growth_line.columns:
                st.error("National series not found in the dataset. Check data.")
            else:
                # Compute alpha/beta for each MSA vs. national
                results_ab = []
                nat_col = NATIONAL_SERIES_ID

                for sid in growth_line.columns:
                    if sid == NATIONAL_SERIES_ID:
                        continue
                    merged = growth_line[[nat_col, sid]].dropna()
                    if len(merged) < 2:
                        continue
                    X = sm.add_constant(merged[nat_col])
                    y = merged[sid]
                    model = sm.OLS(y, X).fit()

                    alpha_val = model.params["const"]
                    beta_val  = model.params[nat_col]
                    r_sq      = model.rsquared

                    results_ab.append({
                        "series_id": sid,
                        "Metro": MSA_NAME_MAP.get(sid, sid),
                        "Alpha": alpha_val,
                        "Beta": beta_val,
                        "R-Squared": r_sq
                    })

                df_ab_line = pd.DataFrame(results_ab)
                if df_ab_line.empty:
                    st.error("Could not compute alpha/beta. Possibly insufficient overlap.")
                else:
                    # Build a table for quick reference
                    st.write("#### Alpha/Beta Confidence Table")
                    st.dataframe(df_ab_line[["Metro","Alpha","Beta","R-Squared"]])

                    # ---- Build the line chart data for actual historical growth ----
                    # We'll produce a DataFrame with columns: Date, Growth, MSA, Scenario, line_style
                    # Historical portion => scenario="Historical", line_style="solid"
                    # We'll store the national line & each MSA line
                    line_records = []

                    # Include all columns from growth_line (since user might want the national line too)
                    for dt_idx in growth_line.index:
                        for col_sid in growth_line.columns:
                            val = growth_line.loc[dt_idx, col_sid]
                            # Metro name
                            if col_sid == NATIONAL_SERIES_ID:
                                metro_name = "National"
                            else:
                                metro_name = MSA_NAME_MAP.get(col_sid, col_sid)
                            # only add if user selected that MSA or it's National
                            if metro_name in line_msas or metro_name == "National":
                                line_records.append({
                                    "Date": dt_idx,
                                    "Growth": val,
                                    "Metro": metro_name,
                                    "Scenario": "Historical",
                                    "LineStyle": "solid"
                                })

                    # ---- Now build forecast portion for next X months ----
                    # We'll define the last_dt as the max of growth_line.index
                    # Then create monthly steps up to forecast_months
                    last_dt_hist = growth_line.index.max()
                    # If user gave multiple scenarios, we produce multiple lines
                    for sc_label, sc_val in scenarios:
                        # We'll define the national's growth as sc_val each month,
                        # and each MSA's growth as alpha + beta * sc_val (from df_ab_line).
                        # We need a row for "National" as well, so the chart shows the forecast national line.

                        # Convert monthly steps
                        forecast_dates = []
                        for i in range(1, forecast_months+1):
                            future_date = last_dt_hist + pd.DateOffset(months=i)
                            forecast_dates.append(future_date)

                        # Add line for National
                        for fdt in forecast_dates:
                            line_records.append({
                                "Date": fdt,
                                "Growth": sc_val,  # user scenario
                                "Metro": "National",
                                "Scenario": sc_label,
                                "LineStyle": "dot"
                            })

                        # Add lines for each MSA
                        for idx, rowm in df_ab_line.iterrows():
                            metro = rowm["Metro"]
                            alpha_v = rowm["Alpha"]
                            beta_v  = rowm["Beta"]
                            # Only forecast if user included that MSA or it's "National"
                            if metro in line_msas:
                                for fdt in forecast_dates:
                                    # predicted MSA growth
                                    forecast_msa_g = alpha_v + beta_v * sc_val
                                    line_records.append({
                                        "Date": fdt,
                                        "Growth": forecast_msa_g,
                                        "Metro": metro,
                                        "Scenario": sc_label,
                                        "LineStyle": "dot"
                                    })

                    df_line_final = pd.DataFrame(line_records)
                    # We want to plot with x=Date, y=Growth, color=Metro, line_dash=Scenario or line style

                    # We'll do color=Metro, but also differ the line dash: "Historical" => solid, else => dot
                    # Plotly can do "line_dash" with a column, so we'll map "Historical" => "solid", scenario => "dot"
                    dash_map = {
                        "Historical": "solid"
                        # We'll just let "Scenario #1" => "dot" etc. automatically
                    }

                    fig_line = px.line(
                        df_line_final,
                        x="Date",
                        y="Growth",
                        color="Metro",
                        line_dash="Scenario",  # or line_dash="LineStyle"
                        title="Job Growth: Historical (solid) + Forecast (dotted)"
                    )
                    # force national line black?
                    # We can do a custom color_discrete_map if you like:
                    color_map = {}
                    color_map["National"] = "black"
                    # everything else let plotly pick
                    fig_line.update_traces(connectgaps=True)
                    fig_line.update_layout(legend_title_text="Metro + Scenario")

                    # We'll forcibly set dash styles:
                    # By default, plotly might use different dashes for scenario #1, #2, #3
                    # If you want EXACT styling, you can do:
                    # fig_line.update_traces(...), but you'll need to filter each scenario in a loop, etc.

                    st.plotly_chart(fig_line, use_container_width=True)

                    st.markdown("""
                    **Notes:**  
                    - Solid lines = Historical actual monthly growth.  
                    - Dotted lines = Forecast portion. National scenario growth is constant at your input (e.g. 0.2% per month).  
                    - MSA forecast = alpha + beta * (scenario).  
                    - Check "R-Squared" above to see how strongly the MSA historically tracked the nation (higher = more confidence).
                    """)


