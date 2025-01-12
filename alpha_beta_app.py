import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

def main():
    st.title("Alpha vs. Beta (Month-over-Month Growth, 2021–2023)")
    
    # ----- 1) Define your BLS API key and series IDs
    BLS_API_KEY = "f232fdbb532d456b8ed8ca7ce2a1cbb2"

    national_id = "CES0000000001"  # National total nonfarm, SA
    msa_ids = [
        "SMS36356200000000001",  # NYC
        "SMS06310800000000001",  # LA
        "SMS17169800000000001",  # Chicago
        "SMS48191000000000001",  # Dallas–Fort Worth
        "SMS48264200000000001",  # Houston
        "SMS11479000000000001",  # Washington DC
        "SMS42979610000000001",  # Philadelphia
        "SMS12331000000000001",  # Miami
        "SMS13120600000000001",  # Atlanta
        "SMS04380600000000001",  # Phoenix
    ]
    
    name_map = {
        "SMS36356200000000001": "NYC Metro",
        "SMS06310800000000001": "LA Metro",
        "SMS17169800000000001": "Chicago Metro",
        "SMS48191000000000001": "Dallas–Fort Worth",
        "SMS48264200000000001": "Houston",
        "SMS11479000000000001": "Washington DC",
        "SMS42979610000000001": "Philadelphia",
        "SMS12331000000000001": "Miami",
        "SMS13120600000000001": "Atlanta",
        "SMS04380600000000001": "Phoenix",
        national_id: "National"
    }
    
    all_series = [national_id] + msa_ids
    
    # ----- 2) Fetch monthly data from BLS (2021–2023)
    payload = {
        "seriesid": all_series,
        "startyear": "2021",
        "endyear": "2023",
        "registrationkey": BLS_API_KEY
    }
    
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    headers = {"Content-type": "application/json"}
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    response.raise_for_status()
    data_json = response.json()
    
    rows = []
    for series_item in data_json["Results"]["series"]:
        sid = series_item["seriesID"]
        for datum in series_item["data"]:
            period = datum["period"]
            if period.startswith("M") and period != "M13":
                year = int(datum["year"])
                month = int(period[1:])
                value = float(datum["value"])
                rows.append({
                    "series_id": sid,
                    "year": year,
                    "month": month,
                    "value": value
                })
    
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
    df.sort_values(["series_id","date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # ----- 3) Pivot and compute MoM % change
    df_pivot = df.pivot(index="date", columns="series_id", values="value")
    df_growth = df_pivot.pct_change(1) * 100  # MoM percent
    df_growth.dropna(inplace=True)
    
    # ----- 4) Regress for alpha/beta
    alpha_beta_results = []
    for msa_id in msa_ids:
        merged = df_growth[[national_id, msa_id]].dropna()
        X = sm.add_constant(merged[national_id])  # alpha
        y = merged[msa_id]
        model = sm.OLS(y, X).fit()
        alpha, beta = model.params["const"], model.params[national_id]
        alpha_beta_results.append({
            "msa_id": msa_id,
            "msa_name": name_map.get(msa_id, msa_id),
            "alpha": alpha,
            "beta": beta
        })
    
    results_df = pd.DataFrame(alpha_beta_results)
    
    # ----- 5) Plotly chart
    fig = px.scatter(
        results_df,
        x="beta",
        y="alpha",
        text="msa_name",
        title="Alpha vs. Beta (Month-over-Month Growth, 2021–2023)",
        labels={
            "beta": "Beta (slope vs. National MoM%)",
            "alpha": "Alpha (intercept, in %)"
        }
    )
    fig.update_traces(textposition='top center')
    
    # ----- 6) Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Streamlit entry point
if __name__ == "__main__":
    main()
