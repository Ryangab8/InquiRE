"""
alpha_beta_app.py

Usage:
    streamlit run alpha_beta_app.py

Description:
    - Connects to alpha_beta_results in your "inquire_DB" Postgres database.
    - Loads alpha/beta for all MSA series (and national).
    - Displays an XY scatter with Beta on X-axis, Alpha on Y-axis, labeled by MSA name.
"""

import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px

# ------------------------------------------------------------------
# 1) DATABASE CONNECTION SETTINGS
# ------------------------------------------------------------------
DB_HOST = "localhost"  # or "localhost"
DB_PORT = "5433"       # the port your working Postgres is on
DB_NAME = "inquire_DB"
DB_USER = "postgres"
DB_PASS = "givedata"

# ------------------------------------------------------------------
# 2) MSA NAME MAP (plus "National" for CES0000000001)
# ------------------------------------------------------------------
name_map = {
    # National series
    "CES0000000001": "National",
    # 1) New York–Newark–Jersey City, NY–NJ–PA
    "SMS36356200000000001": "NYC Metro",
    # 2) Los Angeles–Long Beach–Anaheim, CA
    "SMS06310800000000001": "LA Metro",
    # 3) Chicago–Naperville–Elgin, IL–IN–WI
    "SMS17169800000000001": "Chicago Metro",
    # 4) Dallas–Fort Worth–Arlington, TX
    "SMS48191000000000001": "Dallas–Fort Worth",
    # 5) Houston–The Woodlands–Sugar Land, TX
    "SMS48264200000000001": "Houston",
    # 6) Washington–Arlington–Alexandria, DC–VA–MD–WV
    "SMS11479000000000001": "Washington DC",
    # 7) Philadelphia–Camden–Wilmington, PA–NJ–DE–MD
    "SMS42979610000000001": "Philadelphia",
    # 8) Miami–Fort Lauderdale–West Palm Beach, FL
    "SMS12331000000000001": "Miami",
    # 9) Atlanta–Sandy Springs–Roswell, GA
    "SMS13120600000000001": "Atlanta",
    # 10) Phoenix–Mesa–Scottsdale, AZ
    "SMS04380600000000001": "Phoenix",
    # 11) Boston–Cambridge–Newton, MA–NH
    "SMS25716540000000026": "Boston",
    # 12) San Francisco–Oakland–Berkeley, CA
    "SMS06418840000000001": "San Francisco–Oakland",
    # 13) Riverside–San Bernardino–Ontario, CA
    "SMS06401400000000001": "Riverside–San Bernardino",
    # 14) Detroit–Warren–Dearborn, MI
    "SMS26198200000000001": "Detroit",
    # 15) Seattle–Tacoma–Bellevue, WA
    "SMS53426600000000001": "Seattle",
    # 16) Minneapolis–St. Paul–Bloomington, MN–WI
    "SMS27334600000000001": "Minneapolis–St. Paul",
    # 17) San Diego–Chula Vista–Carlsbad, CA
    "SMS06417400000000001": "San Diego",
    # 18) Tampa–St. Petersburg–Clearwater, FL
    "SMS12453000000000001": "Tampa",
    # 19) Denver–Aurora–Lakewood, CO
    "SMS08197400000000001": "Denver",
    # 20) St. Louis, MO–IL
    "SMS29411800000000001": "St. Louis",
    # 21) Baltimore–Columbia–Towson, MD
    "SMS24925810000000001": "Baltimore",
    # 22) Charlotte–Concord–Gastonia, NC–SC
    "SMS37167400000000001": "Charlotte",
    # 23) Orlando–Kissimmee–Sanford, FL
    "SMS12367400000000001": "Orlando",
    # 24) San Antonio–New Braunfels, TX
    "SMS48417000000000001": "San Antonio",
    # 25) Portland–Vancouver–Hillsboro, OR–WA
    "SMS41389000000000001": "Portland",
    # 26) Pittsburgh, PA
    "SMS42383000000000001": "Pittsburgh",
    # 27) Sacramento–Roseville–Arden-Arcade, CA
    "SMS06409000000000001": "Sacramento",
    # 28) Las Vegas–Henderson–Paradise, NV
    "SMS32298200000000001": "Las Vegas",
    # 29) Cincinnati, OH–KY–IN
    "SMS39171400000000001": "Cincinnati",
    # 30) Kansas City, MO–KS
    "SMS20928120000000001": "Kansas City",
    # 31) Columbus, OH
    "SMS18180200000000001": "Columbus",
    # 32) Indianapolis–Carmel–Anderson, IN
    "SMS18269000000000001": "Indianapolis",
    # 33) Cleveland–Elyria, OH
    "SMS39174600000000001": "Cleveland",
    # 34) San Jose–Sunnyvale–Santa Clara, CA
    "SMS06419400000000001": "San Jose",
    # 35) Nashville–Davidson–Murfreesboro–Franklin, TN
    "SMS47349800000000001": "Nashville",
    # 36) Virginia Beach–Norfolk–Newport News, VA–NC
    "SMS51472600000000001": "Virginia Beach–Norfolk",
    # 37) Providence–Warwick, RI–MA
    "SMS44772000000000001": "Providence–Warwick",
    # 38) Milwaukee–Waukesha–West Allis, WI
    "SMS55333400000000001": "Milwaukee",
    # 39) Jacksonville, FL
    "SMS12272600000000001": "Jacksonville",
    # 40) Memphis, TN–MS–AR
    "SMS47328200000000001": "Memphis",
    # 41) Richmond, VA
    "SMS51400600000000001": "Richmond",
    # 42) Oklahoma City, OK
    "SMS40364200000000001": "Oklahoma City",
    # 43) Hartford–East Hartford–West, CT (Suspect ID)
    "SMU04380600000000001": "Hartford (?)",
    # 44) New Orleans–Metairie, LA
    "SMS22353800000000001": "New Orleans",
    # 45) Buffalo–Cheektowaga–Niagara Falls, NY
    "SMS36153800000000001": "Buffalo–Cheektowaga",
    # 46) Raleigh, NC
    "SMS37395800000000001": "Raleigh",
    # 47) Birmingham–Hoover, AL
    "SMS01138200000000001": "Birmingham–Hoover",
    # 48) Salt Lake City, UT
    "SMS49416200000000001": "Salt Lake City",
    # 49) Rochester, NY
    "SMS36403800000000001": "Rochester",
    # 50) Louisville/Jefferson County, KY–IN
    "SMS21311400000000001": "Louisville"
}

def load_alpha_beta_data():
    """
    Query alpha_beta_results table into a pandas DataFrame.
    Expects columns: [series_id, alpha, beta, r_squared, start_date, end_date].
    """
    conn = psycopg2.connect(
        host="localhost",
        port="5433",
        dbname="inquire_DB",
        user="postgres",
        password="givedata"
    )
    query = """
        SELECT series_id, alpha, beta, r_squared, start_date, end_date
        FROM alpha_beta_results
        ORDER BY series_id
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Map each series_id to a friendly name, defaulting to the ID if not found
    df["msa_name"] = df["series_id"].apply(lambda sid: name_map.get(sid, sid))

    return df

def main():
    st.title("Alpha–Beta Chart (Nonfarm Employment)")

    df = load_alpha_beta_data()
    st.write(f"Loaded {len(df)} records from alpha_beta_results.")

    # We'll do a scatter plot with beta on x-axis, alpha on y-axis
    fig = px.scatter(
        df,
        x="beta",
        y="alpha",
        text="msa_name",         # label each point by MSA name
        hover_data=["r_squared", "start_date", "end_date", "series_id"]
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(
        xaxis_title="Beta",
        yaxis_title="Alpha",
        title="Alpha vs. Beta for 50 MSAs & National"
    )

    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()