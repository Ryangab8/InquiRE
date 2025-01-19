"""
fetch_and_store_nonfarm.py

Usage:
    python fetch_and_store_nonfarm.py

Description:
    - Fetches total nonfarm (seasonally adjusted) data from BLS for both national
      (CES0000000001) and the specified MSAs.
    - Inserts the raw data into a PostgreSQL table named 'raw_nonfarm_jobs'.
"""

import requests
import json
import psycopg2
import pandas as pd

# ------------------------------------------------
# ------------------------------------------------
# 1) BLS API & Series Config
# ------------------------------------------------
BLS_API_KEY = "f232fdbb532d456b8ed8ca7ce2a1cbb2"

# ------------------------------------------------
# 1) BLS API & Series Config
# ------------------------------------------------
BLS_API_KEY = "f232fdbb532d456b8ed8ca7ce2a1cbb2"

# ------------------------------------------------
# 1) BLS API & Series Config
# ------------------------------------------------
BLS_API_KEY = "f232fdbb532d456b8ed8ca7ce2a1cbb2"

# Dictionary mapping each ID (MSA + National) to a friendly name
MSA_SERIES_DICT = {
    "CES0000000001": "National",

    # 1) New York–Newark–Jersey City, NY–NJ–PA
    "SMS36356200000000001": "New York-Newark-Jersey City",

    # 2) Los Angeles–Long Beach–Anaheim, CA
    "SMS06310800000000001": "Los Angeles-Long Beach-Anaheim",

    # 3) Chicago–Naperville–Elgin, IL–IN–WI
    "SMS17169800000000001": "Chicago-Naperville-Elgin",

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
    "SMS25716540000000001": "Boston",

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

    # 43) Hartford–East Hartford–West, CT
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
    "SMS21311400000000001": "Louisville/Jefferson County"
}

# Build a list of all IDs from the dictionary keys
ALL_SERIES_IDS = list(MSA_SERIES_DICT.keys())

START_YEAR = "1990"
END_YEAR   = "2010"


# ------------------------------------------------
# 2) Postgres Connection Info
# ------------------------------------------------
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "inquire_DB"
DB_USER = "postgres"
DB_PASS = "givedata"  # Replace with your actual password

# ------------------------------------------------
# 3) Fetch Data from BLS
# ------------------------------------------------
def fetch_bls_data():
    """
    Calls the BLS Public Data API for the specified series IDs and date range.
    Returns a DataFrame of [series_id, obs_date, value].
    """
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = {
        "seriesid": ALL_SERIES_IDS,
        "startyear": START_YEAR,
        "endyear": END_YEAR,
        "registrationkey": BLS_API_KEY
    }
    headers = {"Content-type": "application/json"}

    # 1) Make request
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data_json = response.json()

    # 2) Get the "series" list from the JSON
    series_list = data_json.get("Results", {}).get("series", [])

    # 3) Build up all_rows from each series
    all_rows = []  # define this BEFORE your loop
    for series_item in series_list:
        sid = series_item["seriesID"]
        for item in series_item.get("data", []):
            period = item["period"]   # e.g. "M01".."M12" or "M13"
            if period.startswith("M") and period != "M13":
                year = int(item["year"])
                month = int(period[1:])
                value = float(item["value"])

                all_rows.append({
                    "series_id": sid,
                    "obs_date": f"{year}-{month:02d}-01",
                    "value": value
                })

    # 4) Convert to DataFrame
    df = pd.DataFrame(all_rows)
    df["obs_date"] = pd.to_datetime(df["obs_date"])

    # Debug info: earliest/latest date, row count, sample
    print("Total row count from BLS API =", len(df))
    print("Sample of last few rows:\n", df.tail(5))
    print("Earliest date in df:", df["obs_date"].min())
    print("Latest date in df:",   df["obs_date"].max())

    return df

# ------------------------------------------------
# 4) Store in Postgres
# ------------------------------------------------
def store_in_postgres(df):
    """
    Inserts rows into 'raw_nonfarm_jobs' table.
    Table columns expected:
        series_id (text)
        obs_date (date)
        value (float)
      with a UNIQUE constraint on (series_id, obs_date).
    """
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    cur = conn.cursor()

    insert_query = """
        INSERT INTO raw_nonfarm_jobs (series_id, obs_date, value)
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING;
    """

    rows_inserted = 0
    for _, rowdata in df.iterrows():
        cur.execute(insert_query, (rowdata["series_id"], rowdata["obs_date"], rowdata["value"]))
        rows_inserted += cur.rowcount

    conn.commit()
    cur.close()
    conn.close()

    return rows_inserted

def main():
    # If you want the number of MSA+national keys total:
    print(f"Fetching data from BLS ({len(ALL_SERIES_IDS)} total series, {START_YEAR}–{END_YEAR})...")
    df = fetch_bls_data()
    print(f"Fetched {len(df)} total rows from BLS.")

    print("Storing into Postgres table 'raw_nonfarm_jobs'...")
    inserted_count = store_in_postgres(df)
    print(f"Inserted {inserted_count} new rows (or fewer if duplicates existed).")

if __name__ == "__main__":
    main()
