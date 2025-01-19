# nonfarm data to github.py
import psycopg2
import pandas as pd
import subprocess
import os
import datetime

# Adjust these to match your local DB
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "inquire_DB"
DB_USER = "postgres"       # changed back to "postgres"
DB_PASS = "givedata"

# Path to your local GitHub repo folder
LOCAL_REPO_PATH = r"C:\Users\student\Desktop\InquiRE Python" 
# Make sure this matches where your local repo actually resides

def export_csv_from_sql():
    """
    1) Connect to your local Postgres
    2) Fetch alpha_beta_results
    3) Save as CSV in the repo's data/ folder
    4) Git add/commit/push that CSV
    """
    # 1) Connect to Postgres
    conn = psycopg2.connect(
        host="localhost",
        port="5433",
        dbname="inquire_DB",
        user="postgres",
        password="givedata"
    )

    # 2) Query alpha_beta_results (use * or specific columns)
    query = """
        SELECT * FROM alpha_beta_results
        ORDER BY series_id
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # 3) Save as CSV in the "data" folder
    csv_path = os.path.join(LOCAL_REPO_PATH, "data", "nonfarm_data.csv")
    df.to_csv(csv_path, index=False)

    # 4) Commit & push via Git
    os.chdir(LOCAL_REPO_PATH)
    subprocess.run(["git", "add", "data/nonfarm_data.csv"])
    commit_msg = f"Auto-export alpha_beta_results {datetime.datetime.now()}"
    subprocess.run(["git", "commit", "-m", commit_msg])
    subprocess.run(["git", "push", "origin", "main"])
    print("Exported alpha_beta_results CSV and pushed to GitHub.")

if __name__ == "__main__":
    export_csv_from_sql()
