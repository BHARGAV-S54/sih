import os
import re
import csv
import requests
import pandas as pd
import numpy as np
import xarray as xr

# -----------------------
# Configuration
# -----------------------
BASE_URL = "https://data-argo.ifremer.fr"
INDEX_TRAJ = BASE_URL + "/ar_index_global_traj.txt"
INDEX_PROF = BASE_URL + "/ar_index_global_prof.txt"
DOWNLOAD_DIR = "./argo_downloads"
OUTPUT_DIR = "./argo_outputs"
TIMEOUT = 60

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def to_datetime_juld(series: pd.Series) -> pd.Series:
    """Convert Argo JULD (days since 1950-01-01) to datetime"""
    return pd.to_datetime(series, unit="D", origin="1950-01-01", errors="coerce")

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Decode byte strings to str"""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: v.decode() if isinstance(v, (bytes, bytearray)) else v)
    return df

def safe_open_dataset(nc_path: str) -> xr.Dataset:
    return xr.open_dataset(nc_path, decode_timedelta=True)

def stream_index(url: str, platform_id: str):
    """Stream index file line by line and yield only lines for this platform"""
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line and str(platform_id) in line:
                yield line

def parse_index_lines(lines) -> pd.DataFrame:
    """Parse filtered index lines into DataFrame"""
    reader = csv.reader(lines, delimiter="|")
    rows = list(reader)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Path is usually in first column
    df = df.rename(columns={0: "path"})
    df["filename"] = df["path"].apply(lambda p: p.split("/")[-1])
    df["url"] = df["path"].apply(lambda p: BASE_URL + "/" + p.lstrip("/"))
    return df

# -----------------------
# Converters
# -----------------------
def convert_traj(nc_path: str) -> pd.DataFrame:
    ds = safe_open_dataset(nc_path)
    df = ds.to_dataframe().reset_index()
    df = sanitize_columns(df)

    # Convert JULD* vars
    juld_cols = [c for c in df.columns if c.startswith("JULD")]
    for c in juld_cols:
        df[c + "_dt"] = to_datetime_juld(df[c])
        df[c + "_date"] = pd.to_datetime(df[c + "_dt"]).dt.date

    # Representative time
    for c in ["JULD_ASCENT_END", "JULD_TRANSMISSION_START", "JULD"]:
        if c in juld_cols:
            df["representative_time_dt"] = df[c + "_dt"]
            break

    return df

def convert_prof(nc_path: str) -> pd.DataFrame:
    ds = safe_open_dataset(nc_path)
    vars_candidate = [v for v in ["PRES","TEMP","PSAL","DOXY","JULD","LATITUDE","LONGITUDE","CYCLE_NUMBER"] if v in ds.variables]
    df = ds[vars_candidate].to_dataframe().reset_index()
    df = sanitize_columns(df)
    if "JULD" in df.columns:
        df["JULD_dt"] = to_datetime_juld(df["JULD"])
        df["JULD_date"] = pd.to_datetime(df["JULD_dt"]).dt.date
    return df

# -----------------------
# Batch runner
# -----------------------
def process_float(platform_id: str):
    print(f"=== Processing platform {platform_id} ===")

    # Stream and filter indexes
    traj_lines = list(stream_index(INDEX_TRAJ, platform_id))
    prof_lines = list(stream_index(INDEX_PROF, platform_id))

    idx_traj = parse_index_lines(traj_lines)
    idx_prof = parse_index_lines(prof_lines)

    out_dir = os.path.join(OUTPUT_DIR, str(platform_id))
    os.makedirs(out_dir, exist_ok=True)

    # Download + convert traj
    for _, row in idx_traj.iterrows():
        url = row["url"]
        fname = row["filename"]
        local_nc = os.path.join(DOWNLOAD_DIR, fname)
        print(f"Downloading traj: {url}")
        r = requests.get(url, timeout=TIMEOUT)
        with open(local_nc, "wb") as f:
            f.write(r.content)
        print(f"Converting traj: {local_nc}")
        df = convert_traj(local_nc)
        df.to_csv(os.path.join(out_dir, fname.replace(".nc","_events.csv")), index=False)

    # Download + convert last 2 prof files
    for _, row in idx_prof.tail(2).iterrows():
        url = row["url"]
        fname = row["filename"]
        local_nc = os.path.join(DOWNLOAD_DIR, fname)
        print(f"Downloading prof: {url}")
        r = requests.get(url, timeout=TIMEOUT)
        with open(local_nc, "wb") as f:
            f.write(r.content)
        print(f"Converting prof: {local_nc}")
        df = convert_prof(local_nc)
        df.to_csv(os.path.join(out_dir, fname.replace(".nc","_profiles.csv")), index=False)

def main():
    # Example: process float 13857
    process_float("13857")

if __name__ == "__main__":
    main()
