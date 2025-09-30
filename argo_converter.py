import os
import re
import csv
import requests
import pandas as pd
import numpy as np
import xarray as xr
import typing as T
from urllib.parse import urljoin
from datetime import datetime

# -----------------------
# Configuration
# -----------------------
BASE_URL = "https://data-argo.ifremer.fr/"
INDEX_TRAJ = urljoin(BASE_URL, "ar_index_global_traj.txt")
INDEX_PROF = urljoin(BASE_URL, "ar_index_global_prof.txt")
DOWNLOAD_DIR = "./argo_downloads"
OUTPUT_DIR = "./argo_outputs"
TIMEOUT = 60
CHUNK = 1 << 16  # 64KB

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def fetch_text(url: str) -> str:
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text

def stream_download(url: str, out_path: str) -> str:
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(CHUNK):
                    if chunk:
                        f.write(chunk)
        return out_path
    except requests.exceptions.HTTPError:
        print(f"❌ File not found: {url}")
        return None

def to_datetime_juld(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, unit="D", origin="1950-01-01", errors="coerce")

def safe_open_dataset(nc_path: str) -> xr.Dataset:
    return xr.open_dataset(nc_path, decode_timedelta=True)

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: v.decode() if isinstance(v, (bytes, bytearray)) else v)
    return df

# -----------------------
# Index Parsing
# -----------------------
def parse_index_lines(text: str) -> pd.DataFrame:
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if not lines:
        return pd.DataFrame()

    delim = "|" if lines[0].count("|") >= 3 else ","
    reader = csv.reader(lines, delimiter=delim)
    rows = list(reader)
    df = pd.DataFrame(rows)

    path_col_idx = None
    for i in range(min(3, df.shape[1])):
        if "/dac/" in df.iloc[0, i] or df.iloc[0, i].endswith(".nc"):
            path_col_idx = i
            break
    if path_col_idx is None:
        path_col_idx = 0

    df = df.rename(columns={path_col_idx: "path"})

    def extract_platform(p: str) -> str:
        try:
            parts = p.strip("/").split("/")
            idx = parts.index("dac")
            return parts[idx + 2]
        except Exception:
            m = re.search(r"(\d{5,})_", p)
            return m.group(1) if m else ""

    df["platform_number"] = df["path"].apply(extract_platform)
    df["filename"] = df["path"].apply(lambda p: p.split("/")[-1])
    df["filetype"] = df["filename"].apply(lambda f: "traj" if "Rtraj" in f else "prof" if "prof" in f else "unknown")
    df["url"] = df["path"].apply(lambda p: urljoin(BASE_URL + "/", p.lstrip("/")))
    return df

def load_index(url: str) -> pd.DataFrame:
    text = fetch_text(url)
    return parse_index_lines(text)

# -----------------------
# Filters
# -----------------------
def filter_by_platforms(df: pd.DataFrame, platforms: T.List[str]) -> pd.DataFrame:
    platforms = set(str(p) for p in platforms)
    return df[df["platform_number"].astype(str).isin(platforms)].copy()

# -----------------------
# Converters
# -----------------------
def convert_traj(nc_path: str) -> pd.DataFrame:
    ds = safe_open_dataset(nc_path)
    meas_vars = [v for v in ds.data_vars if "N_MEASUREMENT" in ds[v].dims]
    cycle_vars = [v for v in ds.data_vars if "N_CYCLE" in ds[v].dims]
    scalar_vars = [v for v in ds.data_vars if ds[v].dims == ()]

    meas_df = ds[meas_vars].to_dataframe().reset_index() if meas_vars else pd.DataFrame()
    cycle_df = ds[cycle_vars].to_dataframe().reset_index() if cycle_vars else pd.DataFrame()
    scalar_df = pd.DataFrame({v: [ds[v].values.item()] for v in scalar_vars}) if scalar_vars else pd.DataFrame()

    if not meas_df.empty and not cycle_df.empty and "CYCLE_NUMBER" in meas_df.columns and "CYCLE_NUMBER" in cycle_df.columns:
        df = meas_df.merge(cycle_df, on="CYCLE_NUMBER", how="left")
    elif not cycle_df.empty:
        df = cycle_df
    else:
        df = meas_df

    df = sanitize_columns(df)

    juld_cols = [c for c in df.columns if c.startswith("JULD")]
    for c in juld_cols:
        try:
            if np.issubdtype(df[c].dtype, np.datetime64):
                df[c + "_dt"] = df[c]
            else:
                df[c + "_dt"] = to_datetime_juld(df[c])
            df[c + "_date"] = pd.to_datetime(df[c + "_dt"]).dt.date
        except Exception:
            df[c + "_dt"] = pd.NaT
            df[c + "_date"] = pd.NaT

    if "CYCLE_NUMBER" in df.columns:
        for col in ["LATITUDE", "LONGITUDE"]:
            if col in df.columns:
                df[col] = df.groupby("CYCLE_NUMBER")[col].transform(lambda s: s.ffill().bfill())

    rep = next((c for c in ["JULD_ASCENT_END", "JULD_TRANSMISSION_START", "JULD"] if c in juld_cols), None)
    if rep:
        df["representative_time_dt"] = df[rep + "_dt"]
        df["representative_time_date"] = df[rep + "_date"]

    platform = None
    for key in ["PLATFORM_NUMBER", "platform_number"]:
        if key in df.columns:
            platform = str(df[key].iloc[0]) if len(df[key]) else None
            break
    if not platform and "PLATFORM_NUMBER" in ds.variables:
        try:
            platform = str(ds["PLATFORM_NUMBER"].values.item())
        except Exception:
            platform = None
    df["platform_number"] = platform

    return df

# -----------------------
# Batch Runner
# -----------------------
def process_float(platform_id: str, index_traj: pd.DataFrame):
    print(f"=== Processing platform {platform_id} ===")
    traj_files = filter_by_platforms(index_traj, [platform_id])
    out_dir = os.path.join(OUTPUT_DIR, str(platform_id))
    os.makedirs(out_dir, exist_ok=True)

    for _, row in traj_files.iterrows():
        url = row["url"]
        fname = row["filename"]
        local_nc = os.path.join(DOWNLOAD_DIR, fname)
        print(f"Downloading traj: {url}")
        local_nc = stream_download(url, local_nc)
        if not local_nc:
            print(f"⏭️ Skipping traj file: {fname}")
            continue
        try:
            base = os.path.splitext(fname)[0]
            df = convert_traj(local_nc)
            df.to_csv(os.path.join(out_dir, f"{base}_events.csv"), index=False)
            df.to_parquet(os.path.join(out_dir, f"{base}_events.parquet"), index=False)
        except Exception as e:
            print(f"Failed convert traj {fname}: {e}")

def main(platform_ids: T.List[str]):
    print("Loading indexes...")
    idx_traj = load_index(INDEX_TRAJ)
    if idx_traj.empty:
        print("Trajectory index empty; check URL or network.")
        return
    for pid in platform_ids:
        process_float(str(pid), idx_traj)

if __name__ == "__main__":
     main(platform_ids=["6901924"])

