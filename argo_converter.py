import os
import re
import io
import csv
import gzip
import time
import json
import math
import queue
import shutil
import typing as T
from datetime import datetime
from urllib.parse import urljoin
import requests
import pandas as pd
import numpy as np
import xarray as xr

# -----------------------
# Configuration
# -----------------------
BASE_URL = BASE_URL = "https://data-argo.ifremer.fr/"
INDEX_TRAJ = urljoin(BASE_URL, "ar_index_global_traj.txt")   # global trajectory index
INDEX_PROF = urljoin(BASE_URL, "ar_index_global_prof.txt")   # global profile index
DOWNLOAD_DIR = r"./argo_downloads"
OUTPUT_DIR = r"./argo_outputs"
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
    except requests.exceptions.HTTPError as e:
        print(f"❌ File not found: {url}")
        return None
def to_datetime_juld(series: pd.Series) -> pd.Series:
    # Argo JULD: days since 1950-01-01
    return pd.to_datetime(series, unit="D", origin="1950-01-01", errors="coerce")

def safe_open_dataset(nc_path: str) -> xr.Dataset:
    return xr.open_dataset(nc_path, decode_timedelta=True)

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure bytes-like QC/status become decoded strings
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: v.decode() if isinstance(v, (bytes, bytearray)) else v)
    return df

# -----------------------
# Index parsing
# -----------------------
def parse_index_lines(text: str) -> pd.DataFrame:
    """
    Argo index files are pipe '|' or comma separated (historically pipe).
    Typical columns include: file, date, latitude, longitude, cycle, platform, etc.
    We'll detect delimiter and map minimally useful fields.
    """
    # Detect delimiter
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if not lines:
        return pd.DataFrame()

    delim = "|" if lines[0].count("|") >= 3 else ","
    reader = csv.reader(lines, delimiter=delim)
    rows = list(reader)

    # Heuristic header presence: first row may be header; if path contains '/dac/', treat rows uniformly.
    # Build DataFrame without assuming headers.
    df = pd.DataFrame(rows)
    # Identify path column as the one containing '/dac/'.
    path_col_idx = None
    for i in range(min(3, df.shape[1])):  # check first few cols
        if df.iloc[0, i].find("/dac/") != -1 or df.iloc[0, i].endswith(".nc"):
            path_col_idx = i
            break
    if path_col_idx is None:
        # fallback: first column
        path_col_idx = 0

    df = df.rename(columns={path_col_idx: "path"})
    # Extract platform_number if present in path like .../dac/<dac>/<platform>/<file>
    def extract_platform(p: str) -> str:
        try:
            parts = p.strip("/").split("/")
            idx = parts.index("dac")
            return parts[idx+2]  # dac/<dac>/<platform>/
        except Exception:
            # fallback: digits in filename
            m = re.search(r"(\d{5,})_", p)
            return m.group(1) if m else ""
    df["platform_number"] = df["path"].apply(extract_platform)

    # Extract filename and type
    df["filename"] = df["path"].apply(lambda p: p.split("/")[-1])
    df["filetype"] = df["filename"].apply(lambda f: "traj" if "Rtraj" in f else ("prof" if "prof" in f else ("meta" if "meta" in f else "tech" if "tech" in f else "unknown")))
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

def filter_filetypes(df: pd.DataFrame, types: T.List[str]) -> pd.DataFrame:
    types = set(types)
    return df[df["filetype"].isin(types)].copy()

# -----------------------
# Converters
# -----------------------
def convert_traj(nc_path: str) -> pd.DataFrame:
    ds = safe_open_dataset(nc_path)
    # Split by dims
    meas_vars = [v for v in ds.data_vars if "N_MEASUREMENT" in ds[v].dims]
    cycle_vars = [v for v in ds.data_vars if "N_CYCLE" in ds[v].dims]
    scalar_vars = [v for v in ds.data_vars if ds[v].dims == ()]

    meas_df = ds[meas_vars].to_dataframe().reset_index() if meas_vars else pd.DataFrame()
    cycle_df = ds[cycle_vars].to_dataframe().reset_index() if cycle_vars else pd.DataFrame()
    scalar_df = pd.DataFrame({v: [ds[v].values.item()] for v in scalar_vars}) if scalar_vars else pd.DataFrame()

    # Merge measurement with cycle on CYCLE_NUMBER where possible
    if not meas_df.empty and not cycle_df.empty and "CYCLE_NUMBER" in meas_df.columns and "CYCLE_NUMBER" in cycle_df.columns:
        df = meas_df.merge(cycle_df, on="CYCLE_NUMBER", how="left")
    elif not cycle_df.empty:
        df = cycle_df
    else:
        df = meas_df

    df = sanitize_columns(df)

    # Convert all JULD* to *_dt and *_date
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

    # Propagate LAT/LON within cycle
    if "CYCLE_NUMBER" in df.columns:
        for col in ["LATITUDE", "LONGITUDE"]:
            if col in df.columns:
                df[col] = df.groupby("CYCLE_NUMBER")[col].transform(lambda s: s.ffill().bfill())

    # Representative time per row: choose best available event
    cand = [c for c in juld_cols if c.endswith("ASCENT_END") or c.endswith("TRANSMISSION_START") or c == "JULD"]
    rep = None
    for c in ["JULD_ASCENT_END", "JULD_TRANSMISSION_START", "JULD"]:
        if c in juld_cols:
            rep = c
            break
    if rep:
        df["representative_time_dt"] = df[rep + "_dt"]
        df["representative_time_date"] = df[rep + "_date"]

    # Attach platform_number if present as scalar or attribute
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

def convert_prof(nc_path: str) -> pd.DataFrame:
    ds = safe_open_dataset(nc_path)

    # Common measurement dims: N_LEVEL, N_PROF (varies by file)
    # We'll attempt to to_dataframe() for primary variables present.
    vars_candidate = []
    for name in ["PRES", "TEMP", "PSAL", "DOXY", "JULD", "LATITUDE", "LONGITUDE", "CYCLE_NUMBER"]:
        if name in ds.variables:
            vars_candidate.append(name)
    # Include QC if present
    for qc in ["PRES_QC", "TEMP_QC", "PSAL_QC", "DOXY_QC", "POSITION_QC"]:
        if qc in ds.variables:
            vars_candidate.append(qc)

    if not vars_candidate:
        df = ds.to_dataframe().reset_index()
    else:
        df = ds[vars_candidate].to_dataframe().reset_index()

    df = sanitize_columns(df)

    # Convert JULD to datetime
    if "JULD" in df.columns:
        df["JULD_dt"] = to_datetime_juld(df["JULD"])
        df["JULD_date"] = pd.to_datetime(df["JULD_dt"]).dt.date

    # Attach platform_number
    platform = None
    if "PLATFORM_NUMBER" in ds.variables:
        try:
            platform = str(ds["PLATFORM_NUMBER"].values.item())
        except Exception:
            pass
    df["platform_number"] = platform

    # Propagate LAT/LON per cycle if available
    if "CYCLE_NUMBER" in df.columns:
        for col in ["LATITUDE", "LONGITUDE"]:
            if col in df.columns:
                df[col] = df.groupby("CYCLE_NUMBER")[col].transform(lambda s: s.ffill().bfill())

    return df

# -----------------------
# Batch runner
# -----------------------
def process_float(platform_id: str, index_traj: pd.DataFrame, index_prof: pd.DataFrame):
    print(f"=== Processing platform {platform_id} ===")

    # Select files for this platform
    traj_files = filter_by_platforms(index_traj, [platform_id])
    prof_files = filter_by_platforms(index_prof, [platform_id])

    # Prepare output dirs
    out_dir = os.path.join(OUTPUT_DIR, str(platform_id))
    os.makedirs(out_dir, exist_ok=True)

    # Download and convert trajectory files
    for _, row in traj_files.iterrows():
        url = row["url"]
        fname = row["filename"]
        local_nc = os.path.join(DOWNLOAD_DIR, fname)
        print(f"Downloading traj: {url}")
        try:
            df = convert_traj(local_nc)
            base = os.path.splitext(fname)[0]
            df.to_csv(os.path.join(out_dir, f\"{base}_events.csv\") index=False)
            df.to_parquet(os.path.join(out_dir, f"{base}_events.parquet"), index=False)
    except Exception as e:
            print(f"Failed convert traj {fname}: {e}")
 # Download and convert profile files (you can limit to latest N for demo)
    # For demo, keep last 3 profile files
      print(f"⏭️ Skipping prof file: {fname}")
     for _, row in prof_files_sorted.tail(3).iterrows():
        url = row["url"]
  fname = row["filename"]
        local_nc = os.path.join(DOWNLOAD_DIR, fname)
        print(f"Downloading prof: {url}")
        stream_download(url, local_nc)
        print(f"Converting prof: {local_nc}")
 try
           df = convert_prof(local_nc)
          base = os.path.splitext(fname)[0]
          df.to_csv(os.path.join(out_dir, f"{base}_profiles.csv"),index=False)
          df.to_parquet(os.path.join(out_dir, f"{base}_profiles.parquet"),index=False)
  except Exception as e:
            print(f"Failed convert prof {fname}: {e}")
def main(platform_ids: T.List[str]):
    print("Loading indexes...")
    idx_traj = load_index(INDEX_TRAJ)
   <br>local_nc = stream_download(url, local_nc)<br>
if not local_nc:<br>
    print(f"⏭️ Skipping prof file: {fname}")<br>
continue
 idx_prof = load_index(INDEX_PROF)
  # Basic sanity
    if idx_traj.empty:
        print("Trajectory index empty; check URL or network.")
    if idx_prof.empty:
        print("Profile index empty; check URL or network.")

    for pid in platform_ids:
        process_float(str(pid), idx_traj, idx_prof)

if __name__ == "__main__":
    # Example: process one float (13857). Add more IDs as needed.
    main(platform_ids=["4901210"])

