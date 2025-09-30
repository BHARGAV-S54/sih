import os
import re
import csv
import typing as T
from urllib.parse import urljoin
from datetime import datetime

import requests
import pandas as pd
import numpy as np
import xarray as xr

# -----------------------
# Configuration
# -----------------------
BASE_URL = "https://data-argo.ifremer.fr"
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
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                if chunk:
                    f.write(chunk)
    return out_path

def to_datetime_juld(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, unit="D", origin="1950-01-01", errors="coerce")

def safe_open_dataset(nc_path: str) -> xr.Dataset:
    # Lazy load with dask chunks to reduce memory usage
    return xr.open_dataset(nc_path, decode_timedelta=True, chunks={})

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: v.decode() if isinstance(v, (bytes, bytearray)) else v)
    return df

# -----------------------
# Index parsing
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
    df["filetype"] = df["filename"].apply(
        lambda f: "traj" if "Rtraj" in f else ("prof" if "prof" in f else
                  ("meta" if "meta" in f else "tech" if "tech" in f else "unknown"))
    )
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
    try:
        meas_vars = [v for v in ds.data_vars if "N_MEASUREMENT" in ds[v].dims]
        cycle_vars = [v for v in ds.data_vars if "N_CYCLE" in ds[v].dims]

        meas_df = ds[meas_vars].to_dataframe().reset_index() if meas_vars else pd.DataFrame()
        cycle_df = ds[cycle_vars].to_dataframe().reset_index() if cycle_vars else pd.DataFrame()

        if not meas_df.empty and not cycle_df.empty and "CYCLE_NUMBER" in meas_df.columns:
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

        rep = None
        for c in ["JULD_ASCENT_END", "JULD_TRANSMISSION_START", "JULD"]:
            if c in juld_cols:
                rep = c
                break
        if rep:
            df["representative_time_dt"] = df[rep + "_dt"]
            df["representative_time_date"] = df[rep + "_date"]

        df["platform_number"] = str(ds.attrs.get("PLATFORM_NUMBER", "")) or None
        return df
    finally:
        ds.close()

def convert_prof(nc_path: str) -> pd.DataFrame:
    ds = safe_open_dataset(nc_path)
    try:
        vars_candidate = [v for v in ["PRES", "TEMP", "PSAL", "DOXY", "JULD",
                                      "LATITUDE", "LONGITUDE", "CYCLE_NUMBER"]
                          if v in ds.variables]
        qc_vars = [qc for qc in ["PRES_QC", "TEMP_QC", "PSAL_QC", "DOXY_QC", "POSITION_QC"]
                   if qc in ds.variables]

        selected = vars_candidate + qc_vars
        df = ds[selected].to_dataframe().reset_index() if selected else ds.to_dataframe().reset_index()

        df = sanitize_columns(df)

        if "JULD" in df.columns:
            df["JULD_dt"] = to_datetime_juld(df["JULD"])
            df["JULD_date"] = pd.to_datetime(df["JULD_dt"]).dt.date

        df["platform_number"] = str(ds.attrs.get("PLATFORM_NUMBER", "")) or None

        if "CYCLE_NUMBER" in df.columns:
            for col in ["LATITUDE", "LONGITUDE"]:
                if col in df.columns:
                    df[col] = df.groupby("CYCLE_NUMBER")[col].transform(lambda s: s.ffill().bfill())

        return df
    finally:
        ds.close()

# -----------------------
# Batch runner
# -----------------------
def process_float(platform_id: str, index_traj: pd.DataFrame, index_prof: pd.DataFrame):
    print(f"=== Processing platform {platform_id} ===")

    traj_files = filter_by_platforms(index_traj, [platform_id])
    prof_files = filter_by_platforms(index_prof, [platform_id])

    out_dir = os.path.join(OUTPUT_DIR, str(platform_id))
    os.makedirs(out_dir, exist_ok=True)

    for _, row in traj_files.iterrows():
        url, fname = row["url"], row["filename"]
        local_nc = os.path.join(DOWNLOAD_DIR, fname)
        print(f"Downloading traj: {url}")
        stream_download(url, local_nc)
        print(f"Converting traj:)
                print(f"Converting traj: {local_nc}")
        try:
            df = convert_traj(local_nc)
            base = os.path.splitext(fname)[0]
            df.to_csv(os.path.join(out_dir, f"{base}_events.csv"), index=False)
            df.to_parquet(os.path.join(out_dir, f"{base}_events.parquet"), index=False)
        except Exception as e:
            print(f"Failed to convert traj {fname}: {e}")

    # For demo mode, only process the last 3 profile files to save memory
    prof_files_sorted = prof_files.sort_values(by="filename")
    for _, row in prof_files_sorted.tail(3).iterrows():
        url, fname = row["url"], row["filename"]
        local_nc = os.path.join(DOWNLOAD_DIR, fname)
        print(f"Downloading prof: {url}")
        stream_download(url, local_nc)
        print(f"Converting prof: {local_nc}")
        try:
            df = convert_prof(local_nc)
            base = os.path.splitext(fname)[0]
            df.to_csv(os.path.join(out_dir, f"{base}_profiles.csv"), index=False)
            df.to_parquet(os.path.join(out_dir, f"{base}_profiles.parquet"), index=False)
        except Exception as e:
            print(f"Failed to convert prof {fname}: {e}")


# -----------------------
# Main entrypoint
# -----------------------
def main(platform_ids: T.List[str]):
    print("Loading indexes...")
    idx_traj = load_index(INDEX_TRAJ)
    idx_prof = load_index(INDEX_PROF)

    if idx_traj.empty:
        print("Trajectory index empty; check URL or network.")
    if idx_prof.empty:
        print("Profile index empty; check URL or network.")

    for pid in platform_ids:
        process_float(str(pid), idx_traj, idx_prof)


if __name__ == "__main__":
    # Example: process float 13858 (valid AOML DAC float)
    main(platform_ids=["13858"])
