import os
import requests
import xarray as xr
import pandas as pd
from urllib.parse import urljoin

# Configuration
BASE_URL = "https://data-argo.ifremer.fr/dac"
PLATFORM_ID = "6902746"  # Replace with another valid ID if needed
FILENAME = f"{PLATFORM_ID}_Rtraj.nc"
URL = urljoin(f"{BASE_URL}/coriolis/{PLATFORM_ID}/", FILENAME)
DOWNLOAD_DIR = "./argo_downloads"
OUTPUT_DIR = "./converted_data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download NetCDF file
def download_nc(url, out_path):
    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Downloaded: {out_path}")
            return out_path
        else:
            print(f"❌ File not found (404): {url}")
            return None
    except Exception as e:
        print(f"❌ Download error: {e}")
        return None

# Convert NetCDF to CSV
def convert_to_csv(nc_path, csv_path):
    try:
        ds = xr.open_dataset(nc_path)
        df = ds.to_dataframe().reset_index()
        df = df.dropna(subset=["LATITUDE", "LONGITUDE", "JULD"])
        df["JULD"] = pd.to_datetime(df["JULD"].astype(str), errors="coerce")
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV saved: {csv_path}")
    except Exception as e:
        print(f"❌ Conversion error: {e}")

# Main flow
def main():
    nc_path = os.path.join(DOWNLOAD_DIR, FILENAME)
    csv_path = os.path.join(OUTPUT_DIR, f"{PLATFORM_ID}_traj.csv")
    downloaded = download_nc(URL, nc_path)
    if downloaded:
        convert_to_csv(nc_path, csv_path)

if __name__ == "__main__":
    main()
