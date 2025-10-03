import os
import requests
import pandas as pd
import xarray as xr
from flask import Flask, send_file, render_template_string
from urllib.parse import urljoin

app = Flask(__name__)

BASE_URL = "https://data-argo.ifremer.fr/dac"
DOWNLOAD_DIR = "./argo_downloads"
OUTPUT_DIR = "./converted_data"
TIMEOUT = 60
CHUNK = 1 << 16  # 64KB

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def safe_open_dataset(nc_path: str) -> xr.Dataset:
    return xr.open_dataset(nc_path, decode_timedelta=True)

def convert_nc_to_csv(nc_path: str, output_csv_path: str):
    ds = safe_open_dataset(nc_path)
    df = ds.to_dataframe().reset_index()
    df.to_csv(output_csv_path, index=False)
    return output_csv_path

@app.route("/")
def index():
    # Example ARGO file (platform 13857)
    nc_url = "https://data-argo.ifremer.fr/dac/aoml/13857/13857_Rtraj.nc"
    nc_fname = os.path.join(DOWNLOAD_DIR, "13857_Rtraj.nc")
    csv_fname = os.path.join(OUTPUT_DIR, "13857.csv")
    if not os.path.exists(csv_fname):
        stream_download(nc_url, nc_fname)
        convert_nc_to_csv(nc_fname, csv_fname)
    html = """
    <h1>ARGO CSV Download</h1>
    <a href="/download" download>Download CSV</a>
    """
    return render_template_string(html)

@app.route("/download")
def download():
    csv_fname = os.path.join(OUTPUT_DIR, "13857.csv")
    return send_file(csv_fname, as_attachment=True, download_name="argo_13857.csv")

if __name__ == "__main__":
    app.run(debug=True)
