import pickle
import numpy as np
import pandas as pd
import streamlit as st

import price_model  # noqa: F401  (needed so model.pkl can be unpickled)
from price_model import VALID_FORM_FACTORS, RANGES, DEFAULTS

st.set_page_config(page_title="Computer Price Predictor", page_icon="💻", layout="wide")


@st.cache_resource
def load():
    model = pickle.load(open("model.pkl", "rb"))
    df = pickle.load(open("df.pkl", "rb"))
    return model, df


model, df = load()

TIER_HELP = {1: "Core i3 / Ryzen 3", 2: "Core i5 / Ryzen 5 (mid)", 3: "Core i5 / Ryzen 5 (high)",
             4: "Core i7 / Ryzen 7 (mid)", 5: "Core i7 / Ryzen 7 (high)", 6: "Core i9 / Ryzen 9"}

st.title("💻 Computer Price Predictor")

# Device type is OUTSIDE any form so the page updates immediately when it changes.
device_type = st.radio("Device type", ["Laptop", "Desktop"], horizontal=True)
d = df[df["device_type"] == device_type]
is_laptop = device_type == "Laptop"

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("General")
    brand = st.selectbox("Brand", sorted(df["brand"].unique()))
    release_year = st.selectbox("Release year", sorted(df["release_year"].unique(), reverse=True), index=2)
    os_ = st.selectbox("OS", sorted(df["os"].unique()))
    form_factor = st.selectbox("Form factor", VALID_FORM_FACTORS[device_type])
    warranty = st.selectbox("Warranty (months)", sorted(df["warranty_months"].unique()), index=1)
    weight = st.number_input("Weight (kg)", 0.5, 20.0, 1.7 if is_laptop else 8.0, 0.1)

with c2:
    st.subheader("Processor & graphics")
    cpu_brand = st.selectbox("CPU family", ["Intel", "AMD", "Apple"])
    cpu_tier = st.selectbox("CPU tier", [1, 2, 3, 4, 5, 6], format_func=lambda t: f"{t}  ({TIER_HELP[t]})")
    cpu_cores = st.selectbox("CPU cores", sorted(df["cpu_cores"].unique()), index=1)
    cpu_ghz = st.number_input("CPU base speed (GHz)", 1.0, 5.0, 2.5, 0.1)
    gpu_brand = st.selectbox("GPU brand", sorted(df["gpu_brand"].unique()))
    gpu_tier = st.selectbox("GPU tier (1 = entry, 6 = top)", [1, 2, 3, 4, 5, 6])
    vram = st.selectbox("VRAM (GB)", sorted(df["vram_gb"].unique()))

with c3:
    st.subheader("Memory, storage & display")
    ram = st.selectbox("RAM (GB)", sorted(df["ram_gb"].unique()), index=1)
    drives = st.selectbox("Storage drive count", sorted(df["storage_drive_count"].unique()))
    ssd = st.selectbox("SSD / NVMe / Hybrid (GB)", [0, 128, 256, 512, 1024, 2048, 4096], index=3)
    hdd = st.selectbox("HDD (GB)", [0, 500, 512, 1024, 2048, 4096])
    display_type = st.selectbox("Display type", sorted(d["display_type"].unique()))
    resolution = st.selectbox("Resolution", sorted(d["resolution"].unique(),
                              key=lambda r: int(r.split("x")[0]) * int(r.split("x")[1])))
    size = st.selectbox("Screen size (inches)", sorted(d["display_size_in"].unique()))
    refresh = st.selectbox("Refresh rate (Hz)", sorted(df["refresh_hz"].unique()))
    if is_laptop:
        battery = st.slider("Battery (Wh)", *RANGES["battery_wh"], DEFAULTS["battery_wh"])
        psu = 0
    else:
        psu = st.slider("PSU (watts)", *RANGES["psu_watts"], DEFAULTS["psu_watts"], step=50)
        battery = 0

if st.button("Predict price", type="primary"):
    w, h = map(int, resolution.split("x"))
    row = pd.DataFrame([dict(
        device_type=device_type, brand=brand, release_year=release_year, os=os_, form_factor=form_factor,
        cpu_brand=cpu_brand, cpu_tier=cpu_tier, cpu_cores=cpu_cores, cpu_base_ghz=cpu_ghz,
        gpu_brand=gpu_brand, gpu_tier=gpu_tier, vram_gb=vram, ram_gb=ram, storage_drive_count=drives,
        ssd_gb=ssd, hdd_gb=hdd, display_type=display_type, refresh_hz=refresh,
        ppi=np.hypot(w, h) / size, weight_kg=weight, warranty_months=warranty,
        battery_wh=battery, psu_watts=psu)])
    price = float(np.exp(model.predict(row))[0])
    st.success(f"Estimated price: ${price:,.0f}")
