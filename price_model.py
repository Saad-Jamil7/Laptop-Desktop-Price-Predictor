
import warnings
import numpy as np
import pandas as pd

# ---- columns each device-type model is trained on -------------------------
COMMON = ["brand", "release_year", "os", "form_factor", "cpu_brand", "cpu_tier",
          "cpu_cores", "cpu_base_ghz", "gpu_brand", "gpu_tier", "vram_gb",
          "ram_gb", "storage_drive_count", "ssd_gb", "hdd_gb", "display_type",
          "refresh_hz", "ppi", "weight_kg", "warranty_months"]
DEVICE_ONLY = {"Laptop": ["battery_wh"], "Desktop": ["psu_watts"]}
FEATURES = {dt: COMMON + extra for dt, extra in DEVICE_ONLY.items()}

VALID_FORM_FACTORS = {
    "Laptop":  ["Mainstream", "Gaming", "Ultrabook", "2-in-1", "Workstation"],
    "Desktop": ["ATX", "Micro-ATX", "Full-Tower", "SFF", "Mini-ITX"],
}
DEFAULT_FORM_FACTOR = {"Laptop": "Mainstream", "Desktop": "ATX"}
# realistic ranges of the device-only specs (taken from the training data)
RANGES = {"battery_wh": (50, 99), "psu_watts": (300, 1200)}
DEFAULTS = {"battery_wh": 70, "psu_watts": 650}

# "Core i3" / "Ryzen 5" ... -> the dataset's numeric cpu_tier (1-6)
CPU_LABEL_TO_TIER = {"core i3": 1, "ryzen 3": 1, "core i5": 2, "ryzen 5": 2,
                     "core i7": 4, "ryzen 7": 4, "core i9": 6, "ryzen 9": 6}


def cpu_brand_from_any(value):
    """'Intel Core i5', 'AMD Ryzen 7', 'Intel', 'Apple M2' -> Intel/AMD/Apple."""
    v = str(value).lower()
    for b in ("intel", "amd", "apple"):
        if b in v:
            return b.upper() if b == "amd" else b.capitalize()
    return "Intel"


def cpu_tier_from_label(label):
    """Accepts 1-6 or labels like 'Core i3'. Returns int tier."""
    try:
        return int(label)
    except (TypeError, ValueError):
        return CPU_LABEL_TO_TIER[str(label).strip().lower()]


class DevicePriceModel:
    """Holds {'Laptop': pipeline, 'Desktop': pipeline} and routes by device_type."""

    def __init__(self, models, names=None):
        self.models = models
        self.names = names or {}

    # -- make an app/form row look like the training rows --------------------
    def _prepare(self, X):
        X = X.copy()
        if "cpu_brand" not in X.columns:
            src = "cpu_family" if "cpu_family" in X.columns else None
            X["cpu_brand"] = X[src].map(cpu_brand_from_any) if src else "Intel"
        X["cpu_tier"] = X["cpu_tier"].map(cpu_tier_from_label)

        for i, dt in X["device_type"].items():
            if dt not in self.models:
                raise ValueError(f"device_type must be one of {list(self.models)}, got {dt!r}")
            if X.at[i, "form_factor"] not in VALID_FORM_FACTORS[dt]:
                warnings.warn(f"form_factor {X.at[i, 'form_factor']!r} is not valid for a {dt}; "
                              f"using {DEFAULT_FORM_FACTOR[dt]!r}.")
                X.at[i, "form_factor"] = DEFAULT_FORM_FACTOR[dt]
            col = DEVICE_ONLY[dt][0]
            val = X.at[i, col] if col in X.columns else np.nan
            lo, hi = RANGES[col]
            
            if pd.isna(val) or not (lo <= val <= hi):      # e.g. battery 0 on a laptop
                X.at[i, col] = DEFAULTS[col] if (pd.isna(val) or val == 0) else min(max(val, lo), hi)
        return X

    def predict(self, X):
        """Returns LOG(price) so existing `np.exp(model.predict(x))` code keeps working."""
        X = self._prepare(X)
        out = np.zeros(len(X))
        for dt, mdl in self.models.items():
            mask = (X["device_type"] == dt).values
            if mask.any():
                out[mask] = mdl.predict(X.loc[mask, FEATURES[dt]])
        return out

    def predict_price(self, X):
        return np.exp(self.predict(X))
