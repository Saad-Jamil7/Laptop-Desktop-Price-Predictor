# Laptop & Desktop Price Predictor

A machine learning project that predicts the price of a laptop or desktop from its specifications. It is trained on a dataset of **100,000 computers** with 33 raw columns covering CPU, GPU, RAM, storage, display, battery and more.

## Why this version exists (the bug that was fixed)

The first version ignored `device_type`: switching a machine from **Desktop** to **Laptop** did not change the predicted price at all.

The cause was that `device_type` is fully redundant in this data. Other columns already identify it:

| | Laptop | Desktop |
|---|---|---|
| `battery_wh` | 50–99 | 0 |
| `psu_watts` | 0 | 300–1200 |
| `form_factor` | Mainstream, Gaming, ... | ATX, SFF, ... |

A single model never needed `device_type`, so flipping it had zero effect.

**The fix:** train **one model per device type** and route each prediction by `device_type`.

Other fixes in this version:
- **Hybrid drives** were counted as 0 GB of storage. They are now grouped with SSD/NVMe, since they price like SSDs.
- The model was trained on `cpu_family` strings the app never sends. It now uses `cpu_brand` plus the numeric `cpu_tier`.
- The final cell saved the *last* model in the loop, not the best one. It now saves the best model per device type.

## Project structure

```
.
├── laptop_price_predictor_fixed.ipynb   # EDA, feature engineering, training, sanity checks
├── price_model.py                       # FEATURES per device type + DevicePriceModel wrapper
├── computer_prices_all.csv              # dataset
├── model.pkl                            # saved DevicePriceModel
└── df.pkl                               # processed dataframe
```

## Approach

1. **Exploration:** price distribution, price by brand, CPU family, RAM and storage, plus a correlation heatmap.
2. **Feature engineering:**
   - Split `resolution` into width and height and compute **PPI** (pixels per inch).
   - Split storage into `ssd_gb` and `hdd_gb`.
3. **Target transform:** train on `log(price)` to handle the skewed price distribution, then convert back with `exp()`.
4. **Modeling:** for each device type (Laptop, Desktop), compare three models on an 80/20 split:
   - Ridge Regression
   - Random Forest
   - HistGradientBoosting
5. **Selection:** pick the model with the lowest MAE in dollars, then refit it on all rows of that device type.
6. **Sanity checks:** confirm that device type now changes the prediction and that price rises with RAM, CPU tier and GPU tier.

## Results

Metrics are printed by the training cell, in dollars after converting back from log-price.

| Device | Best model | MAE | R² |
|---|---|---|---|
| Laptop | _fill in_ | _fill in_ | _fill in_ |
| Desktop | _fill in_ | _fill in_ | _fill in_ |

## How to run

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
jupyter notebook laptop_price_predictor_fixed.ipynb
```

Keep `price_model.py` and `computer_prices_all.csv` in the same folder as the notebook. Running all cells trains both models and writes `model.pkl` and `df.pkl`.

## Using the saved model

```python
import pickle
model = pickle.load(open("model.pkl", "rb"))

# `row` is a one-row DataFrame with the columns listed in price_model.FEATURES
price = model.predict_price(row)[0]
```

`predict()` returns log-price, as in the first version. `predict_price()` returns dollars.

## Tech stack

Python, pandas, NumPy, scikit-learn, seaborn, matplotlib

## Notes

- The dataset appears synthetic, so scores here may not carry over to real market prices.
- Possible next steps: cross-validation, hyperparameter tuning, and feature importance analysis per device type.
