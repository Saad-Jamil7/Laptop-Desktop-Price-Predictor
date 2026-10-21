# Computer & Laptop Price Predictor 💻

An end-to-end machine learning project that estimates computer and laptop prices based on hardware specifications through an interactive Streamlit application.

## 📌 Project Overview

This project explores how hardware configurations influence computer prices and uses regression models to predict estimated prices in USD.

The application supports both laptops and desktop computers and uses separate trained models for each device type.

## 🎯 Objectives

- Build a machine learning solution for computer price prediction.
- Explore and preprocess hardware specification data.
- Compare multiple regression algorithms.
- Evaluate model performance using Mean Absolute Error (MAE) and R².
- Develop an interactive web application using Streamlit.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib / Seaborn 

## 📊 Dataset & Features

The project uses a computer hardware pricing dataset containing specifications such as:

- **Device information:** device type, brand, release year, operating system, and form factor
- **CPU:** CPU brand, tier, core count, and base clock speed
- **GPU:** GPU brand, tier, and VRAM
- **Memory and storage:** RAM, SSD capacity, HDD capacity, and drive count
- **Display:** display type, resolution, screen size, refresh rate, and pixel density
- **Other specifications:** battery capacity, power supply wattage, weight, and warranty

The target variable is `price`, and predictions are expressed in USD.

## ⚙️ Machine Learning Workflow

1. **Data preparation:** Prepare hardware features and separate device types.
2. **Feature engineering:** Derive SSD and HDD capacity features from storage information and prepare the model inputs.
3. **Preprocessing:** Apply one-hot encoding to categorical features while passing numerical features through.
4. **Log transformation:** Train models on the logarithm of price and convert predictions back to the original price scale.
5. **Model comparison:** Evaluate Linear Regression, Random Forest, HistGradientBoosting, Gradient Boosting, Extra Trees, and XGBoost.
6. **Model selection:** Select the candidate with the lowest MAE for each device type.
7. **Final training:** Retrain the selected pipeline on all available rows for that device type.
8. **Deployment interface:** Build an interactive Streamlit application for price estimation.

## 📈 Evaluation

The models are compared using:

- **Mean Absolute Error (MAE):** Measures the average absolute prediction error in USD.
- **R² Score:** Measures how well the predictions explain variation in prices.

Laptop   HistGradientBoosting   MAE=$ 143.65  R2=0.8735
Desktop  HistGradientBoosting   MAE=$ 135.43  R2=0.8422

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd computer-laptop-price-predictor
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application should open in your browser.

Ensure the required trained model and dataset files are available in the expected locations before running the app.

## 📷 Application Preview

Add a screenshot of your running Streamlit application here.

## 🔮 Future Improvements

- Deploy the application publicly.
- Improve model evaluation using an untouched test set.
- Add clearer prediction uncertainty estimates.
- Improve input validation and the user interface.
- Expand the dataset and explore additional features.

## 👨‍💻 Author

**Saad Jamil**

BSCS Student | Aspiring Data Scientist 
