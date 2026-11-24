# Diamond Price Predictor

Predicts diamond price (USD) from 9 attributes using the [Kaggle Diamonds dataset](https://www.kaggle.com/datasets/shivam2503/diamonds).

- `diamond_price_model.ipynb`: cleans the data, explores it, compares Linear Regression, Random Forest and XGBoost, and saves the best model.
- `app.py`: Streamlit web app that loads the saved model and predicts the price of a diamond from user inputs.

## Results (held-out test set, 20%)

| Model | CV R² | CV MAE |
|---|---|---|
| **XGBoost (selected)** | 0.982 | $260 |
| Random Forest | 0.981 | $271 |
| Linear Regression | 0.950 | $462 |

On the test set, the selected XGBoost model gets **R² 0.983**, **MAE $258** and **MAPE 6.1%**.

## Project structure

```
diamond-price-predictor/
├── app.py                        # Streamlit web app
├── diamond_price_model.ipynb     # Training notebook (already executed, outputs included)
├── data/diamonds.csv             # Dataset
├── models/
│   ├── diamond_model.joblib      # Full pipeline: encoding → XGBoost → log back-transform
│   └── model_metadata.json       # Input ranges, grade orders, metrics
├── requirements.txt
└── README.md
```

## Run it

```bash
cd diamond-price-predictor
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: retrain (the model is already in models/)
jupyter notebook diamond_price_model.ipynb      # Run All

# Start the web app
streamlit run app.py               # opens http://localhost:8501
```

## Inputs

| Column | Type | Values |
|---|---|---|
| carat | number | 0.20 – 5.01 |
| cut | grade | Fair, Good, Very Good, Premium, Ideal |
| color | grade | J (worst) … D (best) |
| clarity | grade | I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF |
| depth | % | z / mean(x, y) × 100 |
| table | % | top-facet width relative to widest point |
| x, y, z | mm | length, width, depth |

The app limits inputs to the ranges seen in training. It also warns when carat doesn't match x·y·z, or when depth % doesn't match z / mean(x, y).

## Notes

- **Cleaning:** removes duplicate rows, rows where x, y or z is 0, and dimension typos above 15 mm. That leaves 53,772 rows.
- **Target:** the model is trained on `log(1 + price)` and its predictions are converted back to dollars automatically.
- **Version pinning:** `scikit-learn` and `xgboost` are pinned to the versions used to save the model, because joblib files may not load across versions. If you change them, re-run the notebook to regenerate the model.
- **Correlated inputs:** carat, x, y and z are highly correlated, so the feature-importance ranking spreads importance across them arbitrarily.
