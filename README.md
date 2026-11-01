# Mahikeng Weather Prediction System

A production-oriented version of the weather ML project developed from the
`learning_by doing.ipynb` notebook.

## Current scope

The system predicts:

1. Tomorrow's maximum temperature.
2. Tomorrow's minimum temperature.
3. Whether measurable rain is expected tomorrow.
4. Rain probability for tomorrow.

The first version is deliberately Mahikeng-specific and uses the NASA POWER
historical dataset already used in the notebook.

## Why this is different from the notebook

The notebook is useful for learning and experimentation. This project separates:

- data loading and cleaning
- leakage-safe feature engineering
- model definitions
- training
- evaluation
- prediction
- Streamlit presentation
- tests

Important corrections from the notebook are included:

- NASA POWER `-999.0` values are converted to missing values before imputation.
- Missing values are not blindly replaced with zero.
- Rolling features use `shift(1)` so future information cannot enter the feature.
- Targets are explicitly named.
- Training/testing follows chronological order instead of random shuffling.

## Project structure

```text
weather_prediction_system/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   │   └── mahikeng_weather.csv
│   └── processed/
├── models/
├── src/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── evaluate.py
│   ├── train.py
│   └── predict.py
└── tests/
    ├── test_data.py
    └── test_features.py
```

## Setup in VS Code

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the existing NASA POWER file at:

```text
data/raw/mahikeng_weather.csv
```

## Train

From the project root:

```bash
python -m src.train
```

If the NASA POWER file has a different number of metadata rows:

```bash
python -m src.train --skiprows 15
```

Training saves models into `models/` and processed data into
`data/processed/weather_processed.csv`.

## Run the application

```bash
streamlit run app.py
```

## Run tests

```bash
pytest
```

## Model strategy

The project keeps the Ridge model from the original notebook as a baseline
and adds tree-based models.

Regression:

- Ridge
- Extra Trees
- Random Forest
- HistGradientBoosting
- Voting ensemble

Rain classification:

- Logistic Regression
- Extra Trees
- Random Forest
- HistGradientBoosting
- Soft-voting ensemble

The ensemble is evaluated rather than assumed to be better.

## Important academic point

Do not report the ensemble as "best" until the test results actually show it
outperforms the individual models. The report should compare all models using
the same chronological test period.

## Next development stage

After this pipeline works locally:

1. Add rolling-origin backtesting.
2. Tune model hyperparameters.
3. Add a live weather API.
4. Build a richer Streamlit dashboard.
5. Add model feature importance/explainability.
6. Add deployment.
