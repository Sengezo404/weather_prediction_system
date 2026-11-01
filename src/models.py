from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingRegressor,
    VotingClassifier,
    StackingRegressor,
    StackingClassifier,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def regression_models(random_state: int = 42):
    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])

    extra = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    hist = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=0.5,
            random_state=random_state,
        )),
    ])

    ensemble = VotingRegressor([
        ("ridge", ridge),
        ("extra", extra),
        ("rf", rf),
        ("hist", hist),
    ])

    return {
        "ridge": ridge,
        "extra_trees": extra,
        "random_forest": rf,
        "hist_gradient_boosting": hist,
        "ensemble": ensemble,
    }


def classification_models(random_state: int = 42):
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    extra = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])

    hist = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=0.5,
            random_state=random_state,
        )),
    ])

    ensemble = VotingClassifier(
        [
            ("logistic", logistic),
            ("extra", extra),
            ("rf", rf),
            ("hist", hist),
        ],
        voting="soft",
    )

    return {
        "logistic": logistic,
        "extra_trees": extra,
        "random_forest": rf,
        "hist_gradient_boosting": hist,
        "ensemble": ensemble,
    }
