import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.config import (
    LOGREG_PARAMS,
    RF_PARAMS,
    CLASSIFIER_DIR,
    RESULTS_DIR,
    TARGET_COL,
    METADATA_COLS,
    POSITIVE_CLASS
)

TRAIN_PCA_FILE = RESULTS_DIR / 'training_pca_features.csv'

def build_classifier(classifier_type: str) -> Pipeline:
    if classifier_type == 'logistic':
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(**LOGREG_PARAMS))
        ])
    elif classifier_type == 'random_forest':
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(**RF_PARAMS))
        ])
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

def train_classifier(classifier_type: str):
    df = pd.read_csv(TRAIN_PCA_FILE)
    train_df = df[df['batch'] == 'discovery'].copy()
    X = train_df.drop(columns=METADATA_COLS + [TARGET_COL])
    y = (train_df[TARGET_COL] == POSITIVE_CLASS).astype(int)

    model = build_classifier(classifier_type)
    model.fit(X, y)

    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CLASSIFIER_DIR / f"{classifier_type}_classifier.joblib"
    joblib.dump(model, model_path)
    print(f"✓ {classifier_type} model saved to: {model_path}")

    return model, X, y 
