import os
import joblib
import dask.dataframe as dd
import sys

import pandas as pd
import shap
from sklearn import set_config

ROOT_DIR = os.getcwd()
MODEL_PATH = os.path.join(ROOT_DIR, 'model/pipeline_lightGBM.pkl')
TEST_DATA_PATH = os.path.join(ROOT_DIR, 'data/cleaned/test_processed.csv')
TRAIN_DATA_PATH = os.path.join(ROOT_DIR, 'data/cleaned/train_processed.csv')

sys.path.insert(0, ROOT_DIR)

def load_model():
    model_pipeline = joblib.load(MODEL_PATH)

    return model_pipeline.named_steps["classifier"]


def load_client_data(client_id):
    test_df = dd.read_csv(TEST_DATA_PATH)
    test_df = test_df[test_df['SK_ID_CURR'] == int(client_id)]
    test_df = test_df.compute()

    test_df = test_df.drop('SK_ID_CURR', axis=1)

    return test_df


def process_client_data(client_id):

    X_client = load_client_data(client_id)

    return X_client


def predict(client_id):
    model = load_model()
    X_client = load_client_data(client_id)

    target = model.predict(X_client)
    target_proba = model.predict_proba(X_client)

    return {
        'target': int(target[0]),
        'proba': float(target_proba[::, 1][0])
    }


def explain(client_id):
    model = load_model()
    X_client = load_client_data(client_id)

    tree_explainer = shap.TreeExplainer(model)
    shap_values = tree_explainer.shap_values(X_client)

    return {
        'expected_value': tree_explainer.expected_value[1],
        'shap_values': list(shap_values[1][0]),
        'features': X_client.to_dict()
    }


def simulate_predict(data):
    model = load_model()

    X_client_simulation = pd.read_json(data)

    target = model.predict(X_client_simulation)
    target_proba = model.predict_proba(X_client_simulation)

    return {
        'target': int(target[0]),
        'proba': float(target_proba[::, 1][0])
    }


def analyse_feature(feature_name):
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    return train_df[[feature_name, 'TARGET']].to_dict()
