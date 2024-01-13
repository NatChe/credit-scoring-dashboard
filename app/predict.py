import os
import joblib
import dask.dataframe as dd
import pandas as pd
import shap
from sklearn import set_config

ROOT_DIR = os.getcwd()
MODEL_PATH = os.path.join(ROOT_DIR, 'model/pipeline_lightGBM_model.pkl')
SCALER_PATH = os.path.join(ROOT_DIR, 'model/scaler.pkl')
TEST_DATA_PATH = os.path.join(ROOT_DIR, 'data/cleaned/test_processed.csv')
TEST_SCALED_DATA_PATH = os.path.join(ROOT_DIR, 'data/cleaned/test_scaled.csv')
TRAIN_DATA_PATH = os.path.join(ROOT_DIR, 'data/cleaned/train_processed.csv')

FEATURES_ORDER = ['CODE_GENDER_F', 'NAME_EDUCATION_TYPE_Higher_education',
                  'NAME_FAMILY_STATUS_Married', 'ORGANIZATION_TYPE_Self_employed',
                  'AMT_CREDIT', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OWN_CAR_AGE',
                  'REGION_RATING_CLIENT', 'DAYS_LAST_PHONE_CHANGE',
                  'BUREAU_DAYS_CREDIT_MAX', 'BUREAU_DAYS_CREDIT_ENDDATE_MAX',
                  'BUREAU_AMT_CREDIT_SUM_MAX', 'BUREAU_AMT_CREDIT_SUM_DEBT_MEAN',
                  'ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN', 'PREV_AMT_ANNUITY_MIN',
                  'PREV_DAYS_DECISION_MIN', 'PREV_CNT_PAYMENT_MEAN',
                  'REFUSED_AMT_APPLICATION_MIN', 'POS_MONTHS_BALANCE_MAX',
                  'INSTAL_DPD_MAX', 'INSTAL_PAYMENT_DIFF_MEAN', 'CC_AMT_BALANCE_MIN',
                  'FLAG_OWN_CAR', 'FLAG_DOCUMENT_3']


set_config(transform_output="pandas")

def load_model():
    model = joblib.load(MODEL_PATH)

    return model


def load_scaler():
    scaler = joblib.load(SCALER_PATH)

    return scaler


def load_client_data(client_id):
    test_df = dd.read_csv(TEST_DATA_PATH)
    test_df = test_df[test_df['SK_ID_CURR'] == int(client_id)]
    test_df = test_df.compute()

    test_df = test_df.drop('SK_ID_CURR', axis=1)

    return test_df


def scale_data(data):
    scaler = load_scaler()

    X_client_scaled = scaler.transform(data)
    X_client_scaled = X_client_scaled.reindex(columns=FEATURES_ORDER)

    return X_client_scaled


def load_scaled_client_data(client_id):
    test_df = dd.read_csv(TEST_SCALED_DATA_PATH)
    test_df = test_df[test_df['SK_ID_CURR'] == int(client_id)]
    test_df = test_df.compute()

    test_df = test_df.drop('SK_ID_CURR', axis=1)

    return test_df


def process_client_data(client_id):
    X_client = load_client_data(client_id)

    return X_client


def predict(client_id):
    model = load_model()
    X_client = load_scaled_client_data(client_id)

    target = model.predict(X_client)
    target_proba = model.predict_proba(X_client)

    return {
        'target': int(target[0]),
        'proba': float(target_proba[::, 1][0])
    }


def explain(client_id):
    model = load_model()
    X_client_scaled = load_scaled_client_data(client_id)
    X_client_data = load_client_data(client_id)
    X_client_data = X_client_data.reindex(columns=FEATURES_ORDER)

    tree_explainer = shap.TreeExplainer(model)
    shap_values = tree_explainer.shap_values(X_client_scaled)

    return {
        'expected_value': tree_explainer.expected_value[1],
        'shap_values': list(shap_values[1][0]),
        'features': X_client_data.to_dict()
    }


def simulate_predict(data):
    model = load_model()

    X_client_simulation = pd.read_json(data)
    X_client_scaled = scale_data(X_client_simulation)

    target = model.predict(X_client_scaled)
    target_proba = model.predict_proba(X_client_scaled)

    return {
        'target': int(target[0]),
        'proba': float(target_proba[::, 1][0])
    }


def analyse_feature(feature_name):
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    return train_df[[feature_name, 'TARGET']].to_dict()


def get_features_df(features):
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    return train_df[features + ['TARGET']].to_dict()
