import os
import joblib
import dask.dataframe as dd
import sys
import shap
from sklearn import set_config

ROOT_DIR = os.getcwd()
MODEL_PATH = os.path.join(ROOT_DIR, 'model/pipeline_lightGBM.pkl')
TEST_DATA_PATH = os.path.join(ROOT_DIR, 'data/source/application_test.csv')

sys.path.insert(0, ROOT_DIR)
set_config(transform_output="pandas")


def load_model():
    return joblib.load(MODEL_PATH)


def load_client_data(client_id):
    test_df = dd.read_csv(TEST_DATA_PATH)
    test_df = test_df[test_df['SK_ID_CURR'] == int(client_id)]
    test_df = test_df.compute()

    return test_df


def process_client_data(client_id):
    set_config(transform_output="pandas")

    model_pipeline = load_model()
    X_client = load_client_data(client_id)
    X_client_processed = model_pipeline.named_steps["preprocessor"].transform(X_client)

    return X_client_processed



def predict(client_id):
    set_config(transform_output="pandas")
    model = load_model()
    X_client = load_client_data(client_id)

    target = model.predict(X_client)
    target_proba = model.predict_proba(X_client)

    return {
        'target': int(target[0]),
        'proba': float(target_proba[::, 1][0])
    }


def explain(client_id):
    set_config(transform_output="pandas")
    model_pipeline = load_model()
    X_client = load_client_data(client_id)

    model = model_pipeline.named_steps["classifier"]
    X_client_processed = model_pipeline.named_steps["preprocessor"].transform(X_client)

    tree_explainer = shap.TreeExplainer(model)
    shap_values = tree_explainer.shap_values(X_client_processed)

    return {
        'expected_value': tree_explainer.expected_value[1],
        'shap_values': list(shap_values[1][0]),
        'features': X_client_processed.to_dict()
    }






