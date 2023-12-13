import os
import joblib
import dask.dataframe as dd
import sys
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


def predict(client_id):
    set_config(transform_output="pandas")
    model = load_model()
    X_client = load_client_data(client_id)

    # TODO set up 404 response http code
    if X_client.shape[0] == 0:
        return {'Error': 'Client not found'}

    target = model.predict(X_client)
    target_proba = model.predict_proba(X_client)

    return {
        'target': int(target[0]),
        'proba': float(target_proba[::, 1][0])
    }
