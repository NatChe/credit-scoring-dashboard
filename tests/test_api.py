import pytest
import json
from api import app

TEST_CLIENT_ID = 100001

def test_client_id_route_200():
    response = app.test_client().get(f'/clients/{TEST_CLIENT_ID}')
    assert response.status_code == 200

    data = json.loads(response.data.decode('utf-8'))
    assert type(data) is dict
    assert 'AMT_CREDIT' in data

def test_client_id_route_404():
    response = app.test_client().get('/clients/1')

    assert response.status_code == 404


def test_client_scores():
    response = app.test_client().get(f'/clients/{TEST_CLIENT_ID}/scores')
    assert response.status_code == 200

    data = json.loads(response.data.decode('utf-8'))
    assert 'target' in data
    assert 'proba' in data

    assert type(data['target']) is int
    assert type(data['proba']) is float


def test_client_features_explained():
    response = app.test_client().get(f'/clients/{TEST_CLIENT_ID}/features_explained')
    assert response.status_code == 200

    data = json.loads(response.data.decode('utf-8'))
    assert 'expected_value' in data
    assert 'shap_values' in data
    assert 'features' in data

    assert type(data['expected_value']) is float
    assert type(data['shap_values']) is list
    assert type(data['features']) is dict


def test_client_simulate():
    payload = '{"CODE_GENDER_F":{"0":1},"NAME_EDUCATION_TYPE_Higher_education":{"0":1},"NAME_FAMILY_STATUS_Married":{"0":1},"ORGANIZATION_TYPE_Self_employed":{"0":0},"FLAG_OWN_CAR":{"0":0},"AMT_CREDIT":{"0":568800.0},"DAYS_BIRTH":{"0":-19241},"DAYS_EMPLOYED":{"0":-2329},"OWN_CAR_AGE":{"0":0},"REGION_RATING_CLIENT":{"0":2},"DAYS_LAST_PHONE_CHANGE":{"0":-1740.0},"FLAG_DOCUMENT_3":{"0":1},"BUREAU_DAYS_CREDIT_MAX":{"0":-49},"BUREAU_DAYS_CREDIT_ENDDATE_MAX":{"0":1778.0},"BUREAU_AMT_CREDIT_SUM_MAX":{"0":378000.0},"BUREAU_AMT_CREDIT_SUM_DEBT_MEAN":{"0":85240.9285714286},"ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN":{"0":0},"PREV_AMT_ANNUITY_MIN":{"0":3951.0},"PREV_DAYS_DECISION_MIN":{"0":-1740},"PREV_CNT_PAYMENT_MEAN":{"0":8.0},"REFUSED_AMT_APPLICATION_MIN":{"0":0},"POS_MONTHS_BALANCE_MAX":{"0":-53},"INSTAL_DPD_MAX":{"0":11.0},"INSTAL_PAYMENT_DIFF_MEAN":{"0":0},"CC_AMT_BALANCE_MIN":{"0":0}}'

    response = app.test_client().post(f'/clients/{TEST_CLIENT_ID}/simulate', json=payload)
    data = json.loads(response.data.decode('utf-8'))
    assert 'target' in data
    assert 'proba' in data

    assert type(data['target']) is int
    assert type(data['proba']) is float


def test_client_feature_profiling():
    response = app.test_client().get(f'/features/AMT_CREDIT')
    assert response.status_code == 200

    data = json.loads(response.data.decode('utf-8'))
    assert type(data) is dict
    assert 'AMT_CREDIT' in data
    assert 'TARGET' in data


def test_client_features_profiling():
    response = app.test_client().get(f'/features?q=AMT_CREDIT,DAYS_BIRTH')
    assert response.status_code == 200

    data = json.loads(response.data.decode('utf-8'))
    assert type(data) is dict
    assert 'AMT_CREDIT' in data
    assert 'DAYS_BIRTH' in data
    assert 'TARGET' in data