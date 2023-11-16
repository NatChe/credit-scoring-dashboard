import numpy as np
import pandas as pd
import os, gc

RANDOM_SEED = 42
DEV_SAMPLE_SIZE = 5000

ID_COLUMN = 'SK_ID_CURR'
TARGET_COLUMN = 'TARGET'

ROOT_DIR = os.curdir
DATA_SOURCE_FOLDER = 'data/source'
APPLICATION_TRAIN_FILEPATH = os.path.join(ROOT_DIR, DATA_SOURCE_FOLDER, 'application_train.csv')
BUREAU_FILEPATH = os.path.join(ROOT_DIR, DATA_SOURCE_FOLDER, 'bureau.csv')
BUREAU_BALANCE_FILEPATH = os.path.join(ROOT_DIR, DATA_SOURCE_FOLDER, 'bureau_balance.csv')

'''credit_card_balance_filepath: YOUR / PATH / TO / credit_card_balance.csv
installments_payments_filepath: YOUR / PATH / TO / installments_payments.csv
POS_CASH_balance_filepath: YOUR / PATH / TO / POS_CASH_balance.csv
previous_application_filepath: YOUR / PATH / TO / previous_application.csv
sample_submission_filepath: YOUR / PATH / TO / sample_submission.csv
'''


def load_data(dev_mode=False):
    """
    Loads the data

    Input:
        dev_mode: if set to True, load only a sample of data

    Output:

    """

    num_rows = DEV_SAMPLE_SIZE if dev_mode else None

    raw_data = {}

    print('Loading application_train ...')
    application_train = pd.read_csv(APPLICATION_TRAIN_FILEPATH, nrows=num_rows)
    raw_data['application'] = application_train
    raw_data['train_set'] = pd.DataFrame(application_train[[ID_COLUMN, TARGET_COLUMN]])

    print("Loading Done.")

    return raw_data


def load_bureau_and_balance(dev_mode=True):
    num_rows = DEV_SAMPLE_SIZE if dev_mode else None

    raw_data = {'bureau': pd.read_csv(BUREAU_FILEPATH, nrows=num_rows),
                'bureau_balance': pd.read_csv(BUREAU_BALANCE_FILEPATH, nrows=num_rows)}

    return raw_data


def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# TODO: create a transformer
def get_bureau_and_balance_features(dev_mode=True, nan_as_category=True):
    num_rows = DEV_SAMPLE_SIZE if dev_mode else None

    df_bureau = pd.read_csv(BUREAU_FILEPATH, nrows=num_rows)
    df_bureau_balance = pd.read_csv(BUREAU_BALANCE_FILEPATH, nrows=num_rows)

    bb, bb_cat = one_hot_encoder(df_bureau_balance, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(df_bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']

    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()

    return bureau_agg



