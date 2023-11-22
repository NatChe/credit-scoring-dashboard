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
PREVIOUS_APPLICATION_FILEPATH = os.path.join(ROOT_DIR, DATA_SOURCE_FOLDER, 'previous_application.csv')
POS_CASH_BALANCE_FILEPATH = os.path.join(ROOT_DIR, DATA_SOURCE_FOLDER, 'POS_CASH_balance.csv')
INSTALLMENTS_PAYMENTS_FILEPATH = os.path.join(ROOT_DIR, DATA_SOURCE_FOLDER, 'installments_payments.csv')
CREDIT_CARD_BALANCE_FILEPATH = os.path.join(ROOT_DIR, DATA_SOURCE_FOLDER, 'credit_card_balance.csv')


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

    # clean
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

    # clean
    del df_bureau, df_bureau_balance, closed, closed_agg, bureau
    gc.collect()

    return bureau_agg


def get_previous_applications_features(dev_mode=True):
    num_rows = DEV_SAMPLE_SIZE if dev_mode else None

    df_prev_app = pd.read_csv(PREVIOUS_APPLICATION_FILEPATH, nrows=num_rows)
    prev, cat_cols = one_hot_encoder(df_prev_app, nan_as_category=True)

    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        #'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }

    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')

    # clean
    del df_prev_app, refused, refused_agg, approved, approved_agg, prev
    gc.collect()

    return prev_agg


def get_pos_cash_balance_features(dev_mode=False):
    num_rows = DEV_SAMPLE_SIZE if dev_mode else None

    df_pos_cash = pd.read_csv(POS_CASH_BALANCE_FILEPATH, nrows=num_rows)
    pos_cash, cat_cols = one_hot_encoder(df_pos_cash, nan_as_category=True)

    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_cash_agg = pos_cash.groupby('SK_ID_CURR').agg(aggregations)
    pos_cash_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_cash_agg.columns.tolist()])

    # Count pos cash accounts
    pos_cash_agg['POS_COUNT'] = pos_cash.groupby('SK_ID_CURR').size()

    # clean
    del pos_cash, df_pos_cash
    gc.collect()

    return pos_cash_agg


def get_installments_payments_features(dev_mode=False):
    num_rows = DEV_SAMPLE_SIZE if dev_mode else None

    df_installments = pd.read_csv(INSTALLMENTS_PAYMENTS_FILEPATH, nrows=num_rows)

    installments, cat_cols = one_hot_encoder(df_installments, nan_as_category= True)

    # Percentage and difference paid in each installment (amount paid and installment value)
    installments['PAYMENT_PERC'] = installments['AMT_PAYMENT'] / installments['AMT_INSTALMENT']
    installments['PAYMENT_DIFF'] = installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']

    # Days past due and days before due (no negative values)
    installments['DPD'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    installments['DBD'] = installments['DAYS_INSTALMENT'] - installments['DAYS_ENTRY_PAYMENT']
    installments['DPD'] = installments['DPD'].apply(lambda x: x if x > 0 else 0)
    installments['DBD'] = installments['DBD'].apply(lambda x: x if x > 0 else 0)

    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
       # 'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    installments_agg = installments.groupby('SK_ID_CURR').agg(aggregations)
    installments_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in installments_agg.columns.tolist()])

    # Count installments accounts
    installments_agg['INSTALL_COUNT'] = installments.groupby('SK_ID_CURR').size()

    # clean
    del df_installments, installments
    gc.collect()

    return installments_agg


def get_credit_card_balance_features(dev_mode=False):
    num_rows = DEV_SAMPLE_SIZE if dev_mode else None

    df_cc_balance = pd.read_csv(CREDIT_CARD_BALANCE_FILEPATH, nrows=num_rows)
    cc_balance, cat_cols = one_hot_encoder(df_cc_balance, nan_as_category= True)

    # General aggregations
    cc_balance = cc_balance.drop(['SK_ID_PREV'], axis= 1)
    cc_balance_agg = cc_balance.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_balance_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_balance_agg.columns.tolist()])

    # Count credit card lines
    cc_balance_agg['CC_COUNT'] = cc_balance.groupby('SK_ID_CURR').size()

    # clean
    del df_cc_balance, cc_balance
    gc.collect()

    return cc_balance_agg
