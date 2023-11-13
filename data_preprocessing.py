import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
DEV_SAMPLE_SIZE = 1000

ID_COLUMN = 'SK_ID_CURR'
TARGET_COLUMN = 'TARGET'

ROOT_DIR = os.curdir
DATA_SOURCE_FOLDER = 'data/source'
APPLICATION_TRAIN_FILEPATH = os.path.join(ROOT_DIR, DATA_SOURCE_FOLDER, 'application_train.csv')

'''bureau_balance_filepath: YOUR / PATH / TO / bureau_balance.csv
bureau_filepath: YOUR / PATH / TO / bureau.csv
credit_card_balance_filepath: YOUR / PATH / TO / credit_card_balance.csv
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

