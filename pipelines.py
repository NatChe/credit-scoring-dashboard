from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn import set_config
from imblearn.pipeline import Pipeline as Pipeline_imb
from imblearn.over_sampling import SMOTENC, SMOTE
import transformers
import data_preprocessing

set_config(transform_output="pandas")

DEFAULT_CONFIG = {
    'preprocessing': {
        'should_fill_na': True,
        'num_imputer': SimpleImputer(strategy='median'),
        'cat_imputer': SimpleImputer(strategy='most_frequent'),
        'should_scale': False,
        'scaler': StandardScaler(),
        'use_bureau_and_balance': False,
        'use_previous_applications': False,
        'use_pos_cash_balance': False,
        'use_installments_payments': False,
        'use_credit_card_balance': False,
    },
    'balancing': {
        'should_oversample': False,
        'with_categorical': False
    },
    'model_params': {}
}


def get_preprocessing_steps(preprocessing_config, balancing_config, dev_mode):
    steps = [
        ('cleaner', transformers.ApplicationCleaner()),
        ('feature_extractor', transformers.ApplicationFeaturesExtractor()),
    ]

    # add bureau balance features
    if preprocessing_config['use_bureau_and_balance']:
        X_bureau_features = data_preprocessing.get_bureau_and_balance_features(dev_mode)
        steps.append(('merge_bureau_and_balance', transformers.ApplicationFeaturesMerger(X_bureau_features)))

    # add previous applications features
    if preprocessing_config['use_previous_applications']:
        X_prev_app_features = data_preprocessing.get_previous_applications_features(dev_mode)
        steps.append(('merge_previous_applications', transformers.ApplicationFeaturesMerger(X_prev_app_features)))

    # add pos cash balance features
    if preprocessing_config['use_pos_cash_balance']:
        X_pos_cash_balance_features = data_preprocessing.get_pos_cash_balance_features(dev_mode)
        steps.append(('merge_pos_cash_balance', transformers.ApplicationFeaturesMerger(X_pos_cash_balance_features)))

    # add installments payments features
    if preprocessing_config['use_installments_payments']:
        X_installment_features = data_preprocessing.get_installments_payments_features(dev_mode)
        steps.append(('merge_installments_payments', transformers.ApplicationFeaturesMerger(X_installment_features)))

    # add credit card balance features
    if preprocessing_config['use_credit_card_balance']:
        X_cc_balance_features = data_preprocessing.get_credit_card_balance_features(dev_mode)
        steps.append(('merge_credit_card_balance', transformers.ApplicationFeaturesMerger(X_cc_balance_features)))

    # fill missing values
    if preprocessing_config['should_fill_na']:
        steps.append(('imputer',
                      transformers.ApplicationImputer(num_imputer=preprocessing_config['num_imputer'],
                                                      cat_imputer=preprocessing_config['cat_imputer']))
                     )

    if preprocessing_config['should_scale']:
        steps.append(('scalar', transformers.ApplicationScaler(scaler=preprocessing_config['scaler'])))

    if balancing_config['should_oversample'] and balancing_config['with_categorical']:
        steps.append(('smote_nc',
                      SMOTENC(
                          categorical_features='auto',
                          categorical_encoder=OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                          random_state=18))
                     )

    steps.append(('encoder', transformers.ApplicationEncoder()))

    if balancing_config['should_oversample'] and not balancing_config['with_categorical']:
        steps.append(('smote', SMOTE(random_state=18)))

    return steps


def dummy_classifier_pipeline(config):
    return DummyClassifier(strategy=config['strategy'])


def log_reg_pipeline(config, dev_mode=False):
    verbose = True if dev_mode else False

    preprocessing_config = config['preprocessing']
    balancing_config = config['balancing']

    # get preprocessing steps
    steps = get_preprocessing_steps(preprocessing_config, balancing_config, dev_mode)

    # define the classifier
    classifier = LogisticRegression(
        random_state=config['model_params']['random_state'],
        max_iter=2000,
        **config['model_params']['params']
    )

    # call imblearn pipeline if oversampling
    if balancing_config['should_oversample']:
        steps.append(('classifier', classifier))
        return Pipeline_imb(steps, verbose=verbose)

    # call sklearn pipeline
    preprocessing_pipeline = Pipeline(steps=steps, verbose=verbose)
    steps = [
        ('preprocessor', preprocessing_pipeline),
        ('classifier', classifier)
    ]

    return Pipeline(steps, verbose=verbose)


PIPELINES = {
    'log_regression': log_reg_pipeline,
    'dummy': dummy_classifier_pipeline
}
