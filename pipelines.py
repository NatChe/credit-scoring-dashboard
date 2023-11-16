from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn import set_config
from imblearn.pipeline import Pipeline as Pipeline_imb
from imblearn.over_sampling import SMOTENC, SMOTE
import transformers
from data_preprocessing import get_bureau_and_balance_features

set_config(transform_output="pandas")

DEFAULT_CONFIG = {
    'preprocessing': {
        'should_fill_na': True,
        'num_imputer': SimpleImputer(strategy='median'),
        'cat_imputer': SimpleImputer(strategy='most_frequent'),
        'should_scale': False,
        'scaler': StandardScaler(),
        'use_bureau': False,
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

    if preprocessing_config['use_bureau']:
        X_bureau_features = get_bureau_and_balance_features(dev_mode)
        steps.append(('merge_bureau_and_balance', transformers.ApplicationFeaturesMerger(X_bureau_features)))

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
    preprocessing_config = config['preprocessing']
    balancing_config = config['balancing']

    steps = get_preprocessing_steps(preprocessing_config, balancing_config, dev_mode)

    if balancing_config['should_oversample']:
        steps.append(('classifier', LogisticRegression(**config['model_params'])))
        return Pipeline_imb(steps, verbose=True)

    preprocessing_pipeline = Pipeline(steps=steps, verbose=True)
    steps = [('preprocessor', preprocessing_pipeline), ('classifier', LogisticRegression(**config['model_params']))]

    return Pipeline(steps, verbose=True)


PIPELINES = {
    'log_regression': log_reg_pipeline,
    'dummy': dummy_classifier_pipeline
}
