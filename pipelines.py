from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn import set_config
from imblearn.pipeline import Pipeline as Pipeline_imb
from imblearn.over_sampling import SMOTENC, SMOTE
from transformers import ApplicationCleaner, ApplicationImputer, ApplicationEncoder, ApplicationFeaturesExtractor, ApplicationScaler

set_config(transform_output="pandas")

DEFAULT_CONFIG = {
    'preprocessing': {
        'should_fill_na': True,
        'num_imputer': SimpleImputer(strategy='median'),
        'cat_imputer': SimpleImputer(strategy='most_frequent'),
        'should_scale': False,
        'scaler': StandardScaler(),
    },
    'balancing': {
        'should_oversample': False,
        'with_categorical': False
    },
    'model_params': {}
}


def get_preprocessing_steps(preprocessing_config, balancing_config):
    steps = [
        ('cleaner', ApplicationCleaner()),
        ('feature_extractor', ApplicationFeaturesExtractor()),
    ]

    if preprocessing_config['should_fill_na']:
        steps.append(('imputer',
                      ApplicationImputer(num_imputer=preprocessing_config['num_imputer'],
                                         cat_imputer=preprocessing_config['cat_imputer']))
                     )

    if preprocessing_config['should_scale']:
        steps.append(('scalar', ApplicationScaler(scaler=preprocessing_config['scaler'])))

    if balancing_config['should_oversample'] and balancing_config['with_categorical']:
        steps.append(('smote_nc',
                      SMOTENC(
                          categorical_features='auto',
                          categorical_encoder=OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                          random_state=18))
                     )

    steps.append(('encoder', ApplicationEncoder()))

    if balancing_config['should_oversample'] and not balancing_config['with_categorical']:
        steps.append(('smote', SMOTE(random_state=18)))

    return steps


def get_preprocessing_pipeline(preprocessing_config):
    steps = get_preprocessing_steps(preprocessing_config)

    preprocessing_pipeline = Pipeline(steps=steps, verbose=True)

    return preprocessing_pipeline


def dummy_classifier_pipeline(config):
    return DummyClassifier(strategy=config['strategy'])


def log_reg_pipeline(config):
    preprocessing_config = config['preprocessing']
    balancing_config = config['balancing']

    steps = get_preprocessing_steps(preprocessing_config, balancing_config)

    if balancing_config['should_oversample']:
        steps.append(('classifier', LogisticRegression(**config['model_params'])))
        #print(steps)
        return Pipeline_imb(steps, verbose=True)

    preprocessing_pipeline = Pipeline(steps=steps, verbose=True)
    steps = [('preprocessor', preprocessing_pipeline), ('classifier', LogisticRegression(**config['model_params']))]

    return Pipeline(steps, verbose=True)


PIPELINES = {
    'log_regression': log_reg_pipeline,
    'dummy': dummy_classifier_pipeline
}
