from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import set_config
from transformers import ApplicationCleaner, ApplicationImputer, ApplicationEncoder, ApplicationFeaturesExtractor, ApplicationScaler

set_config(transform_output="pandas")

DEFAULT_CONFIG = {
    'preprocessing': {
        'should_fill_na': True,
        'num_imputer': SimpleImputer(strategy='median'),
        'cat_imputer': SimpleImputer(strategy='most_frequent'),
        'should_scale': False,
        'scaler': StandardScaler()
    }
}


def get_preprocessing_pipeline(preprocessing_config):
    steps = [
        ('cleaner', ApplicationCleaner()),
        ('feature_extractor', ApplicationFeaturesExtractor()),
    ]

    if preprocessing_config['should_fill_na']:
        steps.append(('imputer',
                      ApplicationImputer(num_imputer=preprocessing_config['num_imputer'],
                                         cat_imputer=preprocessing_config['cat_imputer']))
                     )

    steps.append(('encoder', ApplicationEncoder()))

    if preprocessing_config['should_scale']:
        steps.append(('scalar', ApplicationScaler(scaler=preprocessing_config['scaler'])))

    return Pipeline(steps=steps)


def log_reg_pipeline(config):
    preprocessing_pipeline = get_preprocessing_pipeline(config['preprocessing'])

    steps = [
        ('preprocessor', preprocessing_pipeline),
        ('classifier', LogisticRegression())
    ]

    return Pipeline(steps)


PIPELINES = {
    'log_regression': log_reg_pipeline
}
