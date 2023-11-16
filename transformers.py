import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class ApplicationCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['CODE_GENDER'] = X['CODE_GENDER'].replace('XNA', 'Unknown')
        X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)
        X['DAYS_LAST_PHONE_CHANGE'] = X['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan)
        X['ORGANIZATION_TYPE'] = X['ORGANIZATION_TYPE'].replace('XNA', 'Unknown')

        # Map category binary columns to 0 and 1
        flag_cols_dict = {'N': 0, 'Y': 1, 'No': 0, 'Yes': 1}
        X['FLAG_OWN_CAR'] = X['FLAG_OWN_CAR'].apply(lambda x: flag_cols_dict.get(x))
        X['FLAG_OWN_REALTY'] = X['FLAG_OWN_REALTY'].apply(lambda x: flag_cols_dict.get(x))
        X['EMERGENCYSTATE_MODE'] = X['EMERGENCYSTATE_MODE'].apply(lambda x: flag_cols_dict.get(x))

        # convert object types to categorical
        object_columns = X.select_dtypes(['object']).columns
        for object_column in object_columns:
            X[object_column] = X[object_column].astype("category")

        return X


class ApplicationImputer(BaseEstimator, TransformerMixin):
    def __init__(self, num_imputer, cat_imputer):
        self.num_imputer = num_imputer
        self.cat_imputer = cat_imputer
        self.imputer = None

    def fit(self, X, y=None):
        # drop columns with > 60% of null values
        X = X.dropna(axis=1, thresh=int(0.4 * X.shape[0]))

        numerical_features = X.drop('SK_ID_CURR', axis=1).select_dtypes(['int64', 'float64']).columns
        categorical_features = X.select_dtypes(['object']).columns

        self.imputer = ColumnTransformer(
            transformers=[
                ('num', self.num_imputer, numerical_features),
                ('cat', self.cat_imputer, categorical_features),
            ], remainder='passthrough', verbose_feature_names_out=False
        )

        self.imputer.fit(X)

        return self

    def transform(self, X):

        X = self.imputer.transform(X)

        return X


class ApplicationEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = None

    def fit(self, X, y=None):
        categorical_features = X.select_dtypes(['category']).columns

        transformers = [
            ('cat_encoder', OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features)
        ]

        self.encoder = ColumnTransformer(transformers=transformers, remainder='passthrough',
                                         verbose_feature_names_out=False)
        self.encoder.fit(X)

        return self

    def transform(self, X):

        X = self.encoder.transform(X)

        return X


class ApplicationScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler
        self.transformer = None

    def fit(self, X, y=None):
        numerical_features = X.drop('SK_ID_CURR', axis=1).select_dtypes(['int64', 'float64']).columns
        non_binary_numerical_features = [col for col in numerical_features if not col.startswith('FLAG')]
        transformers = [('scaler', self.scaler, non_binary_numerical_features)]

        self.transformer = ColumnTransformer(transformers=transformers, remainder='passthrough',
                                             verbose_feature_names_out=False)
        self.transformer.fit(X)

        return self

    def transform(self, X):

        X = self.transformer.transform(X)

        return X


class ApplicationFeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['DAYS_EMPLOYED_PERC'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
        X['INCOME_CREDIT_PERC'] = X['AMT_INCOME_TOTAL'] / X['AMT_CREDIT']
        X['INCOME_PER_PERSON'] = X['AMT_INCOME_TOTAL'] / X['CNT_FAM_MEMBERS']
        X['INCOME_PER_CHILD'] = X['AMT_INCOME_TOTAL'] / (1 + X['CNT_CHILDREN'])
        X['ANNUITY_INCOME_PERC'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
        X['PAYMENT_RATE'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
        X['CHILDREN_RATIO'] = X['CNT_CHILDREN'] / X['CNT_FAM_MEMBERS']

        return X


class ApplicationFeaturesMerger(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_merge):
        self.id_column = 'SK_ID_CURR'
        self.features_to_merge = features_to_merge

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print('shape before ', X.shape)
        X_merged = X.join(self.features_to_merge, how='left', on=self.id_column)
        print('new shape after merge', X_merged.shape)
        return X_merged
