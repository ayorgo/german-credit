import numpy as np
import pandas as pd
import mapper
import sequence

from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.compose import make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (OrdinalEncoder, FunctionTransformer,
                                   StandardScaler, OneHotEncoder)


class ThresholdBinner(BaseEstimator, TransformerMixin):

    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X):
        counts = X.iloc[:,0].value_counts(normalize=True).sort_index()
        self.max_value = counts.index.max()
        self.above_threshold = counts[counts >= self.threshold].index.values
        return self

    def transform(self, X):
        def mapper(x):
            for i in self.above_threshold:
                if x <= i:
                    return i
            return self.max_value
        return X.iloc[:,0].apply(mapper).values.reshape(-1, 1)


class NominalMapper(BaseEstimator, TransformerMixin):

    def __init__(self, mapper):
        self.mapper = mapper

    def fit(self, X):
        return self

    def transform(self, X):
        return X.iloc[:,0].map(self.mapper).values.reshape(-1, 1)


def _nominal_selector(X):
    return pd.DataFrame(X).infer_objects().dtypes == object


def specific():
    tuples = ((NominalMapper(mapper.credit_history), ['credit_history']),
              (NominalMapper(mapper.purpose), ['purpose']),
              (NominalMapper(mapper.sex), ['personal_status']),
              (NominalMapper(mapper.status), ['personal_status']),
              (NominalMapper(mapper.job), ['job']),
              (OrdinalEncoder(sequence.over_draft), ['over_draft']),
              (OrdinalEncoder(sequence.acb), ['Average_Credit_Balance']),
              (OrdinalEncoder(sequence.employment), ['employment']),
              (ThresholdBinner(0.05), ['credit_usage']))

    return make_column_transformer(*tuples, remainder='passthrough')


def nominal():
    def select_nominal(X):
        return X[:, _nominal_selector(X)]

    return FunctionTransformer(select_nominal, validate=False)


def numeric():
    def select_numeric(X):
        return X[:, np.logical_not(_nominal_selector(X))]

    return FunctionTransformer(select_numeric, validate=False)


def onehot():
    return make_column_transformer((OneHotEncoder(drop='first'),
                                    _nominal_selector),
                                   remainder='passthrough')


def scale():
    return StandardScaler()
