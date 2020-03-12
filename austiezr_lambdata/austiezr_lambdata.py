import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae

"""
TransformDF:
    date_split: handles date columns
    add_to_df: appends columns

MVP:
    fast_first: quick results for EDA and baselines
"""


class TransformDF:
    """base class for all df transformations."""

    @staticmethod
    def date_split(df, date_col):
        """take date column, convert to DateTime, split into relevant columns, return data frame."""
        df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors='raise')
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_year'] = df[date_col].dt.dayofyear
        return df

    @staticmethod
    def add_to_df(new_list, df):
        """convert list to series and create new column containing series."""
        new_list = pd.Series(new_list)
        df['new_column'] = new_list
        return df


class MVP:
    """
    Base class for quickly producing Minimum Viable Product for EDA and Baselines
    Expects a predictive model

    Methods available:

    fast_first - quick EDA tool
    """

    def __init__(self, model):
        """requires a predictive model at construction."""
        self.model = model

    def fast_first(self, df, target):
        """
        Quick method for producing baseline results.
        return baseline, split data, fit basic encoder/imputer/model
        return accuracy for classification, MAE for regression

        Keyword arguments:

        df -- data frame to be used for modeling
        target -- str title of column to be used as target
        """

        train, test = train_test_split(df, train_size=0.80, test_size=0.20, random_state=33)

        x_train = train.drop(columns=[target])
        y_train = train[target]

        x_test = test.drop(columns=[target])
        y_test = test[target]

        pipe = make_pipeline(
            OneHotEncoder(handle_unknown='ignore'),
            SimpleImputer(),
            self.model
        )

        pipe.fit(X=x_train, y=y_train)

        try:
            mean = round(np.mean(df[target]))
            mae(y_train, pipe.predict(x_train))
            y_pred = df.copy()
            y_pred[target] = mean
            y_pred = y_pred[target]
            print(f'Baseline MAE: {mae(df[target], y_pred)}\n')
            print(f'Training MAE: {mae(y_train, pipe.predict(x_train))}\n')
            print(f'Test MAE: {mae(y_test, pipe.predict(x_test))}\n')
        except TypeError:
            print(f'Baseline Accuracy:\n{df[target].value_counts(normalize=True)}\n')
            print(f'\nTraining Accuracy: {pipe.score(x_train, y_train)}\n')
            print(f'Test Accuracy: {pipe.score(x_test, y_test)}\n')
