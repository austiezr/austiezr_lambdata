import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


class TransformDF:
    """The base class for all df transformations"""

    @staticmethod
    def date_split(df, date_col):
        df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors='raise')
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_year'] = df[date_col].dt.dayofyear
        return df

    @staticmethod
    def add_to_df(new_list, df):
        new_list = pd.Series(new_list)
        df['new_column'] = new_list
        return df


class MVP:
    """Base class for quickly producing Minimum Viable Product for EDA and Baselines"""
    def __init__(self, model):
        self.model = model

    def fastFirst(self, df, target):

        try:
            mean = round(np.mean(df[target]))
            base = len(df[df[target] == mean])/len(df)
            print(f'Baseline: {base:.2%}\n')
        except TypeError:
            print(f'Baseline:\n{df[target].value_counts(normalize=True)}\n')

        pipe = make_pipeline(
            OneHotEncoder(handle_unknown='ignore'),
            SimpleImputer(),
            self.model
        )

        train, test = train_test_split(df, train_size=0.80, test_size=0.20, random_state=33)

        X_train = train.drop(columns=[target])
        y_train = train[target]

        X_test = test.drop(columns=[target])
        y_test = test[target]

        pipe.fit(X=X_train, y=y_train)

        print(f'\nTraining Score: {pipe.score(X_train, y_train)}\n')
        print(f'Test Score: {pipe.score(X_test, y_test)}\n')
