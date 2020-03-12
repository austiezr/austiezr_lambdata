import unittest
from austiezr_lambdata.austiezr_lambdata import TransformDF, MVP
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.datasets import make_moons
from sklearn.datasets import make_regression


class TransformTestCase(unittest.TestCase):

    def test_date(self):
        df = pd.DataFrame(data=['06-22-2006', '01-01-2001'], columns=['date'])
        df2 = pd.DataFrame(data=['05-30-2005', '20-02-1994'], columns=['date'])

        TransformDF.date_split(df, date_col='date')
        self.assertEqual(6, len(df.columns))
        TransformDF.date_split(df2, date_col='date')
        self.assertEqual(6, len(df2.columns))
        self.assertEqual(len(df.columns), len(df2.columns))
        print(df)
        print(df2)

    def test_add(self):
        df = pd.DataFrame(data=['06-22-2006', '01-01-2001'], columns=['date'])
        df2 = pd.DataFrame(data=['05-30-2005', '20-02-1994'], columns=['date'])
        test_list = [1, 2]
        test_list2 = [3, 4]

        TransformDF.add_to_df(test_list, df=df)
        self.assertCountEqual(test_list, df['new_column'])
        TransformDF.add_to_df(test_list2, df=df2)
        self.assertCountEqual(test_list2, df2['new_column'])
        print(df)
        print(df2)


class MVPTestCase(unittest.TestCase):

    @staticmethod
    def test_ff_class():
        x, y = make_moons(500, noise=.2)
        df = pd.DataFrame(dict(x=x[:, 0], y=x[:, 1], label=y))
        df['label'] = df['label'].map({0: False, 1: True})
        mvp = MVP(LogisticRegressionCV())
        mvp.fast_first(df=df, target='label')

    @staticmethod
    def test_ff_reg():
        x, y = make_regression(500, n_features=2, noise=.2, n_targets=1)
        df = pd.DataFrame(dict(x=x[:, 0], y=x[:, 1]))
        df = TransformDF.add_to_df(y, df)
        mvp = MVP(LinearRegression())
        mvp.fast_first(df=df, target='new_column')


if __name__ == '__main__':
    unittest.main()
