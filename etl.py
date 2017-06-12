# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from scipy.stats import skew
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

LOCAL_CONFIG = {

}


class ETL:
    def __init__(self):
        pass

    @staticmethod
    def get_data():
        df_train = pd.read_csv('data/train.csv')
        df_test = pd.read_csv('data/test.csv')
        return df_train, df_test

    def overview_missing(self):
        df_train, df_test = self.get_data()
        total = df_train.isnull().sum().sort_values(ascending=False)
        percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        return missing_data

    def etl_data(self):
        df_train, df_test = self.get_data()
        all_data = pd.concat((df_train.loc[:, 'MSSubClass':'SaleCondition'],
                              df_test.loc[:, 'MSSubClass':'SaleCondition']))

        # log transform the target:
        df_train["SalePrice"] = np.log1p(df_train["SalePrice"])

        # log transform skewed numeric features:
        numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

        skewed_feats = df_train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
        skewed_feats = skewed_feats[skewed_feats > 0.75]
        skewed_feats = skewed_feats.index

        all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
        all_data = pd.get_dummies(all_data)
        # filling NA's with the mean of the column:
        all_data = all_data.fillna(all_data.mean())
        print(all_data)

        # creating matrices for sklearn:
        X_train = all_data[:df_train.shape[0]]
        X_test = all_data[df_train.shape[0]:]
        y = df_train.SalePrice

if __name__ == '__main__':
    etl = ETL()
    etl.get_data()


    # res = etl.overview_missing()
    # print(res)

