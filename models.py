# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from etl import ETL
from sklearn.model_selection import ShuffleSplit
import math

LOCAL_CONFIG = {
    'METHODS': ['lr', 'lasso', 'dt', 'rf', 'gb', 'adb'],
    'METHOD': 'gb'
}


class Model:
    def __init__(self):
        self.etl = ETL()

    def get_train_data(self):
        X_train, X_test, y = self.etl.get_result()
        y = np.array(list(map(lambda x: math.log(x + 1, math.e), y)))
        return X_train, X_test, y

    @staticmethod
    def normalize(X_train, X_test):
        """
        scaler and nomalizer
        """
        # scaler = preprocessing.StandardScaler().fit(X)
        # X = scaler.transform(X)
        # X_noEx = scaler.transform(X_noEx)
        normalizer = preprocessing.Normalizer().fit(X_train)
        X_train = normalizer.transform(X_train)
        X_test = normalizer.transform(X_test)
        return X_train, X_test

    @staticmethod
    def rmsle(y_pred, y):
        # map1 = list(map(lambda x: math.log(x + 1, math.e), y_pred))
        # map2 = list(map(lambda x: math.log(x + 1, math.e), y))
        res = np.sqrt(metrics.mean_squared_error(y_pred, y))
        return res

    def method_pattern(self, model):
        X_train1, X_test1, y = self.get_train_data()
        X_train1, X_test1 = self.normalize(X_train1, X_test1)

        # training
        kf = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
        rmsles = []
        for train_indices, test_indices in kf.split(X_train1):
            X_train = X_train1[train_indices, :]
            y_train = y[train_indices]
            X_test = X_train1[test_indices, :]
            y_test = y[test_indices]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmsle = self.rmsle(y_pred, y_test)
            rmsles.append(rmsle)
            print('rmsle: ', rmsle)
        print('mean of rmsle: ', np.mean(rmsles))
        return np.mean(rmsles)

    def try_diff_method(self, method):
        if method == 'lr':
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            self.method_pattern(lr)
        elif method == 'lasso':
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=0.1)
            self.method_pattern(lasso)
        elif method == 'dt':
            from sklearn.tree import DecisionTreeRegressor
            dt = DecisionTreeRegressor()
            self.method_pattern(dt)
        elif method == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor()
            self.method_pattern(rf)
        elif method == 'gb':
            from sklearn.ensemble import GradientBoostingRegressor
            gb = GradientBoostingRegressor()
            self.method_pattern(gb)
        elif method == 'adb':
            from sklearn.ensemble import AdaBoostRegressor
            adab = AdaBoostRegressor()
            self.method_pattern(adab)
        else:
            print('do nothing')

    def get_result(self):
        self.try_diff_method(LOCAL_CONFIG['METHOD'])

if __name__ == '__main__':
    model = Model()
    model.get_result()





