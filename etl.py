# -*- coding: UTF-8 -*-

import pandas as pd

LOCAL_CONFIG = {

}


class ETL:
    def __init__(self):
        df_train, df_test = self.get_data()
        self.train = df_train
        self.test = df_test

    @staticmethod
    def get_data():
        df_train = pd.read_csv('data/train.csv')
        df_test = pd.read_csv('data/test.csv')
        return df_train, df_test

    @staticmethod
    def replace_by_dummies(df_X, col):
        dummy1 = pd.get_dummies(df_X.loc[:, col], prefix=col)
        df_X.drop([col], axis=1, inplace=True)
        df_X = pd.concat([df_X, dummy1], axis=1)
        return df_X

    def overview_missing(self):
        total = self.train.isnull().sum().sort_values(ascending=False)
        percent = (self.train.isnull().sum() / self.train.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        return missing_data

    def etl_data(self):
        # 删除 missing value 比例大于15%的 column
        types = self.train.dtypes.to_frame(name='type')
        miss_percent = self.overview_missing()
        cols = pd.merge(miss_percent, types, left_index=True, right_index=True)
        delete_cols = cols.loc[(cols['Percent'] >= 0.15) & (cols['type'] != 'object')].index
        delete_cols = list(delete_cols)
        # print(delete_cols)
        all_data = pd.concat([self.train.loc[:, 'MSSubClass':'SaleCondition'],
                              self.test.loc[:, 'MSSubClass':'SaleCondition']], axis=0, ignore_index=True)
        all_data.drop(delete_cols, axis=1, inplace=True)

        # one hot encode category features
        numeric_feats = self.train.dtypes[self.train.dtypes != "object"].index
        categ_feats = self.train.dtypes[self.train.dtypes == "object"].index
        for col in categ_feats:
            all_data = self.replace_by_dummies(all_data, col)

        # fill missing data
        all_data = all_data.fillna(all_data.mean())

        # creating matrices for sklearn:
        X_train = all_data[:self.train.shape[0]]
        X_test = all_data[self.train.shape[0]:]
        y = self.train.loc[:, 'SalePrice']

        return X_train, X_test, y

    def get_result(self):
        X_train, X_test, y = self.etl_data()

        has_nan = X_train.isnull().any().any()
        print('if has nan: ', has_nan)

        X_train = X_train.values
        X_test = X_test.values
        y = y.values

        return X_train, X_test, y


if __name__ == '__main__':
    etl = ETL()
    X_train, X_test, y = etl.get_result()
    print(X_train)
    print(y)
