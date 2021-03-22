# -*- coding: utf-8 -*-
import io
import os
import random
import glob
import pickle
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from sklearn import utils
from sklearn import metrics
from scipy import stats

class ScoringService(object):
    # 訓練期間終了日
    TRAIN_END = "2018-12-31"
    # 評価期間開始日
    VAL_START = "2019-03-27" # "2019-02-01"
    # 評価期間終了日
    VAL_END = "2019-12-01"
    # テスト期間開始日
    TEST_START = "2020-01-01"
    # 目的変数
    TARGET_LABELS = ["label_high_20", "label_low_20"]
    # compute cv?
    IS_VAL = False
    
    for i in [5, 10]:
        TARGET_LABELS += [f'label_high_{i}', f'label_low_{i}']

    # model names
    MODEL_NAMES = ['mlp', ]

    # データをこの変数に読み込む
    dfs = None
    # モデルをこの変数に読み込む
    models = None
    # 対象の銘柄コードをこの変数に読み込む
    codes = None

    @classmethod
    def get_inputs(cls, dataset_dir):
        """
        Args:
            dataset_dir (str)  : path to dataset directory
        Returns:
            dict[str]: path to dataset files
        """
        if len(glob.glob(f"{dataset_dir}/*.gz")) > 0:
            inputs = {
                "stock_list": f"{dataset_dir}/stock_list.csv.gz",
                "stock_price": f"{dataset_dir}/stock_price.csv.gz",
                "stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
                # "stock_fin_price": f"{dataset_dir}/stock_fin_price.csv.gz",
                "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
            }
        else: # already de-frozen?
            inputs = {
                "stock_list": f"{dataset_dir}/stock_list.csv",
                "stock_price": f"{dataset_dir}/stock_price.csv",
                "stock_fin": f"{dataset_dir}/stock_fin.csv",
                # "stock_fin_price": f"{dataset_dir}/stock_fin_price.csv",
                "stock_labels": f"{dataset_dir}/stock_labels.csv",
            }
        return inputs

    @classmethod
    def get_dataset(cls, inputs):
        """
        Args:
            inputs (list[str]): path to dataset files
        Returns:
            dict[pd.DataFrame]: loaded data
        """
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            cls.dfs[k] = pd.read_csv(v)
            # DataFrameのindexを設定します。
            if k == "stock_price":
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "EndOfDayQuote Date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
            elif k in ["stock_fin", "stock_fin_price", "stock_labels"]:
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "base_date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
        return cls.dfs

    @classmethod
    def get_codes(cls, dfs):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
        Returns:
            array: list of stock codes
        """
        stock_list = dfs["stock_list"].copy()
        # 予測対象の銘柄コードを取得
        cls.codes = stock_list[stock_list["prediction_target"] == True][
            "Local Code"
        ].values
        return cls.codes

    @classmethod
    def get_features_and_label(cls, dfs, codes, feature, label):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
            codes  (array) : target codes
            feature (pd.DataFrame): features
            label (str) : label column name
        Returns:
            train_X (pd.DataFrame): training data
            train_y (pd.DataFrame): label for train_X
            val_X (pd.DataFrame): validation data
            val_y (pd.DataFrame): label for val_X
            test_X (pd.DataFrame): test data
            test_y (pd.DataFrame): label for test_X
        """
        # 分割データ用の変数を定義
        trains_X, vals_X, tests_X = [], [], []
        trains_y, vals_y, tests_y = [], [], []

        # 銘柄コード毎に特徴量を作成
        for code in tqdm(codes):
            # 特徴量取得
            feats = feature[feature["code"] == code]

            # stock_labelデータを読み込み
            stock_labels = dfs["stock_labels"]
            # 特定の銘柄コードのデータに絞る
            stock_labels = stock_labels[stock_labels["Local Code"] == code]

            # 特定の目的変数に絞る
            labels = stock_labels[label].copy()
            # nanを削除
            labels.dropna(inplace=True)

            if feats.shape[0] > 0 and labels.shape[0] > 0:
                # 特徴量と目的変数のインデックスを合わせる
                labels = labels.loc[labels.index.isin(feats.index)]
                feats = feats.loc[feats.index.isin(labels.index)]
                labels.index = feats.index

                # データを分割
                _train_X = feats[: cls.TRAIN_END]
                _val_X = feats[cls.VAL_START : cls.VAL_END]
                _test_X = feats[cls.TEST_START :]

                _train_y = labels[: cls.TRAIN_END]
                _val_y = labels[cls.VAL_START : cls.VAL_END]
                _test_y = labels[cls.TEST_START :]

                # データを配列に格納 (後ほど結合するため)
                trains_X.append(_train_X)
                vals_X.append(_val_X)
                tests_X.append(_test_X)

                trains_y.append(_train_y)
                vals_y.append(_val_y)
                tests_y.append(_test_y)

        # 銘柄毎に作成した説明変数データを結合します。
        train_X = pd.concat(trains_X)
        val_X = pd.concat(vals_X)
        test_X = pd.concat(tests_X)
        # 銘柄毎に作成した目的変数データを結合します。
        train_y = pd.concat(trains_y)
        val_y = pd.concat(vals_y)
        test_y = pd.concat(tests_y)

        return train_X, train_y, val_X, val_y, test_X, test_y
    
    @classmethod
    def fin_fe(cls, fin_data):
        # # obj to int
        f = 'Result_FinancialStatement AccountingStandard'
        mapper = {
            'NonConsolidated': 0,
            'ConsolidatedJP': 1,
            'ConsolidatedUS': 2,
            'ConsolidatedIFRS': 3
        }
        fin_data[f] = fin_data[f].map(mapper).fillna(0)
        fin_data[f] = fin_data[f].astype(int)

        f = 'Result_FinancialStatement ReportType'
        mapper = {
            'Q1': 1.0,
            'Q2': 2.0,
            'Q3': 3.0,
            'Annual': 4.0
        }
        fin_data[f] = fin_data[f].map(mapper).fillna(4.0)

        f = 'Result_FinancialStatement CompanyType'
        mapper = {
            'GB': 0,
            'BK': 1,
            'SE': 2,
            'IN': 3
        }
        fin_data[f] = fin_data[f].map(mapper).fillna(0)
        fin_data[f] = fin_data[f].astype(int)

        # zaimu
        fin_data["Result_FinancialStatement NetSales"] = fin_data["Result_FinancialStatement NetSales"] / fin_data["Result_FinancialStatement ReportType"]
        fin_data["profit_margin"] = fin_data["Result_FinancialStatement NetIncome"]/ (fin_data["Result_FinancialStatement NetSales"]+1)
        fin_data["profit_margin"][fin_data["Result_FinancialStatement CashFlowsFromOperatingActivities"] == 0] = np.nan
        fin_data["equity_ratio"] = fin_data["Result_FinancialStatement NetAssets"]/ (fin_data["Result_FinancialStatement TotalAssets"]+1)
        
        # only 1 year column
        drops = [f for f in fin_data.columns.values.tolist() if ('Year' in f) & (f != 'Result_FinancialStatement FiscalYear')]
        drops += [
            'Forecast_FinancialStatement AccountingStandard', 'Forecast_FinancialStatement FiscalPeriodEnd',
            'Forecast_FinancialStatement ReportType', 'Forecast_FinancialStatement ModifyDate',
            'Forecast_FinancialStatement CompanyType', 'Forecast_Dividend RecordDate',
            'Forecast_Dividend FiscalPeriodEnd', 'Forecast_Dividend ReportType', 
            'Forecast_Dividend ModifyDate', 'Result_Dividend DividendPayableDate',
            'Result_FinancialStatement ReportType'
            ]
        fin_data = fin_data[[f for f in fin_data.columns.values.tolist() if f not in drops]]
        
        # fillna
        fin_data.fillna(method='ffill', inplace=True)
        fin_data.fillna(method='bfill', inplace=True)
        return fin_data

    @classmethod
    def price_fe(cls, feats):
        # technical indicators
        def rsi(close: pd.DataFrame, period: int = 14) -> pd.Series:
            # https://gist.github.com/jmoz/1f93b264650376131ed65875782df386
            """See source https://github.com/peerchemist/finta
            and fix https://www.tradingview.com/wiki/Talk:Relative_Strength_Index_(RSI)
            Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.
            RSI oscillates between zero and 100. Traditionally, and according to Wilder, RSI is considered overbought when above 70 and oversold when below 30.
            Signals can also be generated by looking for divergences, failure swings and centerline crossovers.
            RSI can also be used to identify the general trend."""

            delta = close.diff()

            up, down = delta.copy(), delta.copy()
            up[up < 0] = 0
            down[down > 0] = 0

            _gain = up.ewm(com=(period - 1), min_periods=period).mean()
            _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

            RS = _gain / _loss
            return pd.Series(100 - (100 / (1 + RS)))

        def macd(close : pd.DataFrame, span1=12, span2=26):
            # https://www.learnpythonwithrune.org/pandas-calculate-the-moving-average-convergence-divergence-macd-for-a-stock/
            exp1 = close.ewm(span=span1, adjust=False).mean()
            exp2 = close.ewm(span=span2, adjust=False).mean()
            macd = exp1 - exp2
            exp3 = macd.ewm(span=9, adjust=False).mean()

            return macd, exp3

        # datetime features
        feats['day'] = pd.to_datetime(feats['EndOfDayQuote Date']).dt.day
        feats['dayofweek'] = pd.to_datetime(feats['EndOfDayQuote Date']).dt.dayofweek

        # minmax
        feats['price_min2max'] = feats['EndOfDayQuote Low'] / (feats['EndOfDayQuote High'] + 1)
        
        # open close
        feats['price_open2close'] = feats['EndOfDayQuote Open'] / (feats['EndOfDayQuote Close'] + 1)
        
        # fのX営業日...
        features = ["EndOfDayQuote ExchangeOfficialClose", 'EndOfDayQuote Volume', ]
        new_feats = ['price_min2max', 'price_open2close', 
            'day', 'dayofweek', ]
        for i, f in enumerate(features):
            for x in [5, 10, 20, 40, ]:
                # return 
                feats[f"{f}_return_{x}days"] = feats[
                    f
                ].pct_change(x)

                # volatility
                feats[f"{f}_volatility_{x}days"] = (
                    np.log1p(feats[f])
                    .diff()
                    .rolling(x)
                    .std()
                )

                # skew
                feats[f"{f}_skew_{x}days"] = (
                    np.log1p(feats[f])
                    .diff()
                    .rolling(x)
                    .skew()
                )

                # kurt
                feats[f"{f}_kurt_{x}days"] = (
                    np.log1p(feats[f])
                    .diff()
                    .rolling(x)
                    .kurt()
                )

                # kairi mean
                feats[f"{f}_MA_gap_{x}days"] = feats[f] / (
                    feats[f].rolling(x).mean()
                )
                
                # kairi max
                feats[f"{f}_MAmax_gap_{x}days"] = feats[f] / (
                    feats[f].rolling(x).max()
                )

                # # kairi min
                # feats[f"{f}_MAmin_gap_{x}days"] = feats[f] / (
                #     feats[f].rolling(x).min()
                # )

                # features to use
                new_feats += [
                    f"{f}_return_{x}days", 
                    f"{f}_volatility_{x}days",
                    f"{f}_skew_{x}days", 
                    f"{f}_kurt_{x}days", 
                    f"{f}_MA_gap_{x}days",
                    f"{f}_MAmax_gap_{x}days",
                    # f"{f}_MAmin_gap_{x}days",
                             ]

        # RSI
        rsi = rsi(feats["EndOfDayQuote ExchangeOfficialClose"], 14)
        feats['RSI'] = rsi.values

        # MACD
        macd, exp3 = macd(feats["EndOfDayQuote ExchangeOfficialClose"], 12, 26)
        feats['MACD'] = macd.values
        feats['MACD_9'] = exp3.values
        feats['MACD_d'] = feats['MACD'] / (feats['MACD_9'] + 0.001)
        
        new_feats += ['RSI', 'MACD', 'MACD_9', 'MACD_d']

        # 元データのカラムを削除
        feats = feats[new_feats + features]
        
        # 欠損値処理
        feats.fillna(method='ffill', inplace=True)
        feats.fillna(method='bfill', inplace=True)
        
        return feats

    @classmethod
    def get_features_for_predict(cls, dfs, code, start_dt="2016-01-01"):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            code (int)  : A local code for a listed company
            start_dt (str): specify date range
        Returns:
            feature DataFrame (pd.DataFrame)
        """
        # stock_finデータを読み込み
        stock_fin = dfs["stock_fin"]

        # 特定の銘柄コードのデータに絞る
        fin_data = stock_fin[stock_fin["Local Code"] == code]

        # fin fe
        fin_data = cls.fin_fe(fin_data)

        # 特徴量の作成には過去60営業日のデータを使用しているため、
        # 予測対象日からバッファ含めて土日を除く過去90日遡った時点から特徴量を生成します
        n = 90
        # 特徴量の生成対象期間を指定
        fin_data = fin_data.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :]
        # fin_dataのnp.float64のデータのみを取得
        fin_feats = fin_data.select_dtypes(include=["float64", 'int'])
#         # 欠損値処理
#         fin_feats = fin_feats.fillna(0)

        # stock_priceデータを読み込む
        price = dfs["stock_price"]
        # 特定の銘柄コードのデータに絞る
        price_data = price[price["Local Code"] == code]
        
        # 終値のみに絞る
#         feats = price_data[["EndOfDayQuote ExchangeOfficialClose"]]
        feats = price_data.copy()
        
        # 特徴量の生成対象期間を指定
        feats = feats.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :].copy()
        
        # price fe
        feats = cls.price_fe(feats)

        # stock_list
        stock_list = dfs['stock_list']
        # 特定の銘柄コードのデータに絞る
        list_data = stock_list[stock_list["Local Code"] == code]

        # merge list
        list_data = list_data[["Local Code", "17 Sector(Code)", "33 Sector(Code)", "Size Code (New Index Series)", "IssuedShareEquityQuote IssuedShare"]]
        list_data.columns = ["Local Code", "sector17", "sector33", "size_group", "share"]
        list_data["size_group"] = list_data["size_group"].replace('-', 0).astype(float)

        # 財務データの特徴量とマーケットデータの特徴量のインデックスを合わせる
        feats = feats.loc[feats.index.isin(fin_feats.index)]
        fin_feats = fin_feats.loc[fin_feats.index.isin(feats.index)]

        # データを結合
        feats = pd.concat([feats, fin_feats], axis=1)
#         feats = pd.concat([feats, fin_feats], axis=1).dropna()

        # zaimu feats
        feats['sector17'] = list_data['sector17'].values[-1]
        feats['sector33'] = list_data['sector33'].values[-1]
        feats['size_group'] = list_data['size_group'].values[-1]
        feats['per_like'] = feats["EndOfDayQuote ExchangeOfficialClose"] / (feats["Result_FinancialStatement NetIncome"] + 1)
        feats['pbr_like'] = feats["EndOfDayQuote ExchangeOfficialClose"] / (feats["Result_FinancialStatement NetAssets"] + 1)
        feats["roe_like"] = feats["pbr_like"] / (feats["per_like"] + 0.0001)
        # feats["market_cap"] = feats["EndOfDayQuote ExchangeOfficialClose"] * list_data["share"]
        # feats["per"] = feats["EndOfDayQuote ExchangeOfficialClose"]/(feats["Result_FinancialStatement NetIncome"]*1000000 / (list_data["share"]+1))
        # feats["per"][feats["Result_FinancialStatement CashFlowsFromOperatingActivities"] == 0] = np.nan
        # feats["pbr"] = feats["EndOfDayQuote ExchangeOfficialClose"]/(feats["Result_FinancialStatement NetAssets"]*1000000 / (list_data["share"]+1))
        # feats["roe"] = feats["pbr"] / feats["per"]

        # drops
        drops = ["EndOfDayQuote ExchangeOfficialClose", "EndOfDayQuote Volume", "Local Code"]
        feats = feats[[f for f in feats.columns.values.tolist() if f not in drops]]

        # 欠損値処理を行います。
        feats = feats.replace([np.inf, -np.inf], np.nan)
        feats.fillna(method='ffill', inplace=True)
        feats.fillna(method='bfill', inplace=True)        

        # 銘柄コードを設定
        feats["code"] = code

        # 生成対象日以降の特徴量に絞る
        feats = feats.loc[pd.Timestamp(start_dt) :]

        return feats

    @classmethod
    def get_features_for_predict(cls, dfs, code, start_dt="2016-01-01"):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            code (int)  : A local code for a listed company
            start_dt (str): specify date range
        Returns:
            feature DataFrame (pd.DataFrame)
        """
        # stock_finデータを読み込み
        stock_fin = dfs["stock_fin"]

        # 特定の銘柄コードのデータに絞る
        fin_data = stock_fin[stock_fin["Local Code"] == code]

        # fin fe
        fin_data = cls.fin_fe(fin_data)

        # 特徴量の作成には過去60営業日のデータを使用しているため、
        # 予測対象日からバッファ含めて土日を除く過去90日遡った時点から特徴量を生成します
        n = 90
        # 特徴量の生成対象期間を指定
        fin_data = fin_data.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :]
        # fin_dataのnp.float64のデータのみを取得
        fin_feats = fin_data.select_dtypes(include=["float64", 'int'])
#         # 欠損値処理
#         fin_feats = fin_feats.fillna(0)

        # stock_priceデータを読み込む
        price = dfs["stock_price"]
        # 特定の銘柄コードのデータに絞る
        price_data = price[price["Local Code"] == code]
        
        # 終値のみに絞る
#         feats = price_data[["EndOfDayQuote ExchangeOfficialClose"]]
        feats = price_data.copy()
        
        # 特徴量の生成対象期間を指定
        feats = feats.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :].copy()
        
        # price fe
        feats = cls.price_fe(feats)

        # stock_list
        stock_list = dfs['stock_list']
        # 特定の銘柄コードのデータに絞る
        list_data = stock_list[stock_list["Local Code"] == code]

        # merge list
        list_data = list_data[["Local Code", "17 Sector(Code)", "33 Sector(Code)", "Size Code (New Index Series)", "IssuedShareEquityQuote IssuedShare"]]
        list_data.columns = ["Local Code", "sector17", "sector33", "size_group", "share"]
        list_data["size_group"] = list_data["size_group"].replace('-', 0).astype(float)

        # 財務データの特徴量とマーケットデータの特徴量のインデックスを合わせる
        feats = feats.loc[feats.index.isin(fin_feats.index)]
        fin_feats = fin_feats.loc[fin_feats.index.isin(feats.index)]

        # データを結合
        feats = pd.concat([feats, fin_feats], axis=1)
#         feats = pd.concat([feats, fin_feats], axis=1).dropna()

        # zaimu feats
        feats['sector17'] = list_data['sector17'].values[-1]
        feats['sector33'] = list_data['sector33'].values[-1]
        feats['size_group'] = list_data['size_group'].values[-1]
        # feats["market_cap"] = feats["EndOfDayQuote ExchangeOfficialClose"] * list_data["share"]
        # feats["per"] = feats["EndOfDayQuote ExchangeOfficialClose"]/(feats["Result_FinancialStatement NetIncome"]*1000000 / (list_data["share"]+1))
        # feats["per"][feats["Result_FinancialStatement CashFlowsFromOperatingActivities"] == 0] = np.nan
        # feats["pbr"] = feats["EndOfDayQuote ExchangeOfficialClose"]/(feats["Result_FinancialStatement NetAssets"]*1000000 / (list_data["share"]+1))
        # feats["roe"] = feats["pbr"] / feats["per"]

        # drops
        drops = ["EndOfDayQuote ExchangeOfficialClose", "EndOfDayQuote Volume", "Local Code"]
        feats = feats[[f for f in feats.columns.values.tolist() if f not in drops]]

        # 欠損値処理を行います。
        feats = feats.replace([np.inf, -np.inf], np.nan)
        feats.fillna(method='ffill', inplace=True)
        feats.fillna(method='bfill', inplace=True)        

        # 銘柄コードを設定
        feats["code"] = code

        # 生成対象日以降の特徴量に絞る
        feats = feats.loc[pd.Timestamp(start_dt) :]

        return feats

    @classmethod
    def get_feature_columns(cls, dfs, train_X, column_group="fundamental+technical"):
        # 特徴量グループを定義
        # ファンダメンタル
        fundamental_cols = dfs["stock_fin"].select_dtypes("float64").columns
        fundamental_cols = fundamental_cols[
            fundamental_cols != "Result_Dividend DividendPayableDate"
        ]
        fundamental_cols = fundamental_cols[fundamental_cols != "Local Code"]
        # 価格変化率
        returns_cols = [x for x in train_X.columns if "return" in x]
        # テクニカルa
        technical_cols = [
            x for x in train_X.columns if (x not in fundamental_cols) and (x != "code")
        ]
        columns = {
            "fundamental_only": fundamental_cols,
            "return_only": returns_cols,
            "technical_only": technical_cols,
#             "fundamental+technical": list(fundamental_cols) + list(technical_cols),
            "fundamental+technical": [f for f in train_X.columns.values.tolist() if f not in ['code', 
                                        "Result_Dividend DividendPayableDate", "Local Code"]], # all features
        }
        return columns[column_group]

    @classmethod
    def compute_cv(cls, ypred, ytrue):
        # cv
        cvs = pd.DataFrame()
        cvs['metric'] = np.array(['rmse', 'mae', 'corr', 'spearman_corr'])
        cvs['value'] = 0

        # RMSE
        cvs.loc[cvs['metric'] == 'rmse', 'value'] = np.sqrt(metrics.mean_squared_error(ypred, ytrue))
        # MAE
        cvs.loc[cvs['metric'] == 'mae', 'value'] = metrics.mean_absolute_error(ypred, ytrue)
        # 相関係数
        cvs.loc[cvs['metric'] == 'corr', 'value'] = np.corrcoef(ytrue, ypred)[0, 1]
        # 順位相関
        cvs.loc[cvs['metric'] == 'spearman_corr', 'value'] = stats.spearmanr(ytrue, ypred)[0]

        return cvs

    @classmethod
    def get_params(cls):
        """
        dispatch model hyperparameters
        """
        params = {
            'input_dropout': 0.0,
            'embedding_out_dim': 4,
            'hidden_layers': 3,
            'hidden_units': 256,
            'hidden_activation': 'relu', 
            'hidden_dropout': 0.2,
            'gauss_noise': 0.01,
            'lr': 1e-3,
            'batch_size': 1024,
            'epochs': 100
        }
        return params
    
    @classmethod
    def fit_mlp(cls, train_X: pd.DataFrame, train_y: np.array, val_X: pd.DataFrame, val_y: np.array):
        """
        fit a MLP
        
        """
        print('TRAIN:')
        print(train_X.info())

        print('VALID:')
        print(val_X.info())

        # fillna
        train_X.fillna(method='ffill', inplace=True)
        train_X.fillna(method='bfill', inplace=True)
        train_X.fillna(train_X.median(), inplace=True)
        val_X.fillna(method='ffill', inplace=True)
        val_X.fillna(method='bfill', inplace=True)
        val_X.fillna(val_X.median(), inplace=True)        

        # set seed to reproduce the result
        def seed_everything(seed : int):    
            random.seed(seed)
            np.random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            tf.random.set_seed(seed)

        seed_everything(326) 

        # hyperparameters
        params = cls.get_params()
        
        # embedding for categorical features 
        categoricals = [c for c in train_X.columns.values.tolist() if train_X[c].dtype == 'int']
        features = [c for c in train_X.columns.values.tolist() if c not in categoricals]
        if len(categoricals) > 0:
            inputs = []
            embeddings = []
            embedding_out_dim = params['embedding_out_dim']
            for i in categoricals:
                input_ = tf.keras.layers.Input(shape=(1,))
                maxv = np.max(np.array([train_X[i].abs().max(), val_X[i].abs().max()]))
                embedding = tf.keras.layers.Embedding(int(maxv + 1), 
                    embedding_out_dim, input_length=1)(input_)
                embedding = tf.keras.layers.Reshape(target_shape=(embedding_out_dim,))(embedding)
                inputs.append(input_)
                embeddings.append(embedding)
            input_numeric = tf.keras.layers.Input(shape=(len(features),))
            embedding_numeric = tf.keras.layers.Dense(params['hidden_units'], 
                activation=params['hidden_activation'])(input_numeric)
            inputs.append(input_numeric)
            embeddings.append(embedding_numeric)
            x = tf.keras.layers.Concatenate()(embeddings)
            x = tf.keras.layers.BatchNormalization()(x)

        else:
            # NN model architecture        
            inputs = tf.keras.layers.Input(shape=(train_X.shape[1], ))
            x = tf.keras.layers.BatchNormalization()(inputs)
            x = tf.keras.layers.Dense(params['hidden_units'], 
                activation=params['hidden_activation'])(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.GaussianNoise(params['gauss_noise'])(x)
            x = tf.keras.layers.Dropout(params['hidden_dropout'])(x)
        
        # more layers
        for i in np.arange(params['hidden_layers'] - 1):
            x = tf.keras.layers.Dense(params['hidden_units'] // (2 * (i+1)), 
                activation=params['hidden_activation'])(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.GaussianNoise(params['gauss_noise'])(x)
            x = tf.keras.layers.Dropout(params['hidden_dropout'])(x)

        # output
        outs = tf.keras.layers.Dense(1, activation="linear", name="out")(x)
        loss = "mae" # mean absolute error as a loss
        model = tf.keras.models.Model(inputs=inputs, outputs=outs) 

        # compile
        opt = tf.keras.optimizers.Adam(lr=params['lr'], beta_1=0.9, beta_2=0.999, decay=params['lr']/100)
        model.compile(loss=loss, optimizer=opt, metrics=[])

        # callbacks
        early_stop = tf.keras.callbacks.EarlyStopping(patience=16, min_delta=params['lr'], 
                                                      restore_best_weights=True, monitor='val_loss')
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, 
                                                           verbose=1, epsilon=params['lr'], mode='min')

        # fit
        history = model.fit(train_X, train_y, callbacks=[early_stop, lr_schedule],
                    epochs=params['epochs'], batch_size=params['batch_size'],
                    validation_data=(val_X, val_y))

        # predict
        pred_y = model.predict(val_X).ravel()

        return model, pred_y
    
    @classmethod
    def create_model(cls, dfs, codes, label, model_name='xgb1'):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            codes (list[int]): A local code for a listed company
            label (str): prediction target label
        Returns:
            RandomForestRegressor
        """
        # 特徴量を取得
        buff = []
        for code in codes:
            buff.append(cls.get_features_for_predict(cls.dfs, code))
        feature = pd.concat(buff)
        # 特徴量と目的変数を一致させて、データを分割
        train_X, train_y, val_X, val_y, test_X, test_y = cls.get_features_and_label(
            dfs, codes, feature, label
        )
        # 特徴量カラムを指定
        feature_columns = cls.get_feature_columns(dfs, train_X)

        # params
        params = cls.get_params()

        # model fitting
        model, pred_y = cls.fit_mlp(train_X[feature_columns], train_y,
            val_X[feature_columns], val_y)

        # cv scores
        cvs = cls.compute_cv(pred_y, val_y)

        return model, cvs

    @classmethod
    def save_model(cls, model, label, model_name, model_path="../model"):
        """
        Args:
            model (RandomForestRegressor): trained model
            label (str): prediction target label
            model_path (str): path to save model
        Returns:
            -
        """
        # tag::save_model_partial[]
        # モデル保存先ディレクトリを作成
        os.makedirs(pathlib.Path(model_path), exist_ok=True)
        # with open(os.path.join(model_path, f"my_model_{label}.pkl"), "wb") as f:
        #     # モデルをpickle形式で保存
        #     pickle.dump(model, f)
        # end::save_model_partial[]
        model.save(pathlib.Path(f'{model_path}/{model_name}_{label}.h5'))

    @classmethod
    def get_model(cls, model_path="../model", labels=None, model_names=None):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            labels (arrayt): list of prediction target labels

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        if cls.models is None:
            cls.models = {}
        if labels is None:
            labels = cls.TARGET_LABELS
        if model_names is None:
            model_names = cls.MODEL_NAMES
        for model_name in model_names:
            for label in labels:
                # model path
                m = os.path.join(model_path, f"{model_name}_{label}.h5")

                # load model
                tf.keras.backend.clear_session()
                model = tf.keras.models.load_model(pathlib.Path(m))
                # m = os.path.join(model_path, f"my_model_{label}.pkl")
                # with open(m, "rb") as f:
                #     # pickle形式で保存されているモデルを読み込み
                #     cls.models[label] = pickle.load(f)
                cls.models[f'{model_name}_{label}'] = model

        return True

    @classmethod
    def save_score(cls, cvs, model_path='../model'):
        """
        save validation score records
        """
        os.makedirs(model_path, exist_ok=True)
        # with open(os.path.join(model_path, f"my_model_{label}.pkl"), "wb") as f:
        #     # モデルをpickle形式で保存
        #     pickle.dump(model, f)
        # end::save_model_partial[]
        cvs.to_csv(pathlib.Path(f'{model_path}/cross_validation_scores.csv'), index=False)

    @classmethod
    def train_and_save_model(
        cls, inputs, labels=None, model_names=None, codes=None, model_path="../model"
    ):
        """Predict method

        Args:
            inputs (str)   : paths to the dataset files
            labels (array) : labels which is used in prediction model
            codes  (array) : target codes
            model_path (str): Path to the trained model directory.
        Returns:
            Dict[pd.DataFrame]: Inference for the given input.
        """
        
        # init
        counts = 0

        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS
        if model_names is None:
            model_names = cls.MODEL_NAMES
        for model_name in model_names:
            for label in labels:

                # get model
                model, cvs = cls.create_model(cls.dfs, codes=codes, label=label, model_name=model_name)
                
                # assign
                cvs = cvs.rename(columns={'value': f'{model_name}_{label}'})
                if counts == 0:
                    cv_df = cvs.copy()
                else:
                    cv_df = cv_df.merge(cvs, how='left', on='metric')
                counts += 1

                # save model
                cls.save_model(model, label, model_name, model_path=model_path)
        
        # save feature importance and cv scores
        cls.save_score(cv_df, model_path=model_path)

    @classmethod
    def predict(cls, inputs, labels=None, model_names=None, codes=None, 
                model_path="../model", start_dt=TEST_START):
        """Predict method

        Args:
            inputs (dict[str]): paths to the dataset files
            labels (list[str]): target label names
            codes (list[int]): traget codes
            start_dt (str): specify date range
        Returns:
            str: Inference for the given input.
        """

        # データ読み込み
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)

        # 予測対象の銘柄コードと目的変数を設定
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS
        if model_names is None:
            model_names = cls.MODEL_NAMES

        # 特徴量を作成
        buff = []
        for code in codes:
            buff.append(cls.get_features_for_predict(cls.dfs, code, start_dt))
        feats = pd.concat(buff)

        # 結果を以下のcsv形式で出力する
        # １列目:datetimeとcodeをつなげたもの(Ex 2016-05-09-1301)
        # ２列目:label_high_20　終値→最高値への変化率
        # ３列目:label_low_20　終値→最安値への変化率
        # headerはなし、B列C列はfloat64

        # 日付と銘柄コードに絞り込み
        df = feats.loc[:, ["code"]].copy()
        # codeを出力形式の１列目と一致させる
        df.loc[:, "code"] = df.index.strftime("%Y-%m-%d-") + df.loc[:, "code"].astype(
            str
        )

        # 出力対象列を定義
        output_columns = ["code"]

        # 特徴量カラムを指定
        feature_columns = cls.get_feature_columns(cls.dfs, feats)
        print('{:,} features: {}'.format(len(feature_columns), feature_columns))
        
        # # get model
        # cls.get_model(model_path=model_path, labels=labels, model_names=model_names)

        # 目的変数毎に予測
        for label in ["label_high_20", "label_low_20"]:
            df[label] = 0
        
        num_models = len(model_names) * len(labels) // 2 
        sum_weights = 0
        for label in labels:
            weight = int(label.split('_')[-1])
            if 'high' in label:
                label_ = "label_high_20"
            elif 'low' in label:
                label_ = "label_low_20"
            
            for model_name in model_names:
                print(f'{model_name}_{label}')

                # 予測実施
#                 assert label in df.columns.values.tolist()
#                 assert f'{model_name}_{label}' in list(cls.models.keys())
                
                df[label_] += cls.models[f'{model_name}_{label}'].predict(feats[feature_columns]).ravel() * weight / 25
                
        # 出力対象列に追加
        for label in ["label_high_20", "label_low_20"]:
            output_columns.append(label)

        out = io.StringIO()
        df.to_csv(out, header=False, index=False, columns=output_columns)

        return out.getvalue()