'''
    一元时间序列分析（Time Series Analysis）
'''

import pandas as pd
from statsmodels.tsa.api import ARIMA
from statsmodels.tsa.api import adfuller
from statsmodels.tsa.api import arma_order_select_ic


class TimeSeriesAnalysis:
    def __init__(self, data: pd.Series):
        self.data = data
        self.order = None
        self.model = None
        self.results = None

    def adf(self, regression='c', autolag='BIC'):
        ''' 平稳性检验 '''
        adf = adfuller(self.data, regression=regression, autolag=autolag)
        return adf

    def order_select(self, max_ar=4, max_ma=2, ic="bic", trend="c"):
        ''' 模型定阶（p, q）'''
        order = arma_order_select_ic(self.data, max_ar=max_ar, max_ma=max_ma, ic=ic, trend=trend)
        self.order = getattr(order, f'{ic}_min_order')
        return self.order

    def arma(self):
        ''' 模型拟合 '''
        if self.order is None:
            raise ValueError("Order is not selected. Please call order_select() first.")
        model = ARIMA(self.data, order=(self.order[0], 0, self.order[1]))
        self.model = model
        results = model.fit()
        self.results = results

    def summary(self):
        ''' 模型摘要 '''
        if self.results is None:
            raise ValueError("Model is not fitted. Please call arma() first.")
        return self.results.summary()

    def forecast(self, steps=1):
        ''' 预测 '''
        if self.results is None:
            raise ValueError("Model is not fitted. Please call arma() first.")
        return self.results.forecast(steps=steps)


