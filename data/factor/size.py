'''
    规模因子
'''

import numpy as np
from utils import WLS
from utils import winsorize, standardize


def LNCAP(df):
    """ 市值 """

    df = df.sort_index()
    circ_mv = df['$circ_mv']
    lncap = np.log(circ_mv)
    lncap.name = 'LNCAP'
    result_df = lncap.to_frame()

    return result_df


def MIDCAP(df):
    """ 中等市值 """

    lncap = LNCAP(df)['LNCAP']
    x = lncap.dropna().values
    y = x ** 3
    beta, alpha, resid = WLS(y, x, intercept=True, weight=1, verbose=True)
    # 这里使用的 lncap 以保证原始数据维度
    midcap = lncap ** 3 - (alpha + beta[0] * lncap)
    midcap = winsorize(midcap, method='quantile', lower=0.01, upper=0.99)
    midcap = standardize(midcap)
    midcap = midcap.to_frame()
    midcap.columns = ['MIDCAP']

    return midcap




