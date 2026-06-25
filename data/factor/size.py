'''
    规模因子
'''

import numpy as np
import pandas as pd
from utils import WLS
from utils import winsorize, standardize, neutralize


def LNCAP(df) -> pd.Series:
    """ 市值(大小盘) """

    circ_mv = df['$circ_mv']
    lncap = np.log(circ_mv)
    lncap.name = 'LNCAP'
    lncap.dropna(inplace=True)
    lncap.sort_index(inplace=True)

    return lncap


def MIDCAP(df):
    """ 中等市值 """

    circ_mv = df['$circ_mv']
    lncap = np.log(circ_mv).dropna()

    y = lncap ** 3
    midcap = neutralize(y, lncap, intercept=True, level='datetime')
    midcap = midcap.to_frame()
    midcap = winsorize(midcap, method='median', lower=0.01, upper=0.99, level='datetime')
    midcap = standardize(midcap, level='datetime')
    midcap.columns = ['MIDCAP']
    midcap.sort_index(inplace=True)

    return midcap



