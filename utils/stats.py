'''
    统计学相关功能
'''

import numpy as np
import pandas as pd
import statsmodels.api as sm


def WLS(y, X, intercept=True, weight=1, verbose=True):
    """ 加权最小二乘法

        y: [array, Series, DataFrame]. 因变量
        X: [array, Series, DataFrame]. 自变量
        intercept: 是否包含截距项
        weight: array_like/float。权重
        verbose: 是否返回残差
    """
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        y = pd.DataFrame(y)
    if not isinstance(X, (pd.Series, pd.DataFrame)):
        X = pd.DataFrame(X)

    if intercept:
        cols = X.columns.tolist()
        X['const'] = 1
        X = X[['const'] + cols]   # cost 放在第1列

    model = sm.WLS(y, X, weights=weight)
    result = model.fit()
    params = result.params

    if verbose:
        resid = y - pd.DataFrame(np.dot(X, params), index=y.index,
                                 columns=y.columns)
        if intercept:
            return params.iloc[1:], params.iloc[0], resid
        else:
            return params, None, resid
    else:
        if intercept:
            return params.iloc[1:]
        else:
            return params


def get_exp_weight(window, half_life):
    """ 半衰期权重
    window: 计算窗口
    half_life: 半衰期
    如：
    [0.04956612 0.05693652 0.06540289 0.07512819 0.08629962 0.09913224
        0.11387304 0.13080577 0.15025637 0.17259925]
    """
    exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
    return exp_wt[::-1] / np.sum(exp_wt)





