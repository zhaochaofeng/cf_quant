'''
    统计学相关功能
'''

import numpy as np
import pandas as pd
import statsmodels.api as sm


def WLS(y, X, intercept=True, weight=1, verbose=True):
    """ 加权最小二乘法

        y: 因变量
        X: 自变量
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





