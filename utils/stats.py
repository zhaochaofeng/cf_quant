'''
    统计学相关功能
'''

import numpy as np
import pandas as pd
import statsmodels.api as sm


def WLS(y, X, intercept=True, weight=1, verbose=True, backend='statsmodels'):
    """ 加权最小二乘法

        y: [array, Series, DataFrame]. 因变量
        X: [array, Series, DataFrame]. 自变量
        intercept: 是否包含截距项
        weight: array_like/float。权重
        verbose: 是否返回残差
        backend: 计算后端，'statsmodels' 或 'numpy'
    """
    if backend == 'numpy':
        return _wls_numpy(y, X, intercept, weight, verbose)
    else:
        return _wls_statsmodels(y, X, intercept, weight, verbose)


def _wls_statsmodels(y, X, intercept=True, weight=1, verbose=True):
    """使用 statsmodels 实现的加权最小二乘法"""
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
    params = result.params  # 参数顺序与 X 特征顺序一致

    if verbose:  # 返回 Bete, alpha, resid
        # 计算预测值
        y_pred = pd.DataFrame(np.dot(X, params), index=y.index)

        # 根据y的类型调整列名
        if isinstance(y, pd.DataFrame):
            y_pred.columns = y.columns
            resid = y - y_pred
        else:
            # y是Series
            resid = y - y_pred.iloc[:, 0]
            resid.name = y.name if hasattr(y, 'name') else None

        if intercept:
            # Beta, alpha, resid
            return params.iloc[1:], params.iloc[0], resid
        else:
            return params, None, resid
    else:  # 仅返回 Beta 参数
        if intercept:
            # 第1个参数为解决项，过滤
            return params.iloc[1:]
        else:
            return params


def _wls_numpy(y, X, intercept=True, weight=1, verbose=True):
    """使用 numpy.linalg.lstsq 实现的加权最小二乘法（性能优化版）"""
    # 转换为 numpy 数组
    y_np = np.asarray(y)
    X_np = np.asarray(X)

    # 确保正确的形状
    if y_np.ndim == 1:
        y_np = y_np.reshape(-1, 1)
    if X_np.ndim == 1:
        X_np = X_np.reshape(-1, 1)

    # 添加截距项
    if intercept:
        X_np = np.column_stack([np.ones(X_np.shape[0]), X_np])

    # 处理权重
    if isinstance(weight, (int, float)) and weight == 1:
        # 等权重
        w_sqrt = 1.0
    else:
        weight_arr = np.asarray(weight)
        if weight_arr.ndim == 1:
            weight_arr = weight_arr.reshape(-1, 1)
        w_sqrt = np.sqrt(weight_arr)

    # 加权最小二乘：对 y 和 X 乘以 sqrt(weight)
    if isinstance(w_sqrt, np.ndarray):
        y_weighted = y_np * w_sqrt
        X_weighted = X_np * w_sqrt
    else:
        y_weighted = y_np
        X_weighted = X_np

    # 使用 numpy.linalg.lstsq 求解
    try:
        beta, residuals, rank, s = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"矩阵奇异或条件数过大，无法求解: {e}")

    if verbose:
        # 计算残差
        y_pred = X_np @ beta
        resid = y_np - y_pred

        # 转换为pandas格式以保持与statsmodels一致
        if isinstance(y, pd.DataFrame):
            # y是DataFrame
            resid_df = pd.DataFrame(resid, index=y.index, columns=y.columns)
            if intercept:
                if beta.shape[0] > 1:
                    # 多特征：beta[1:]是斜率
                    slope = pd.Series(beta[1:].flatten(), index=X.columns if hasattr(X, 'columns') else None)
                else:
                    # 单特征：beta[0]是截距，没有斜率
                    slope = pd.Series([], dtype=float)
                intercept_val = float(beta[0])
                return slope, intercept_val, resid_df
            else:
                # 无截距
                slope = pd.Series(beta.flatten(), index=X.columns if hasattr(X, 'columns') else None)
                return slope, None, resid_df
        elif isinstance(y, pd.Series):
            # y是Series
            resid_df = pd.Series(resid.flatten(), index=y.index)
            if intercept:
                if beta.shape[0] > 1:
                    # 多特征
                    if X.shape[1] == 1:
                        # 单特征X
                        slope = pd.Series([float(beta[1])], index=[X.name] if hasattr(X, 'name') else None)
                    else:
                        # 多特征X
                        slope = pd.Series(beta[1:].flatten(), index=X.columns if hasattr(X, 'columns') else None)
                else:
                    # 单特征，只有截距
                    slope = pd.Series([], dtype=float)
                intercept_val = float(beta[0])
                return slope, intercept_val, resid_df
            else:
                # 无截距
                if X.shape[1] == 1:
                    slope = pd.Series([float(beta[0])], index=[X.name] if hasattr(X, 'name') else None)
                else:
                    slope = pd.Series(beta.flatten(), index=X.columns if hasattr(X, 'columns') else None)
                return slope, None, resid_df
        else:
            # y是numpy数组
            # 转换为pandas Series/DataFrame以保持格式一致
            if y_np.shape[1] == 1:
                # 单因变量 - 匹配statsmodels行为：返回DataFrame
                y_index = pd.RangeIndex(len(y))
                resid_df = pd.DataFrame(resid, index=y_index, columns=[0])

                if intercept:
                    if beta.shape[0] > 1:
                        # 多特征
                        if X_np.shape[1] == 1:
                            # 单特征X
                            slope = pd.Series([float(beta[1])])
                        else:
                            # 多特征X
                            slope = pd.Series(beta[1:].flatten())
                    else:
                        # 单特征，只有截距
                        slope = pd.Series([], dtype=float)
                    intercept_val = float(beta[0])
                    return slope, intercept_val, resid_df
                else:
                    # 无截距
                    if X_np.shape[1] == 1:
                        slope = pd.Series([float(beta[0])])
                    else:
                        slope = pd.Series(beta.flatten())
                    return slope, None, resid_df
            else:
                # 多因变量（不常见）
                y_index = pd.RangeIndex(len(y))
                resid_df = pd.DataFrame(resid, index=y_index)

                if intercept:
                    if beta.shape[0] > 1:
                        slope = pd.DataFrame(beta[1:].T)
                    else:
                        slope = pd.DataFrame([], dtype=float)
                    intercept_val = pd.Series(beta[0].flatten())
                    return slope, intercept_val, resid_df
                else:
                    slope = pd.DataFrame(beta.T)
                    return slope, None, resid_df
    else:
        # 非verbose模式，只返回斜率
        if intercept:
            if beta.shape[0] > 1:
                slope = beta[1:]
                # 转换为pandas Series
                if isinstance(X, pd.DataFrame):
                    return pd.Series(slope.flatten(), index=X.columns)
                elif isinstance(X, pd.Series):
                    return pd.Series([float(slope[0])], index=[X.name])
                else:
                    # numpy数组输入
                    if X_np.shape[1] == 1:
                        return pd.Series([float(slope[0])])
                    else:
                        return pd.Series(slope.flatten())
            else:
                # 只有截距，没有斜率
                return pd.Series([], dtype=float)
        else:
            # 无截距
            if isinstance(X, pd.DataFrame):
                return pd.Series(beta.flatten(), index=X.columns)
            elif isinstance(X, pd.Series):
                return pd.Series([float(beta[0])], index=[X.name])
            else:
                # numpy数组输入
                if X_np.shape[1] == 1:
                    return pd.Series([float(beta[0])])
                else:
                    return pd.Series(beta.flatten())


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





