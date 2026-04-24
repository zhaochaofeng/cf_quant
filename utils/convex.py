"""
    凸优化函数
    提供带线性约束的加权最小二乘估计
"""
import numpy as np
import pandas as pd
import cvxpy as cp


def constrained_wls(y, X, intercept=True, weight=1, constraints=None, verbose=True):
    """带线性约束的加权最小二乘估计

    目标：minimize || sqrt(weight) ⊙ (y - X@beta) ||²
    约束：A @ beta = b

    Args:
        y: array/Series/DataFrame, 因变量
        X: array/Series/DataFrame, 自变量
        intercept: bool, 是否包含截距项
        weight: array_like, 权重 (与回归权重概念一致，无需归一化)
        constraints: list[dict], 每个 dict 包含:
            - 'vars': list[str], 约束涉及的变量名
            - 'weights': list[float], 对应的权重
            - 'rhs': float, 约束值 (默认 0)
            例如: {'vars': ['ind1','ind2'], 'weights': [0.6, 0.4], 'rhs': 0}
        verbose: bool, 是否返回残差

    Returns:
        (params, intercept_val, resid) 当 verbose=True
        (params,) 当 verbose=False
        - params: pd.Series, 因子收益率 (当 intercept=True 时不包含截距项)
        - intercept_val: float, 截距值
        - resid: pd.Series/DataFrame, 残差
    """
    # 统一为 DataFrame
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        y = pd.DataFrame(y)
    if not isinstance(X, (pd.Series, pd.DataFrame)):
        X = pd.DataFrame(X)

    y_np = np.asarray(y, dtype=float)
    X_np = np.asarray(X, dtype=float)
    N = X_np.shape[0]

    # 权重处理
    if isinstance(weight, (int, float)) and weight == 1:
        w_sqrt = np.ones(N)
    else:
        w_sqrt = np.sqrt(np.asarray(weight, dtype=float).flatten())

    # 添加截距项
    X_aug = X_np
    col_names = list(X.columns)
    if intercept:
        X_aug = np.column_stack([np.ones(N), X_np])
        col_names = ['const'] + col_names

    K = X_aug.shape[1]
    beta = cp.Variable(K)

    # 目标函数：|| sqrt(w) ⊙ (y - X@beta) ||²
    objective = cp.Minimize(cp.sum_squares(cp.multiply(w_sqrt, y_np.flatten() - X_aug @ beta)))

    # 构建约束
    cvxpy_constraints = []
    if constraints:
        A = np.zeros((len(constraints), K))
        b = np.zeros(len(constraints))
        for i, con in enumerate(constraints):
            var_names = con.get('vars', [])
            weights = con.get('weights', [1.0] * len(var_names))
            rhs = con.get('rhs', 0.0)
            for var_name, w in zip(var_names, weights):
                if var_name in col_names:
                    j = col_names.index(var_name)
                    A[i, j] = w
            b[i] = rhs
        cvxpy_constraints = [A @ beta == b]

    prob = cp.Problem(objective, cvxpy_constraints)
    prob.solve()

    if beta.value is None:
        raise ValueError(f"凸优化求解失败: {prob.status}")

    beta_vals = beta.value
    resid_np = y_np.flatten() - X_aug @ beta_vals

    if intercept:
        intercept_val = beta_vals[0]
        params = pd.Series(beta_vals[1:], index=X.columns)
    else:
        intercept_val = None
        params = pd.Series(beta_vals, index=X.columns)

    if verbose:
        if isinstance(y, pd.DataFrame):
            resid = pd.DataFrame(resid_np.reshape(-1, 1), index=y.index, columns=y.columns)
        else:
            resid = pd.Series(resid_np, index=y.index, name=y.name)
        return params, intercept_val, resid
    else:
        return params
