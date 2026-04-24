"""横截面回归模块 - 加权最小二乘估计因子收益率"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from utils import WLS
from utils import constrained_wls
from utils import LoggerFactory
from barra.risk_control.config import INDUSTRY_MAPPING
logger = LoggerFactory.get_logger(__name__)


def _build_industry_constraint(X, circ_mv):
    """构建行业因子流通市值加权和为 0 的约束

    以流通市值（circ_mv）为权重，计算各行业的权重占比，
    约束行业因子收益率的加权和为 0。

    Args:
        X: DataFrame, 因子暴露矩阵，行业因子列名为行业名称
        circ_mv: Series, 流通市值

    Returns:
        list[dict], 用于 constrained_wls 的 constraints 参数
    """
    industry_names = list(INDUSTRY_MAPPING.values())
    industry_cols = [c for c in X.columns if c in industry_names]
    if not industry_cols:
        return None

    industry_weights = {}
    for col in industry_cols:
        mask = X[col] == 1
        if mask.any():
            industry_weights[col] = circ_mv[mask].sum()
        else:
            industry_weights[col] = 0.0

    total = sum(industry_weights.values())
    if total <= 0:
        return None
    # 每个行业市值之和权重
    weights_list = [industry_weights[col] / total for col in industry_cols]
    return [{'vars': industry_cols, 'weights': weights_list, 'rhs': 0.0}]


class CrossSectionalRegression:
    """横截面回归估计器

    使用加权最小二乘（WLS）进行横截面回归，估计因子收益率。
    权重为流通市值平方根。
    支持多种回归方法：'wls' (不加截距，全行业因子) 或 'constrained' (加截距，全行业因子 + 市值加权和为0约束)
    """

    def __init__(self):
        self.factor_returns_dict = {}  # {date: Series(factor -> return)}
        self.residuals_dict = {}       # {date: Series(instrument -> residual)}

    def fit_multi_periods(self, returns_df: pd.DataFrame,
                         exposure_df: pd.DataFrame,
                         market_cap_df: pd.DataFrame,
                         method: str = 'constrained') -> pd.DataFrame:
        """多期横截面回归，逐日期使用 WLS 估计因子收益率

        模型：r_t = X_t * b_t + u_t
        权重：sqrt(circ_mv)

        Args:
            returns_df: 收益率，索引 (instrument, datetime)，单列
            exposure_df: 因子暴露，索引 (instrument, datetime)，多列为各因子
            market_cap_df: 市值数据，索引 (instrument, datetime)，含 'circ_mv' 列
            method: 回归方法
                - 'wls': 不加截距，全行业因子 (Barra 标准做法)
                - 'constrained': 加截距，全行业因子 + 行业市值加权和为0约束

        Returns:
            因子收益率矩阵，index=datetime, columns=factors
        """
        logger.info(f'开始多期横截面回归 (method={method})...')
        self.factor_returns_dict.clear()
        self.residuals_dict.clear()

        ret_col = returns_df.columns[0]
        dates = returns_df.index.get_level_values('datetime').unique()

        for date in dates:
            # 提取当期截面数据
            r = returns_df.xs(date, level='datetime')[ret_col]
            X = exposure_df.xs(date, level='datetime')
            mv = market_cap_df.xs(date, level='datetime')['circ_mv']

            # 对齐三组数据的 instrument
            common_idx = r.index.intersection(X.index).intersection(mv.index)
            r, X, mv = r.loc[common_idx], X.loc[common_idx], mv.loc[common_idx]

            # 去除含 NaN 的行
            valid = r.notna() & X.notna().all(axis=1) & mv.notna() & (mv > 0)
            if valid.sum() / len(r) < 0.5:
                err_msg = f'  {date}: 有效数据占比低于50%'
                logger.error(err_msg)
                raise ValueError(err_msg)
            r, X, mv = r[valid], X[valid], mv[valid]

            # 流通市值平方根作为回归权重
            weight = np.sqrt(mv.astype(float))

            try:
                if method == 'constrained':
                    # 加截距，全行业因子 + 行业流通市值加权和为0约束
                    constraints = _build_industry_constraint(X, mv)
                    params, _, resid = constrained_wls(
                        r.to_frame(), X.copy(), intercept=True,
                        weight=weight, constraints=constraints, verbose=True
                    )
                    self.factor_returns_dict[date] = params.squeeze()
                else:
                    # 默认 WLS：不加截距，全行业因子
                    params, _, resid = WLS(
                        r.to_frame(), X.copy(), intercept=False, weight=weight, verbose=True
                    )
                    self.factor_returns_dict[date] = params.squeeze()

                self.residuals_dict[date] = resid.squeeze()
            except Exception as e:
                err_msg = f'  {date}: 回归失败 - {e}'
                logger.error(err_msg)
                raise Exception(err_msg)

        # 构建因子收益率 DataFrame
        if self.factor_returns_dict:
            factor_returns_df = pd.DataFrame(self.factor_returns_dict).T
            factor_returns_df.index.name = 'datetime'
        else:
            factor_returns_df = pd.DataFrame()

        logger.info(f'横截面回归完成，共 {len(self.factor_returns_dict)} 期')
        return factor_returns_df

    def get_residuals(self) -> pd.DataFrame:
        """获取所有期的残差

        Returns:
            DataFrame, 索引 (instrument, datetime)，列 ['residual']
        """
        if not self.residuals_dict:
            return pd.DataFrame()

        frames = []
        for date, resid in self.residuals_dict.items():
            s = resid.rename('residual')
            s.index.name = 'instrument'
            df = s.to_frame()
            df['datetime'] = date
            frames.append(df)
        # 构建 <instrument,datetime> 索引
        result = pd.concat(frames).set_index('datetime', append=True)
        return result
