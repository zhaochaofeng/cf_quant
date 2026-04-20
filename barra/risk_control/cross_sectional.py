"""横截面回归模块 - 加权最小二乘估计因子收益率"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from utils import WLS
from utils import LoggerFactory
logger = LoggerFactory.get_logger(__name__)

class CrossSectionalRegression:
    """横截面回归估计器

    使用加权最小二乘（WLS）进行横截面回归，估计因子收益率。
    权重为流通市值平方根。
    """

    def __init__(self):
        self.factor_returns_dict = {}  # {date: Series(factor -> return)}
        self.residuals_dict = {}       # {date: Series(instrument -> residual)}

    def fit_multi_periods(self, returns_df: pd.DataFrame,
                         exposure_df: pd.DataFrame,
                         market_cap_df: pd.DataFrame) -> pd.DataFrame:
        """多期横截面回归，逐日期使用 WLS 估计因子收益率

        模型：r_t = X_t * b_t + u_t
        权重：sqrt(circ_mv)

        Args:
            returns_df: 收益率，索引 (instrument, datetime)，单列
            exposure_df: 因子暴露，索引 (instrument, datetime)，多列为各因子
            market_cap_df: 市值数据，索引 (instrument, datetime)，含 'circ_mv' 列

        Returns:
            因子收益率矩阵，index=datetime, columns=factors
        """
        logger.info('开始多期横截面回归...')
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

            # 流通市值平方根作为权重
            weight = np.sqrt(mv.astype(float))

            try:
                # Barra 模型不加截距（行业哑变量已充当截距）
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
