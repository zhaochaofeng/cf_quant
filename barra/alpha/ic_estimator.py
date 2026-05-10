"""
IC估计模块 - 滚动信息系数计算
"""
import pandas as pd
import numpy as np

from .config import ROLLING_WINDOW, IC_LAG, MIN_IC_WINDOW
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class ICEstimator:
    """信息系数估计器

    计算池化IC = Corr(z_CS(t), theta(t+lag))
    """

    def __init__(self, window: int = ROLLING_WINDOW, lag: int = IC_LAG):
        """初始化

        Args:
            window: IC计算滚动窗口（交易日）
            lag: 残差收益率滞后期数
        """
        self.window = window
        self.lag = lag

    def compute_ic(
        self,
        z_cs: pd.DataFrame,
        residuals: pd.DataFrame
    ) -> float:
        """计算池化IC

        IC = Corr(z_CS(t), theta(t+lag)), 过去window天所有(n,t)观测

        Args:
            z_cs: 横截面z-score，MultiIndex(instrument, datetime), column='z_cs'
            residuals: 残差收益率，MultiIndex(instrument, datetime), column='residual'

        Returns:
            IC值（标量float）
        """
        z_cs.dropna(inplace=True)
        residuals.dropna(inplace=True)
        residuals.sort_index(inplace=True)
        # [t+1,t+2] 区间的残差收益率
        residuals_s = residuals.groupby('instrument', group_keys=False).apply(lambda x: x.shift(-1)).dropna()
        comm_date = ((z_cs.index.get_level_values('datetime').
                     intersection(residuals_s.index.get_level_values('datetime'))).
                     unique().sort_values())
        logger.info(f'IC计算日期范围: {comm_date[0]} ~ {comm_date[-1]}, 计算天数: {len(comm_date)}')
        if len(comm_date) < MIN_IC_WINDOW:
            err_msg = f'IC计算数据不足({len(comm_date)}<{MIN_IC_WINDOW})'
            logger.warning(err_msg)
            raise ValueError(err_msg)

        comm_index = residuals_s.index.intersection(z_cs.index)
        residuals_s = residuals_s.loc[comm_index]
        z_cs = z_cs.loc[comm_index]

        ic = np.corrcoef(z_cs.values.reshape(-1), residuals_s.values.reshape(-1))[0, 1]
        logger.info(f'IC估计: {ic:.6f}')
        return float(ic)
