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
        residuals: pd.DataFrame,
        as_of_date: str
    ) -> float:
        """计算池化IC

        IC = Corr(z_CS(t), theta(t+lag)), 过去window天所有(n,t)观测

        Args:
            z_cs: 横截面z-score，MultiIndex(instrument, datetime), column='z_cs'
            residuals: 残差收益率，MultiIndex(instrument, datetime), column='residual'
            as_of_date: 计算截止日期

        Returns:
            IC值（标量float）
        """
        as_of_ts = pd.Timestamp(as_of_date)

        # 获取z_cs的交易日列表
        z_dates = z_cs.index.get_level_values('datetime').unique().sort_values()
        z_dates = z_dates[z_dates <= as_of_ts]

        if len(z_dates) < MIN_IC_WINDOW:
            logger.warning(f'IC计算数据不足({len(z_dates)}<{MIN_IC_WINDOW})，返回IC=0')
            return 0.0

        # 取最近window天
        recent_dates = z_dates[-self.window:]

        # 构建残差日期映射：对于z_cs的每个日期t，需要theta(t+lag)
        resid_dates = residuals.index.get_level_values('datetime').unique().sort_values()
        all_trade_dates = z_dates.union(resid_dates).sort_values()

        # 构建日期映射表: date -> date+lag
        date_to_future = {}
        date_list = all_trade_dates.tolist()
        for i, d in enumerate(date_list):
            if i + self.lag < len(date_list):
                date_to_future[d] = date_list[i + self.lag]

        # 收集配对数据
        z_vals = []
        r_vals = []
        z_col = z_cs.columns[0]
        r_col = residuals.columns[0]

        for date in recent_dates:
            future_date = date_to_future.get(date)
            if future_date is None:
                continue

            # 取当日z_cs和未来日残差
            try:
                z_cross = z_cs.xs(date, level='datetime')[z_col]
            except KeyError:
                continue
            try:
                r_cross = residuals.xs(future_date, level='datetime')[r_col]
            except KeyError:
                continue

            # 对齐instrument，去除重复索引
            z_cross = z_cross[~z_cross.index.duplicated(keep='first')]
            r_cross = r_cross[~r_cross.index.duplicated(keep='first')]
            common = z_cross.index.intersection(r_cross.index)
            if common.empty:
                continue

            z_vals.append(z_cross.loc[common].values)
            r_vals.append(r_cross.loc[common].values)

        if not z_vals:
            logger.warning('无有效配对数据，IC=0')
            return 0.0

        z_all = np.concatenate(z_vals)
        r_all = np.concatenate(r_vals)

        # 去除NaN
        valid = ~(np.isnan(z_all) | np.isnan(r_all))
        z_all = z_all[valid]
        r_all = r_all[valid]

        if len(z_all) < 100:
            logger.warning(f'有效观测不足({len(z_all)}<100)，IC=0')
            return 0.0

        ic = np.corrcoef(z_all, r_all)[0, 1]
        logger.info(f'IC估计: {ic:.6f}, 有效观测: {len(z_all)}')
        return float(ic)
