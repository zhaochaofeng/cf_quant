"""
信号处理模块 - 横截面标准化与单信号Alpha计算
"""
import pandas as pd
import numpy as np

from utils import LoggerFactory
from utils.preprocess import standardize

logger = LoggerFactory.get_logger(__name__)


class SignalProcessor:
    """信号处理器

    负责横截面z-score标准化和单信号Alpha公式计算
    """

    def cross_sectional_zscore(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """对每个交易日做横截面z-score标准化

        z_{CS,n}(t) = (g_n(t) - mu_CS(t)) / sigma_CS(t)

        Args:
            signal_df: 原始信号，MultiIndex(instrument, datetime), column='g'

        Returns:
            MultiIndex(instrument, datetime), column='z_cs'
        """
        result = standardize(signal_df, method='zscore', level='datetime')
        result.columns = ['z_cs']
        return result

    def compute_single_alpha(
        self,
        z_cs: pd.DataFrame,
        omega: pd.Series,
        ic: float,
        case: int,
        calc_date: str
    ) -> pd.DataFrame:
        """计算单信号Alpha

        Case 1: alpha_n(t) = omega_n * IC * z_CS_n(t)
        Case 2: alpha_n(t) = IC * z_CS_n(t)

        Args:
            z_cs: 横截面z-score，MultiIndex(instrument, datetime), column='z_cs'
            omega: 残差波动率，Series(instrument -> float)
            ic: 信息系数（标量）
            case: 情形，1 或 2
            calc_date: 计算日期

        Returns:
            DataFrame(index=instrument, column='alpha')，仅calc_date当天
        """
        calc_ts = pd.Timestamp(calc_date)
        dates = z_cs.index.get_level_values('datetime').unique().sort_values()
        if calc_ts not in dates:
            # calc_date不在数据中，取最近的可用交易日
            earlier = dates[dates <= calc_ts]
            if earlier.empty:
                raise ValueError(f'无可用信号数据: calc_date={calc_date}之前无数据')
            actual_date = earlier[-1]
            logger.warning(f'calc_date={calc_date}无信号数据，回退到{actual_date.date()}')
        else:
            actual_date = calc_ts

        z_today = z_cs.xs(actual_date, level='datetime')['z_cs']

        if case == 1:
            common = z_today.index.intersection(omega.index)
            alpha = omega.loc[common] * ic * z_today.loc[common]
        else:
            alpha = ic * z_today
            common = alpha.index

        result = pd.DataFrame({'alpha': alpha}, index=common)
        result.index.name = 'instrument'
        logger.info(f'单信号Alpha计算完成: Case{case}, IC={ic:.4f}, '
                    f'{len(result)}只股票')
        return result

    def compute_alpha_history(
        self,
        z_cs: pd.DataFrame,
        omega: pd.Series,
        ic: float,
        case: int
    ) -> pd.DataFrame:
        """计算历史所有日期的单信号Alpha（用于正交化）

        Args:
            z_cs: 横截面z-score，MultiIndex(instrument, datetime), column='z_cs'
            omega: 残差波动率，Series(instrument -> float)
            ic: 信息系数（标量）
            case: 情形，1 或 2

        Returns:
            MultiIndex(instrument, datetime), column='alpha'
        """
        if case == 1:
            common = z_cs.index.get_level_values('instrument')
            common_mask = common.isin(omega.index)
            z_filtered = z_cs.loc[common_mask].copy()
            inst = z_filtered.index.get_level_values('instrument')
            z_filtered['alpha'] = omega.reindex(inst).values * ic * z_filtered['z_cs'].values
        else:
            z_filtered = z_cs.copy()
            z_filtered['alpha'] = ic * z_filtered['z_cs']

        return z_filtered[['alpha']]
