"""
情形判断模块 - 判断信号属于Case 1还是Case 2
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm

from .config import SCENARIO_WINDOW, R2_THRESHOLD, P_VALUE_THRESHOLD
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class ScenarioClassifier:
    """情形分类器

    通过回归 Std_TS{g_n} = a + b * omega_n + epsilon
    判断信号与残差波动率的关系：
    - Case 1: 无显著关系，Alpha需乘以omega
    - Case 2: 有显著关系，Alpha无需乘以omega
    """

    def __init__(
        self,
        window: int = SCENARIO_WINDOW,
        r2_threshold: float = R2_THRESHOLD,
        p_threshold: float = P_VALUE_THRESHOLD
    ):
        """初始化

        Args:
            window: 用于计算信号时间序列标准差的窗口（交易日）
            r2_threshold: R^2阈值
            p_threshold: 系数b的p值阈值
        """
        self.window = window
        self.r2_threshold = r2_threshold
        self.p_threshold = p_threshold

    def classify(
        self,
        signal_df: pd.DataFrame,
        omega: pd.Series,
        as_of_date: str
    ) -> int:
        """判断信号的情形类型

        Args:
            signal_df: 原始信号，MultiIndex(instrument, datetime), column='g'
            omega: 残差波动率，Series(instrument -> float)
            as_of_date: 计算截止日期

        Returns:
            1 (Case 1) 或 2 (Case 2)
        """
        as_of_ts = pd.Timestamp(as_of_date)
        dates = signal_df.index.get_level_values('datetime').unique()
        dates = dates[dates <= as_of_ts].sort_values()

        if len(dates) < self.window:
            logger.warning(f'情形判断数据不足({len(dates)}<{self.window})，默认Case 1')
            return 1

        # 取最近window天
        recent_dates = dates[-self.window:]
        mask = signal_df.index.get_level_values('datetime').isin(recent_dates)
        recent_signal = signal_df.loc[mask]

        # 每只资产的信号时间序列标准差
        col = recent_signal.columns[0]
        ts_std = recent_signal.groupby(level='instrument')[col].std().dropna()

        # 对齐omega和ts_std
        common = ts_std.index.intersection(omega.index)
        if len(common) < 30:
            logger.warning(f'情形判断样本不足({len(common)}<30)，默认Case 1')
            return 1

        ts_std = ts_std.loc[common].values
        omega_vals = omega.loc[common].values

        # OLS回归: Std_TS = a + b * omega
        X = sm.add_constant(omega_vals)
        model = sm.OLS(ts_std, X).fit()
        r_squared = model.rsquared
        b_pvalue = model.pvalues[1] if len(model.pvalues) > 1 else 1.0

        case = 2 if (r_squared > self.r2_threshold and b_pvalue < self.p_threshold) else 1
        logger.info(f'情形判断: Case {case}, R²={r_squared:.4f}, '
                    f'b_pvalue={b_pvalue:.4f}')
        return case
