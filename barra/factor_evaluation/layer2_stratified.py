"""Stratified (grouped) return analysis — Layer 2 of factor evaluation.

Computes daily group returns for factor-sorted quantile bins via manual
pd.qcut, and long-short returns via qlib's calc_long_short_return.
"""

import numpy as np
import pandas as pd
from qlib.contrib.eva.alpha import calc_long_short_return as qlib_ls
from utils import coef_tstat


class StratifiedReturn:
    """Compute stratified returns by sorting stocks into factor quantile groups.

    For each trading date, stocks are divided into n_groups equal-sized bins
    based on factor values (using pd.qcut). Within each bin, equal-weighted
    mean returns are computed, yielding a daily group return time series.
    注：对于 行业因子，取值[0,1]. 当 n_groups > 2 时，无法划分分组，从而导致 monotonic_tstat 指标为None.
    """

    def __init__(self, n_groups: int = 5):
        """Initialize with number of quantile groups.

        Args:
            n_groups: Number of quantile groups for stratification.
        """
        self.n_groups = n_groups

    def compute(self, df: pd.DataFrame, factor_col: str, ret_col: str) -> dict:
        """Compute daily group returns and long-short return series.

        For each trading date, stocks are sorted into n_groups quantile bins
        by factor_col. Equal-weighted mean ret_col is computed per bin.
        Long-short is the spread between the top and bottom group.

        Args:
            df: MultiIndex (instrument, datetime) DataFrame containing
                factor_col and ret_col.
            factor_col: Column name for factor values used for sorting.
            ret_col: Column name for forward return values.

        Returns:
            dict with keys:
                - 'group_returns': DataFrame(index=datetime, columns=[0..n_groups-1]),
                  equal-weighted mean return per factor quantile group per date.
                - 'long_short': Series(index=datetime), qlib long-short return
                  (top quantile minus bottom quantile).
                - 'avg_return': Series(index=datetime), qlib long-average return
                  (top quantile minus average).
                - 'monotonic_tstat': float, t-statistic of monotonicity slope
                  from panel regression r_g,t = α + β * group_id + ε_g,t.
                  Tests whether group returns show a significant linear trend.

        Raises:
            ValueError: If factor_col or ret_col are not in df.columns.
        """
        if factor_col not in df.columns:
            raise ValueError(f"factor_col '{factor_col}' not in DataFrame columns")
        if ret_col not in df.columns:
            raise ValueError(f"ret_col '{ret_col}' not in DataFrame columns")

        n = self.n_groups
        dates = df.index.get_level_values('datetime').unique()

        records = []
        # 计算每天分组收益率均值
        for dt in dates:
            g = df.xs(dt, level='datetime')
            try:
                labels = pd.qcut(g[factor_col], n, labels=False,
                                 duplicates='drop')
            except ValueError:
                continue
            row = g.groupby(labels)[ret_col].mean()
            row.name = dt
            records.append(row)

        if not records:
            return {
                'group_returns': pd.DataFrame(columns=range(n)),
                'long_short': pd.Series(dtype=float),
                'avg_return': pd.Series(dtype=float),
                'monotonic_tstat': 0.0,
            }

        group_returns = pd.DataFrame(records).reindex(columns=range(n))

        # Long-short via qlib
        quantile = 1.0 / n
        long_short, avg_return = qlib_ls(
            df[factor_col], df[ret_col], date_col='datetime', quantile=quantile,
        )
        long_short.name = 'long_short'

        # Monotonicity: panel regression r_g,t = α + β * group_id + ε_g,t
        # 使用 T×G 条日频分组样本直接拟合（非先聚合再回归）。
        # 相比先对每组求均值再回归（仅 G 个观测），panel 回归保留了
        # 时间维度信息，自由度 T×G−2 >> G−2，统计功效更高。
        y = group_returns.values.flatten()
        x = np.tile(np.arange(n), len(group_returns))
        valid = ~np.isnan(y)
        if valid.sum() >= 3:
            monotonic_tstat = coef_tstat(x[valid], y[valid])["t_beta"]
        else:
            monotonic_tstat = 0.0

        return {
            'group_returns': group_returns,
            'long_short': long_short,
            'avg_return': avg_return,
            'monotonic_tstat': monotonic_tstat,
        }
