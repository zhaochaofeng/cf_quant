"""Stratified (grouped) return analysis — Layer 2 of factor evaluation.

Computes equal-weighted returns for factor-sorted quantile groups,
with qlib cross-validation of long-short returns.
"""

import pandas as pd
from qlib.contrib.eva.alpha import calc_long_short_return as qlib_ls


class StratifiedReturn:
    """Compute stratified returns by sorting stocks into factor quantile groups.

    For each trading date, stocks are divided into n_groups equal-sized bins
    based on factor values (using pd.qcut). Within each bin, equal-weighted
    mean returns are computed, yielding a daily group return time series.
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
                - 'long_short': Series(index=datetime), equal-weighted return of the
                  top quantile group (n_groups-1) minus the bottom group (0).
                  NaN on dates where either group could not be formed.
                - 'long_short_qlib': Series(index=datetime), qlib cross-validation
                  long-short return using quantile=1/n_groups.

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
                'long_short_qlib': pd.Series(dtype=float),
            }

        group_returns = pd.DataFrame(records).reindex(columns=range(n))

        # Long-short: top group minus bottom group among existing groups
        max_grp = group_returns.columns.max()
        min_grp = group_returns.columns.min()
        long_short = group_returns[max_grp] - group_returns[min_grp]
        long_short.name = 'long_short'

        # Cross-validation with qlib's calc_long_short_return
        quantile = 1.0 / n
        ls_qlib, avg_qlib = qlib_ls(
            df[factor_col], df[ret_col], date_col='datetime', quantile=quantile
        )
        ls_qlib.name = 'long_short_qlib'

        return {
            'group_returns': group_returns,
            'long_short': long_short,
            'long_short_qlib': ls_qlib,
            'avg_qlib': avg_qlib,
        }
