'''
    自定义 qlib 操作算子
    注：在qlib.init()初始化时注册，如：
        _setup_kwargs = {'custom_ops': [CMean]}
        qlib.init(provider_uri=provider_uri, **_setup_kwargs)
'''

import sys

import numpy as np
import pandas as pd
from qlib.data import D
from qlib.data.ops import Expression, ExpressionOps
from qlib.data.ops import (
    If, Feature
)
from qlib.data.ops import Operators
from qlib.data.pit import P
from qlib.log import get_module_logger


class LastValue(ExpressionOps):
    """ 获取特征序列最后一个值的操作符。
        如 获取 expanding 滚动均值/标准差 最后一个值得到全局均值/标准差
          'LastValue(Mean($close, 0))', 'LastValue(Std($close, 0))'
    Parameters
    ----------
    feature : Expression
        特征表达式
    Returns
    ----------
    Expression
        返回整个序列的最后一个值
    """

    def __init__(self, feature):
        self.feature = feature

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.feature)

    def _load_internal(self, instrument, start_index, end_index, *args):
        # 加载整个特征序列
        series = self.feature.load(instrument, start_index, end_index, *args)
        # 获取最后一个非NaN值
        if len(series) > 0:
            # 使用 iloc[-1] 获取最后一个值
            return pd.Series([series.iloc[-1]] * len(series), index=series.index)
        else:
            return pd.Series([np.nan] * len(series), index=series.index)

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling()

    def get_extended_window_size(self):
        return self.feature.get_extended_window_size()


class ConstantOps(ExpressionOps):
    """ 常数类型的计算
        如：计算整个序列的均值、方差
        返回相同序列长度的常数序列
    """
    def __init__(self, feature, func):
        self.feature = feature
        self.func = func

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.feature)

    def _load_internal(self, instrument, start_index, end_index, *args):
        # 加载整个特征序列
        series = self.feature.load(instrument, start_index, end_index, *args)
        if len(series) > 0:
            s = getattr(series, self.func)()
            return pd.Series([s] * len(series), index=series.index)
        else:
            return pd.Series([np.nan] * len(series), index=series.index)

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling()

    def get_extended_window_size(self):
        return self.feature.get_extended_window_size()


class CMean(ConstantOps):
    """ 求序列均值 """
    def __init__(self, feature):
        super(CMean, self).__init__(feature, 'mean')


class CStd(ConstantOps):
    """ 求序列标准差 """
    def __init__(self, feature):
        super(CStd, self).__init__(feature, 'std')


class PTTM(P):
    """
    PTTM: 基于 Qlib PIT 系统的 TTM (Trailing Twelve Months) 操作符实现
    实现 TTM 公式：
       - Q4: TTM = 当前值（年报）
       - Q1/Q2/Q3: TTM = 当前累计 + 去年Q4 - 去年同期

    例子：
    import qlib
    from qlib.data import D
    from qlib.config import REG_CN
    from qlib.data.ops import Operators

    # 初始化 Qlib
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    # 注册 PTTM 操作符
    Operators.register([PTTM])

    instruments = ["SH600000"]
    fields = [
        "P($$roewa_q)",  # 原始季度值（YTD 累计）
        "PTTM($$roewa_q)",  # TTM 值
    ]

    data = D.features(
        instruments,
        fields,
        start_time="2009-01-01",
        end_time="2020-01-01",
        freq="day"
    )
    print(data)

    """

    def __str__(self):
        return f"PTTM({self.feature})"

    # TTM 计算需要获取去年同期和去年Q4
    # _load_feature 返回的 Series 已按 period 去重（同一 period 多次公告只保留最新值）
    # 设为 8 可覆盖完整的 2 年数据，确保安全
    TTM_WINDOW_SIZE = 8

    def _load_internal(self, instrument, start_index, end_index, freq):
        """
        重写 P 类的 _load_internal 方法，实现 TTM 计算

        通过增大 window size 来获取足够的历史 period 数据，
        复用 P 类的 _load_feature 方法。
        """
        _calendar = D.calendar(freq=freq)
        resample_data = np.empty(end_index - start_index + 1, dtype="float32")

        # 用于 forward-fill 的最后有效 TTM 值
        last_valid_ttm = np.nan

        for cur_index in range(start_index, end_index + 1):
            cur_time = _calendar[cur_index]
            # To load expression accurately, more historical data are required
            start_ws, end_ws = self.feature.get_extended_window_size()
            if end_ws > 0:
                raise ValueError(
                    "PIT database does not support referring to future period (e.g. expressions like `Ref('$$roewa_q', -1)` are not supported"
                )
            try:
                # ws=4 可获取当前 period + 去年同期 + 去年Q4
                s = self._load_feature(instrument, -self.TTM_WINDOW_SIZE, 0, cur_time)

                if len(s) == 0:
                    resample_data[cur_index - start_index] = last_valid_ttm
                    continue

                ttm_value = self._compute_ttm(s)

                if not np.isnan(ttm_value):
                    last_valid_ttm = ttm_value

                resample_data[cur_index - start_index] = last_valid_ttm

            except FileNotFoundError:
                get_module_logger("PTTM").warning(
                    f"WARN: period data not found for {str(self)}"
                )
                return pd.Series(dtype="float32", name=str(self))

        resample_series = pd.Series(
            resample_data,
            index=pd.RangeIndex(start_index, end_index + 1),
            dtype="float32",
            name=str(self)
        )
        return resample_series

    def _compute_ttm(self, period_series: pd.Series) -> float:
        """
        计算 TTM 值

        Parameters
        ----------
        period_series : pd.Series
            period 到 value 的映射
            例如：Series([0.05, 0.08, 0.10], index=[201801, 201802, 201803])

        Returns
        -------
        float
            TTM 计算结果，如果数据不足则返回 NaN

        逻辑说明
        --------
        1. 提取最新 period（如 201901 = 2019Q1）
        2. 判断是否为 Q4：
           - 是 Q4: TTM = 当前值（年报本身就是 TTM）
           - 非 Q4: TTM = 当前累计 + 去年Q4 - 去年同期
        3. 从 period_series 中查找历史数据
        """
        # 提取最新的 period 和对应的值
        current_period = int(period_series.index[-1])  # 如 201901
        current_value = float(period_series.iloc[-1])  # 当前期的值

        # 解析 period
        year = current_period // 100  # 2019
        quarter = current_period % 100  # 1 (Q1)

        # Q4 特殊处理：年报本身就是 TTM
        if quarter == 4:
            return current_value

        # Q1/Q2/Q3: 需要去年同期和去年Q4
        prev_same_period = (year - 1) * 100 + quarter  # 去年同期，如 201801
        prev_annual_period = (year - 1) * 100 + 4  # 去年Q4，如 201804

        # 从 Series 中查找历史数据
        prev_same_value = period_series.get(prev_same_period, None)
        prev_annual_value = period_series.get(prev_annual_period, None)

        # 检查数据完整性
        if prev_same_value is None or prev_annual_value is None:
            # 数据不足，无法计算 TTM（通常发生在首年）
            # 返回 NaN，但会被 forward-fill 机制处理
            return np.nan

        # 应用 TTM 公式
        ttm = current_value + prev_annual_value - prev_same_value
        return float(ttm)



def standardize(feature: Expression):
    """ 标准化 """
    # mean = LastValue(Mean(feature, 0))
    # std = LastValue(Std(feature, 0))
    mean = CMean(feature)
    std = CStd(feature)
    x = (feature - mean) / std
    return x


def winsorize(feature: Expression, k: int = 3):
    """ 去极值 """
    # mean = LastValue(Mean(feature, 0))
    # std = LastValue(Std(feature, 0))
    mean = CMean(feature)
    std = CStd(feature)
    lower_bound = mean - k * std
    upper_bound = mean + k * std
    x = If(feature > upper_bound, upper_bound, If(feature < lower_bound, lower_bound, feature))
    return x


def main():
    """ 测试函数 """
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq', region=REG_CN)
    # 注册自定义操作
    Operators.register([CMean, CStd, PTTM])
    instruments = ['SZ000001', 'SH600000']

    # feature = Feature('close')
    # feature = winsorize(feature, 1)
    # feature = standardize(feature)
    feature = 'PTTM($$roewa_q)'

    feas = D.features(instruments, ['$close', feature],
                      start_time='2025-07-10', end_time='2025-08-11', freq='day')
    feas.columns = ['close', 'feature']
    print(feas.head())


if __name__ == '__main__':
    sys.exit(main())