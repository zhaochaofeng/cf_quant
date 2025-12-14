import qlib
from qlib.data.pit import P
from qlib.data import D
from qlib.log import get_module_logger
import numpy as np
import pandas as pd


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


_setup_kwargs = {'custom_ops': [PTTM]}
qlib.init(provider_uri="~/.qlib/qlib_data/custom_data_hfq", **_setup_kwargs)

instruments = D.instruments()
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor", "PTTM($$n_income_q)", "PTTM($$n_income_attr_p_q)"]
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2015-01-01":].sort_index()

data.to_hdf("./daily_pv_all.h5", key="data")


fields = ["$open", "$close", "$high", "$low", "$volume", "$factor", "PTTM($$n_income_q)", "PTTM($$n_income_attr_p_q)"]
data = (
    (
        D.features(instruments, fields, start_time="2018-01-01", end_time="2019-12-31", freq="day")
        .swaplevel()
        .sort_index()
    )
    .swaplevel()
    .loc[data.reset_index()["instrument"].unique()[:100]]
    .swaplevel()
    .sort_index()
)

data.to_hdf("./daily_pv_debug.h5", key="data")
