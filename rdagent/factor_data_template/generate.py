from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.config import C
from qlib.data import D
from qlib.data.pit import P
from qlib.log import get_module_logger


class PTTM(P):

    """优化版本：基于事件驱动的 TTM 计算"""

    TTM_WINDOW_SIZE = 8

    def __str__(self):
        return f"PTTM({self.feature})"

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = D.calendar(freq=freq)
        calendar_range = pd.DatetimeIndex([_calendar[i] for i in range(start_index, end_index + 1)])

        try:
            # 1. 一次性读取所有 period 事件数据
            field_name = str(self.feature)
            period_events = self._load_all_period_events(instrument, field_name)
            if period_events.empty:
                return pd.Series(dtype="float32", name=str(self))

            # 2. 批量计算每个公告的 TTM
            ttm_events = self._compute_ttm_events(period_events)

            # 3. Forward-fill 到日历
            dense_series = self._make_dense_series(ttm_events, calendar_range)

            # 4. 转换为 RangeIndex
            resample_data = dense_series.values.astype("float32")
            return pd.Series(
                resample_data,
                index=pd.RangeIndex(start_index, end_index + 1),
                dtype="float32",
                name=str(self)
            )
        except FileNotFoundError:
            get_module_logger("PTTM").warning(f"WARN: period data not found for {str(self)}")
            return pd.Series(dtype="float32", name=str(self))

    def _load_all_period_events(self, instrument: str, field: str) -> pd.DataFrame:
        """一次性读取所有 period 事件"""
        field_token = field[2:] if field.startswith("$$") else field
        data_root = Path(C.dpm.get_data_uri())
        data_path = data_root / "financial" / instrument.lower() / f"{field_token}.data"

        if not data_path.exists():
            get_module_logger(f"{data_path} 不存在")
            return pd.DataFrame()

        record_dtype = np.dtype([
            ("ann_date", C.pit_record_type["date"]),
            ("period", C.pit_record_type["period"]),
            ("value", C.pit_record_type["value"]),
            ("_next", C.pit_record_type["index"]),
        ])
        raw = np.fromfile(data_path, dtype=record_dtype)
        df = pd.DataFrame(raw)[["ann_date", "period", "value"]]

        df = df[df["ann_date"] > 0].copy()
        df["ann_date"] = pd.to_datetime(df["ann_date"].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["ann_date"])
        df["period"] = df["period"].astype(int)
        df["value"] = df["value"].astype(float)
        return df.sort_values(["ann_date", "period"]).reset_index(drop=True)

    def _compute_ttm_events(self, period_events: pd.DataFrame) -> pd.DataFrame:
        """批量计算所有 TTM 值"""
        value_map = {}
        results = []

        for _, row in period_events.iterrows():
            period = int(row["period"])
            value = float(row["value"])
            value_map[period] = value

            year, quarter = period // 100, period % 100

            if quarter == 4:
                ttm = value
            else:
                prev_same = (year - 1) * 100 + quarter
                prev_annual = (year - 1) * 100 + 4
                prev_same_val = value_map.get(prev_same)
                prev_annual_val = value_map.get(prev_annual)

                ttm = value + prev_annual_val - prev_same_val if (
                            prev_same_val is not None and prev_annual_val is not None) else np.nan

            results.append({
                "ann_date": row["ann_date"],
                "period": period,
                "ttm": ttm,
            })
        return pd.DataFrame(results)

    def _make_dense_series(self, ttm_events: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
        """Forward-fill 到交易日频率"""
        effective = (
            ttm_events.dropna(subset=["ttm"])
            .sort_values("ann_date")
            .set_index("ann_date")["ttm"]
        )
        if effective.empty:
            return pd.Series(np.nan, index=calendar, dtype=float)

        effective = effective[~effective.index.duplicated(keep="last")]
        dense = effective.reindex(calendar, method="ffill")
        return dense


_setup_kwargs = {'custom_ops': [PTTM]}
qlib.init(provider_uri="~/.qlib/qlib_data/custom_data_hfq", **_setup_kwargs)

# 减少数据量
instruments = D.instruments(market='csi300')
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor", "PTTM($$n_income_q)", "PTTM($$n_income_attr_p_q)"]
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2015-01-01":].sort_index()

data.dropna(how='all', subset=["$open", "$close", "$high", "$low"], inplace=True)

data.to_hdf("./daily_pv_all.h5", key="data")


fields = ["$open", "$close", "$high", "$low", "$volume", "$factor", "PTTM($$n_income_q)", "PTTM($$n_income_attr_p_q)"]
# 获取前一个data中的股票代码
target_instruments = data.reset_index()["instrument"].unique()[:100]

# 直接获取目标时间范围的数据，并过滤出共同存在的股票
subset_data = D.features(target_instruments, fields,
                         start_time="2018-01-01", end_time="2019-12-31", freq="day")

# 确保只保留两个数据集都有的股票
available_instruments = subset_data.index.get_level_values('instrument').unique()
final_data = (
    subset_data.swaplevel().sort_index()
).swaplevel().loc[available_instruments.intersection(target_instruments)].swaplevel().sort_index()

final_data.to_hdf("./daily_pv_debug.h5", key="data")


# data = (
#     (
#         D.features(instruments, fields, start_time="2018-01-01", end_time="2019-12-31", freq="day")
#         .swaplevel()
#         .sort_index()
#     )
#     .swaplevel()
#     .loc[data.reset_index()["instrument"].unique()[:100]]
#     .swaplevel()
#     .sort_index()
# )
#
# data.to_hdf("./daily_pv_debug.h5", key="data")

'''
PTTM($$n_income_q): Net Profit Including Non-controlling Interests. Net profit for the reporting period on a consolidated basis, representing the total profit attributable to both the parent company’s shareholders and non-controlling interests (minority interests).
PTTM($$n_income_attr_p_q): Net Profit Attributable to Parent Company Shareholders. Net profit for the reporting period attributable to the shareholders of the parent company, excluding the portion attributable to non-controlling interests.
'''
