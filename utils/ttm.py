from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

import qlib
from qlib.config import C, REG_CN
from qlib.data import D


def load_period_events(instrument: str, field: str) -> pd.DataFrame:
    """读取单支股票的原始财报事件数据（公告日、期间、原始值）。"""
    field_token = field[2:] if field.startswith("$$") else field
    # qlib数据路径：/Users/chaofeng/.qlib/qlib_data/cn_data
    data_root = Path(C.dpm.get_data_uri())
    data_path = data_root / "financial" / instrument.lower() / f"{field_token}.data"
    if not data_path.exists():
        print(f"{data_path} 不存在，请确认已经准备好财务数据。")
        return pd.DataFrame()

    record_dtype = np.dtype(
        [
            ("ann_date", C.pit_record_type["date"]),
            ("period", C.pit_record_type["period"]),
            ("value", C.pit_record_type["value"]),
            ("_next", C.pit_record_type["index"]),
        ]
    )
    raw = np.fromfile(data_path, dtype=record_dtype)
    df = pd.DataFrame(raw)[["ann_date", "period", "value"]]

    df = df[df["ann_date"] > 0].copy()
    df["ann_date"] = pd.to_datetime(df["ann_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["ann_date"])
    df["period"] = df["period"].astype(int)
    df["value"] = df["value"].astype(float)
    df = df.sort_values(["ann_date", "period"]).reset_index(drop=True)
    return df


def compute_ttm_events(period_events: pd.DataFrame) -> pd.DataFrame:
    """
    按照 TTM 公式计算逐次公告的 TTM（近12个月滚动值）。

    逻辑说明：
    - 中国财报季度值为年初至今的累计值(YTD)
    - Q4 = 全年年报，因此 Q4 的 TTM 就是其本身
    - Q1/Q2/Q3: TTM = 当前期累计 + 去年年报 - 去年同期累计

    例如：
    - 2020Q1 TTM = 2020Q1累计 + 2019年报 - 2019Q1累计
    - 2020Q4 TTM = 2020Q4累计（即2020年报）
    """
    value_map: Dict[int, float] = {}
    results = []
    for _, row in period_events.iterrows():
        period = int(row["period"])
        value = float(row["value"])
        value_map[period] = value

        year = period // 100
        sub_period = period % 100  # 1=Q1, 2=Q2, 3=Q3, 4=Q4

        # 特殊处理：Q4 就是年报，TTM 直接等于 Q4 值
        if sub_period == 4:
            ttm = value
        else:
            # Q1/Q2/Q3: 需要去年年报和去年同期数据
            prev_same = (year - 1) * 100 + sub_period  # 去年同期
            prev_annual = (year - 1) * 100 + 4         # 去年年报

            prev_same_val = value_map.get(prev_same)
            prev_annual_val = value_map.get(prev_annual)

            if prev_same_val is None or prev_annual_val is None:
                ttm = np.nan
            else:
                ttm = value + prev_annual_val - prev_same_val

        results.append(
            {
                "ann_date": row["ann_date"],
                "period": period,
                "value": value,
                "ttm": ttm,
            }
        )
    return pd.DataFrame(results)


def make_dense_series(ttm_events: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    """将离散公告的 TTM 结果扩展到交易日频率（公告日之后前向填充）。"""
    effective = (
        ttm_events.dropna(subset=["ttm"])
        .sort_values("ann_date")
        .set_index("ann_date")["ttm"]
    )
    if effective.empty:
        return pd.Series(index=calendar, dtype=float)
    # 去除重复的日期
    effective = effective[~effective.index.duplicated(keep="last")]
    dense = effective.reindex(calendar, method="ffill")
    dense.index.name = "datetime"
    return dense


def build_ttm_features(
        instruments: Iterable[str],
        fields: Iterable[str],
        start_time: str,
        end_time: str,
) -> pd.DataFrame:
    """批量构造指定财务指标的 TTM 稠密数据。"""
    calendar = pd.DatetimeIndex(D.calendar(start_time, end_time))
    all_frames = []

    for field in fields:
        inst_series = {}
        for instrument in instruments:
            # 读取财报事件数据。DataFrame(ann_date, period, value)
            period_events = load_period_events(instrument, field)
            if period_events is None or period_events.empty:
                continue
            # 计算TTM (ann_date, period, value, ttm)
            ttm_events = compute_ttm_events(period_events)
            # 按照calendar 扩种ttm
            dense_series = make_dense_series(ttm_events, calendar)
            inst_series[instrument] = dense_series
        dense_df = pd.DataFrame(inst_series, index=calendar)
        dense_df.columns.name = "instrument"
        dense_df.index.name = "datetime"
        stacked = dense_df.stack().to_frame(name=f"TTM({field})")
        stacked = stacked.reorder_levels(["instrument", "datetime"]).sort_index()
        '''
                                           TTM($$roewa_q)
            datetime   instrument                
            2010-01-04 SH600000          0.254042
            2010-01-05 SH600000          0.009000
        '''
        all_frames.append(stacked)

    result = pd.concat(all_frames, axis=1).sort_index()
    return result


if __name__ == "__main__":
    qlib.init(provider_uri="~/.qlib/qlib_data/custom_data_hfq", region=REG_CN)

    instruments = ["SZ300498"]
    fields = ["$$n_income_q"]
    start_time = "2014-08-15"
    end_time = "2014-08-15"

    dense_ttm = build_ttm_features(instruments, fields, start_time, end_time)
    print(dense_ttm)
