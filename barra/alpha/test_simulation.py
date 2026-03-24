"""
Alpha模块模拟测试 - 端到端验证单信号和多信号流水线

使用合成数据，无需MySQL/Qlib依赖
"""
import sys
import os
import tempfile

import numpy as np
import pandas as pd

# 确保项目根目录在path中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from barra.alpha.alpha_engine import AlphaEngine


# ---- 模拟数据生成 ----

N_STOCKS = 100
N_DAYS = 600
N_INDUSTRIES = 10
CALC_DATE = '2026-03-04'


def _generate_dates() -> pd.DatetimeIndex:
    """生成交易日序列，确保包含CALC_DATE"""
    end = pd.Timestamp(CALC_DATE)
    # 向前推 N_DAYS 个交易日
    dates = pd.bdate_range(end=end, periods=N_DAYS)
    return dates


def _generate_instruments(n: int) -> list[str]:
    """生成股票代码"""
    return [f'SH60{i:04d}' for i in range(n)]


def _build_multiindex(instruments: list[str], dates: pd.DatetimeIndex) -> pd.MultiIndex:
    """构建 (instrument, datetime) MultiIndex"""
    tuples = [(inst, dt) for inst in instruments for dt in dates]
    return pd.MultiIndex.from_tuples(tuples, names=['instrument', 'datetime'])


def generate_residuals(instruments: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """模拟残差收益率

    残差 ~ N(0, sigma_n)，其中 sigma_n 因股票而异
    """
    np.random.seed(42)
    idx = _build_multiindex(instruments, dates)
    n = len(instruments)
    # 每只股票有不同的波动率
    sigmas = np.random.uniform(0.01, 0.05, size=n)
    values = []
    for sigma in sigmas:
        values.append(np.random.normal(0, sigma, size=len(dates)))
    residual = np.concatenate(values)
    return pd.DataFrame({'residual': residual}, index=idx).sort_index()


def generate_signal(
    instruments: list[str],
    dates: pd.DatetimeIndex,
    residuals: pd.DataFrame,
    correlation: float = 0.0,
    seed: int = 100
) -> pd.DataFrame:
    """模拟信号数据

    当 correlation > 0 时，信号与 t+2 残差部分相关

    Args:
        instruments: 股票列表
        dates: 交易日列表
        residuals: 残差DataFrame
        correlation: 信号与未来残差的目标相关性
        seed: 随机种子
    """
    np.random.seed(seed)
    idx = _build_multiindex(instruments, dates)
    noise = np.random.randn(len(idx))

    if correlation > 0:
        # 对每只股票，用 t+2 残差构造部分相关信号
        signal_parts = []
        for inst in instruments:
            r = residuals.xs(inst, level='instrument')['residual']
            # shift -2 获取 t+2 残差（信号时点看到未来残差）
            r_future = r.shift(-2).reindex(dates).fillna(0).values
            n_noise = np.random.randn(len(dates))
            sig = correlation * r_future + np.sqrt(1 - correlation ** 2) * n_noise
            signal_parts.append(sig)
        signal = np.concatenate(signal_parts)
    else:
        signal = noise

    return pd.DataFrame({'g': signal}, index=idx).sort_index()


def generate_industry(instruments: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """模拟行业数据"""
    np.random.seed(77)
    idx = _build_multiindex(instruments, dates)
    # 每只股票固定行业
    ind_map = {inst: f'ind_{i % N_INDUSTRIES}' for i, inst in enumerate(instruments)}
    codes = [ind_map[inst] for inst, _ in idx]
    return pd.DataFrame({'industry_code': codes}, index=idx).sort_index()


def generate_market_cap(instruments: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """模拟流通市值（万元）"""
    np.random.seed(88)
    idx = _build_multiindex(instruments, dates)
    n = len(instruments)
    base_mv = np.random.uniform(100000, 5000000, size=n)
    values = []
    for mv in base_mv:
        # 简单随机游走
        returns = np.random.normal(0, 0.01, size=len(dates))
        mv_series = mv * np.cumprod(1 + returns)
        values.append(mv_series)
    circ_mv = np.concatenate(values)
    return pd.DataFrame({'circ_mv': circ_mv}, index=idx).sort_index()


# ---- 测试函数 ----

def test_single_signal():
    """测试1: 单信号端到端"""
    print('\n' + '=' * 70)
    print('测试1: 单信号Alpha端到端测试')
    print('=' * 70)

    dates = _generate_dates()
    instruments = _generate_instruments(N_STOCKS)
    print(f'日期范围: {dates[0].date()} ~ {dates[-1].date()}, 共{len(dates)}天')
    print(f'股票数: {len(instruments)}')

    # 生成数据
    residuals = generate_residuals(instruments, dates)
    signal = generate_signal(instruments, dates, residuals, correlation=0.3, seed=100)
    industry = generate_industry(instruments, dates)
    market_cap = generate_market_cap(instruments, dates)

    print(f'残差 shape: {residuals.shape}')
    print(f'信号 shape: {signal.shape}')

    # 运行
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = AlphaEngine(output_dir=tmpdir)
        alpha = engine.run_with_data(
            signals={'signal_1': signal},
            residuals=residuals,
            industry_df=industry,
            market_cap_df=market_cap,
            calc_date=CALC_DATE
        )

    print('\n--- 单信号Alpha结果 ---')
    print(f'Shape: {alpha.shape}')
    print(f'统计:\n{alpha.describe()}')
    print(f'前10只:\n{alpha.head(10)}')

    # 验证
    assert len(alpha) > 0, 'Alpha结果为空'
    assert 'alpha' in alpha.columns, '缺少alpha列'
    assert not alpha['alpha'].isna().any(), '存在NaN'
    print('\n[PASS] 单信号测试通过')
    return alpha


def test_multi_signal():
    """测试2: 多信号正交化端到端"""
    print('\n' + '=' * 70)
    print('测试2: 多信号Alpha正交化端到端测试')
    print('=' * 70)

    dates = _generate_dates()
    instruments = _generate_instruments(N_STOCKS)
    print(f'日期范围: {dates[0].date()} ~ {dates[-1].date()}, 共{len(dates)}天')
    print(f'股票数: {len(instruments)}')

    # 生成数据
    residuals = generate_residuals(instruments, dates)
    signal_1 = generate_signal(instruments, dates, residuals, correlation=0.3, seed=100)
    signal_2 = generate_signal(instruments, dates, residuals, correlation=0.15, seed=200)
    signal_3 = generate_signal(instruments, dates, residuals, correlation=0.0, seed=300)
    industry = generate_industry(instruments, dates)
    market_cap = generate_market_cap(instruments, dates)

    print(f'信号1 shape: {signal_1.shape} (corr=0.3)')
    print(f'信号2 shape: {signal_2.shape} (corr=0.15)')
    print(f'信号3 shape: {signal_3.shape} (noise)')

    # 运行
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = AlphaEngine(output_dir=tmpdir)
        alpha = engine.run_with_data(
            signals={
                'signal_1': signal_1,
                'signal_2': signal_2,
                'signal_3': signal_3,
            },
            residuals=residuals,
            industry_df=industry,
            market_cap_df=market_cap,
            calc_date=CALC_DATE
        )

    print('\n--- 多信号正交化Alpha结果 ---')
    print(f'Shape: {alpha.shape}')
    print(f'统计:\n{alpha.describe()}')
    print(f'前10只:\n{alpha.head(10)}')

    # 验证
    assert len(alpha) > 0, 'Alpha结果为空'
    assert 'alpha' in alpha.columns, '缺少alpha列'
    assert not alpha['alpha'].isna().any(), '存在NaN'
    print('\n[PASS] 多信号正交化测试通过')
    return alpha


if __name__ == '__main__':
    alpha_single = test_single_signal()
    alpha_multi = test_multi_signal()

    # 对比
    print('\n' + '=' * 70)
    print('单信号 vs 多信号 Alpha 对比')
    print('=' * 70)
    common = alpha_single.index.intersection(alpha_multi.index)
    if len(common) > 0:
        corr = alpha_single.loc[common, 'alpha'].corr(alpha_multi.loc[common, 'alpha'])
        print(f'共同股票数: {len(common)}')
        print(f'单信号Alpha std: {alpha_single["alpha"].std():.6f}')
        print(f'多信号Alpha std: {alpha_multi["alpha"].std():.6f}')
        print(f'相关系数: {corr:.4f}')
    print('\n全部测试完成!')
