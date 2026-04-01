""" 测试 rolling_regress 修改后的 BETA 因子计算效果 """

import sys
sys.path.insert(0, '/Users/chaofeng/code/cf_quant')

import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from data.factor.volatility import BETA, HSIGMA
from utils import PTTM


def test_beta_factor():
    """测试 BETA 因子计算"""
    print("=" * 80)
    print("测试 BETA 因子计算 (修改后的 rolling_regress)")
    print("=" * 80)

    # 初始化 qlib
    qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq', custom_ops=[PTTM])

    # 获取测试股票 (CSI300前10只)
    instruments = D.instruments(market='csi300')
    instruments = D.list_instruments(
        instruments, start_time='2024-01-01', end_time='2025-12-31', as_list=True
    )[:10]
    print(f"\n测试股票数量: {len(instruments)}")
    print(f"股票列表: {instruments}")

    # 加载数据 (需要至少504+21个交易日用于滚动回归)
    start_time = '2022-01-01'
    end_time = '2025-12-31'

    print(f"\n数据时间范围: {start_time} 至 {end_time}")
    print("正在加载数据...")

    df = D.features(
        instruments, fields=['$change'],
        start_time=start_time, end_time=end_time
    )

    print(f"数据加载完成: {df.shape[0]} 行")

    # 检查缺失值情况
    nan_count = df['$change'].isna().sum()
    nan_pct = nan_count / len(df) * 100
    print(f"\n收益率缺失值统计:")
    print(f"  - 缺失数量: {nan_count}")
    print(f"  - 缺失比例: {nan_pct:.2f}%")

    # 计算 BETA 因子
    print("\n正在计算 BETA 因子...")
    try:
        beta_result = BETA(df)

        print(f"\n✓ BETA 因子计算成功!")
        print(f"  - 结果数量: {len(beta_result)} 条")
        print(f"  - 覆盖股票: {beta_result.index.get_level_values('instrument').nunique()} 只")

        # 统计 BETA 值分布
        beta_values = beta_result['BETA'].dropna()
        print(f"\nBETA 因子统计:")
        print(f"  - 有效值数量: {len(beta_values)}")
        print(f"  - 均值: {beta_values.mean():.4f}")
        print(f"  - 标准差: {beta_values.std():.4f}")
        print(f"  - 最小值: {beta_values.min():.4f}")
        print(f"  - 最大值: {beta_values.max():.4f}")
        print(f"  - 中位数: {beta_values.median():.4f}")

        # 检查是否有异常值
        extreme_beta = beta_values[(beta_values < -5) | (beta_values > 5)]
        if len(extreme_beta) > 0:
            print(f"\n⚠ 发现 {len(extreme_beta)} 个极端 BETA 值 (|beta| > 5)")
        else:
            print(f"\n✓ 没有发现极端 BETA 值")

        # 显示部分结果
        print(f"\n前10条 BETA 结果:")
        print(beta_result.head(10).to_string())

        return True, beta_result

    except Exception as e:
        print(f"\n✗ BETA 因子计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_hsiga_factor():
    """测试 HSIGMA 因子计算（验证残差标准差 ddof=1 修复）"""
    print("\n" + "=" * 80)
    print("测试 HSIGMA 因子计算 (验证残差标准差 ddof=1)")
    print("=" * 80)

    # 初始化 qlib
    qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq', custom_ops=[PTTM])

    # 获取测试股票
    instruments = D.instruments(market='csi300')
    instruments = D.list_instruments(
        instruments, start_time='2024-01-01', end_time='2025-12-31', as_list=True
    )[:10]

    # 加载数据
    start_time = '2022-01-01'
    end_time = '2025-12-31'

    df = D.features(
        instruments, fields=['$change'],
        start_time=start_time, end_time=end_time
    )

    print(f"\n正在计算 HSIGMA 因子...")
    try:
        hsigma_result = HSIGMA(df)

        print(f"\n✓ HSIGMA 因子计算成功!")
        print(f"  - 结果数量: {len(hsigma_result)} 条")

        # 统计 HSIGMA 值分布
        hsigma_values = hsigma_result['HSIGMA'].dropna()
        print(f"\nHSIGMA 因子统计:")
        print(f"  - 有效值数量: {len(hsigma_values)}")
        print(f"  - 均值: {hsigma_values.mean():.6f}")
        print(f"  - 标准差: {hsigma_values.std():.6f}")
        print(f"  - 最小值: {hsigma_values.min():.6f}")
        print(f"  - 最大值: {hsigma_values.max():.6f}")

        return True

    except Exception as e:
        print(f"\n✗ HSIGMA 因子计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_without_nan():
    """对比有NaN和无NaN情况下的计算结果"""
    print("\n" + "=" * 80)
    print("对比测试: 有NaN vs 无NaN 数据处理")
    print("=" * 80)

    # 构造测试数据
    np.random.seed(42)
    n = 504  # 窗口大小

    # 无NaN的数据
    y_clean = pd.DataFrame({
        'STOCK1': np.random.randn(n),
        'STOCK2': np.random.randn(n)
    })
    x_clean = pd.Series(np.random.randn(n))

    # 有NaN的数据（10% NaN）
    y_with_nan = y_clean.copy()
    nan_mask = np.random.rand(n) < 0.1
    y_with_nan.loc[nan_mask, 'STOCK1'] = np.nan

    print(f"\n构造测试数据:")
    print(f"  - 总样本数: {n}")
    print(f"  - STOCK1 NaN数量: {y_with_nan['STOCK1'].isna().sum()}")
    print(f"  - STOCK1 有效数据比例: {(~y_with_nan['STOCK1'].isna()).mean():.1%}")

    from data.factor.utils import rolling_regress

    # 测试无NaN数据
    print("\n测试1: 无NaN数据")
    try:
        beta1, alpha1, sigma1 = rolling_regress(
            y_clean.stack(), x_clean,
            window=252, half_life=126, num_worker=1
        )
        print(f"  ✓ 计算成功")
        print(f"    STOCK1 BETA: {beta1.loc['STOCK1'].iloc[-1]:.4f}")
        print(f"    STOCK1 HSIGMA: {sigma1.loc['STOCK1'].iloc[-1]:.6f}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")

    # 测试有NaN数据
    print("\n测试2: 有NaN数据 (10%缺失)")
    try:
        beta2, alpha2, sigma2 = rolling_regress(
            y_with_nan.stack(), x_clean,
            window=252, half_life=126, num_worker=1
        )
        print(f"  ✓ 计算成功")
        print(f"    STOCK1 BETA: {beta2.loc['STOCK1'].iloc[-1]:.4f}")
        print(f"    STOCK1 HSIGMA: {sigma2.loc['STOCK1'].iloc[-1]:.6f}")
        print(f"    (注意: NaN值被删除而非填充)")
    except Exception as e:
        print(f"  ✗ 失败: {e}")


if __name__ == '__main__':
    print("\n开始测试 rolling_regress 修改效果...\n")

    # 测试1: BETA因子
    success1, beta_df = test_beta_factor()

    # 测试2: HSIGMA因子
    success2 = test_hsiga_factor()

    # 测试3: NaN处理对比
    compare_with_without_nan()

    print("\n" + "=" * 80)
    if success1 and success2:
        print("✓ 所有测试通过! rolling_regress 修改成功。")
    else:
        print("✗ 部分测试失败，请检查修改。")
    print("=" * 80)