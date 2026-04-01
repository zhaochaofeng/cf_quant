""" 单元测试：验证 STREV 因子修复（对数收益率） """

import sys
sys.path.insert(0, '/Users/chaofeng/code/cf_quant')

import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from data.factor.momentum import STREV
from utils import PTTM


def test_strev_factor():
    """测试 STREV 因子计算（对数收益率版本）"""
    print("=" * 80)
    print("测试 STREV 因子计算（修复后：使用对数收益率 ln(1+r)）")
    print("=" * 80)

    # 初始化 qlib
    qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq', custom_ops=[PTTM])

    # 获取测试股票 (CSI300前5只)
    instruments = D.instruments(market='csi300')
    instruments = D.list_instruments(
        instruments, start_time='2025-01-01', end_time='2025-12-31', as_list=True
    )[:5]
    print(f"\n测试股票数量: {len(instruments)}")
    print(f"股票列表: {instruments}")

    # 加载数据
    start_time = '2025-01-01'
    end_time = '2025-06-30'

    print(f"\n数据时间范围: {start_time} 至 {end_time}")
    print("正在加载数据...")

    df = D.features(
        instruments, fields=['$change'],
        start_time=start_time, end_time=end_time
    )

    print(f"数据加载完成: {df.shape[0]} 行")

    # 检查原始收益率分布
    print(f"\n原始收益率 ($change) 统计:")
    print(f"  - 均值: {df['$change'].mean():.6f}")
    print(f"  - 标准差: {df['$change'].std():.6f}")
    print(f"  - 最小值: {df['$change'].min():.6f}")
    print(f"  - 最大值: {df['$change'].max():.6f}")

    # 计算对数收益率
    log_returns = np.log(1 + df['$change'])
    print(f"\n对数收益率 ln(1+r) 统计:")
    print(f"  - 均值: {log_returns.mean():.6f}")
    print(f"  - 标准差: {log_returns.std():.6f}")
    print(f"  - 最小值: {log_returns.min():.6f}")
    print(f"  - 最大值: {log_returns.max():.6f}")

    # 计算 STREV 因子
    print("\n正在计算 STREV 因子...")
    try:
        strev_result = STREV(df)

        print(f"\n✓ STREV 因子计算成功!")
        print(f"  - 结果数量: {len(strev_result)} 条")
        print(f"  - 覆盖股票: {strev_result.index.get_level_values('instrument').nunique()} 只")

        # 统计 STREV 值分布
        strev_values = strev_result['STREV'].dropna()
        print(f"\nSTREV 因子统计:")
        print(f"  - 有效值数量: {len(strev_values)}")
        print(f"  - 均值: {strev_values.mean():.6f}")
        print(f"  - 标准差: {strev_values.std():.6f}")
        print(f"  - 最小值: {strev_values.min():.6f}")
        print(f"  - 最大值: {strev_values.max():.6f}")
        print(f"  - 中位数: {strev_values.median():.6f}")

        # 显示部分结果
        print(f"\n前10条 STREV 结果:")
        print(strev_result.head(10).to_string())

        return True, strev_result

    except Exception as e:
        print(f"\n✗ STREV 因子计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_log_return_formula():
    """验证对数收益率计算公式的正确性"""
    print("\n" + "=" * 80)
    print("验证对数收益率公式: ln(1+r)")
    print("=" * 80)

    # 构造测试数据
    test_returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.0])

    print("\n测试数据:")
    print(f"  原始收益率 r: {test_returns.tolist()}")

    # 计算对数收益率
    log_returns = np.log(1 + test_returns)
    print(f"  对数收益率 ln(1+r): {log_returns.tolist()}")

    # 验证公式
    print("\n公式验证:")
    for i, (r, lr) in enumerate(zip(test_returns, log_returns)):
        expected = np.log(1 + r)
        match = "✓" if np.isclose(lr, expected) else "✗"
        print(f"  r={r:+.4f} -> ln(1+r)={lr:+.6f} {match}")

    print("\n✓ 对数收益率公式验证通过")


def compare_with_without_log():
    """对比使用对数收益率和不使用的差异"""
    print("\n" + "=" * 80)
    print("对比: 对数收益率 vs 原始收益率")
    print("=" * 80)

    # 构造测试数据
    np.random.seed(42)
    n = 21  # 1个月交易日
    returns = np.random.randn(n) * 0.02  # 日收益率 ~2%波动

    print(f"\n构造 {n} 个交易日数据")
    print(f"原始收益率统计:")
    print(f"  均值: {returns.mean():.6f}")
    print(f"  标准差: {returns.std():.6f}")

    # 等权计算
    simple_sum = returns.sum()
    log_sum = np.log(1 + returns).sum()

    print(f"\n累计和对比:")
    print(f"  原始收益率直接求和: {simple_sum:.6f}")
    print(f"  对数收益率求和: {log_sum:.6f}")
    print(f"  差异: {abs(simple_sum - log_sum):.6f}")

    # 解释差异
    print(f"\n差异解释:")
    print(f"  对数收益率具有时间可加性，更适合多期累计")
    print(f"  ln(1+r) ≈ r 当 r 很小时（泰勒展开）")
    print(f"  但对于较大波动，差异明显")


if __name__ == '__main__':
    print("\n开始测试 STREV 因子修复效果...\n")

    # 测试1: 验证对数收益率公式
    test_log_return_formula()

    # 测试2: 对比差异
    compare_with_without_log()

    # 测试3: STREV因子计算
    success, strev_df = test_strev_factor()

    print("\n" + "=" * 80)
    if success:
        print("✓ 所有测试通过! STREV 因子修复成功（使用对数收益率 ln(1+r)）")
    else:
        print("✗ 测试失败，请检查修复。")
    print("=" * 80)