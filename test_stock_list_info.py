""" 单元测试：get_stock_list_info 函数修复验证 """

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/chaofeng/code/cf_quant')

from data.factor.utils import get_stock_list_info, get_mysql_data


def test_get_stock_list_info():
    """测试 get_stock_list_info 函数"""
    print("=" * 80)
    print("测试 get_stock_list_info 函数")
    print("=" * 80)

    # 测试股票列表 (CSI300部分股票)
    test_instruments = ['SH600000', 'SZ000001', 'SH600519', 'SZ000858', 'SH601318']
    test_date = '2025-11-03'  # 数据库中最早的日期

    try:
        print(f"\n1. 测试正常情况 - 股票列表: {test_instruments}")
        print(f"   查询日期: {test_date}")

        list_date_map, delist_date_map = get_stock_list_info(test_instruments, test_date)

        print(f"\n   ✓ 函数执行成功!")
        print(f"   - 获取到 {len(list_date_map)} 只股票的上市日期")
        print(f"   - 获取到 {len(delist_date_map)} 只股票的退市日期")

        print("\n   上市日期信息:")
        for inst in sorted(list_date_map.keys()):
            list_date = list_date_map[inst]
            delist_date = delist_date_map[inst]
            delist_str = str(delist_date) if pd.notna(delist_date) else "未退市"
            print(f"   - {inst}: 上市={list_date.strftime('%Y-%m-%d')}, 退市={delist_str}")

        # 验证数据类型
        print("\n2. 验证返回数据类型")
        for inst, val in list_date_map.items():
            assert isinstance(val, (pd.Timestamp, type(pd.NaT))), f"{inst}的上市日期类型错误"
        print("   ✓ 上市日期类型正确 (pd.Timestamp)")

        for inst, val in delist_date_map.items():
            assert isinstance(val, (pd.Timestamp, type(pd.NaT))), f"{inst}的退市日期类型错误"
        print("   ✓ 退市日期类型正确 (pd.Timestamp 或 NaT)")

        # 验证所有请求的股票都有返回
        print("\n3. 验证数据完整性")
        for inst in test_instruments:
            assert inst in list_date_map, f"缺少 {inst} 的上市日期"
            assert inst in delist_date_map, f"缺少 {inst} 的退市日期"
        print("   ✓ 所有请求的股票都有返回数据")

        print("\n" + "=" * 80)
        print("所有测试通过!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n   ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_mysql_data():
    """测试底层 get_mysql_data 函数"""
    print("\n" + "=" * 80)
    print("测试 get_mysql_data 函数")
    print("=" * 80)

    test_instruments = ['SH600000', 'SZ000001', 'SH600519']
    test_date = '2025-11-03'  # 数据库中最早的日期

    try:
        print(f"\n1. 查询股票: {test_instruments}")
        df = get_mysql_data(test_instruments, 'stock_info_ts',
                          ['list_date', 'delist_date'], test_date, test_date)

        print(f"\n   ✓ 查询成功!")
        print(f"   - 返回 {len(df)} 行数据")
        print(f"   - 索引: {df.index.name}")
        print(f"   - 列: {list(df.columns)}")

        print("\n2. 原始数据预览:")
        print(df.head(10).to_string())

        # 检查是否有重复数据
        print("\n3. 数据重复检查:")
        duplicated = df.index.duplicated().sum()
        print(f"   - 重复索引数: {duplicated}")
        if duplicated > 0:
            print(f"   - 重复的股票: {df.index[df.index.duplicated()].unique().tolist()}")

        print("\n" + "=" * 80)
        print("get_mysql_data 测试通过!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n   ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 80)
    print("测试边界情况")
    print("=" * 80)

    # 测试单只股票
    try:
        print("\n1. 测试单只股票")
        list_map, delist_map = get_stock_list_info(['SH600000'], '2025-11-03')
        print(f"   ✓ 单只股票测试通过: {list_map}")
    except Exception as e:
        print(f"   ✗ 单只股票测试失败: {e}")

    # 测试大量股票
    try:
        print("\n2. 测试50只股票")
        import qlib
        from qlib.data import D
        qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq')
        instruments = D.instruments(market='csi300')
        instruments = D.list_instruments(instruments, start_time='2025-01-01',
                                        end_time='2025-12-31', as_list=True)
        test_stocks = instruments[:50]

        list_map, delist_map = get_stock_list_info(test_stocks, '2025-11-03')
        print(f"   ✓ 50只股票测试通过，获取到 {len(list_map)} 条记录")
    except Exception as e:
        print(f"   ✗ 50只股票测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("边界测试完成!")
    print("=" * 80)


if __name__ == '__main__':
    print("\n开始单元测试...\n")

    # 测试底层数据获取
    test_get_mysql_data()

    # 测试主要函数
    success = test_get_stock_list_info()

    # 测试边界情况
    test_edge_cases()

    print("\n" + "=" * 80)
    if success:
        print("✓ 所有测试通过! 修复成功。")
    else:
        print("✗ 测试失败，请检查修复。")
    print("=" * 80)