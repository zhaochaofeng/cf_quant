



import pandas as pd
import numpy as np
import qlib
from qlib.data import D
qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq')
df = D.features(['SZ000001'], fields=['P($$revenue_a)', 'P($$revenue_q)'], start_time='2025-01-01', end_time='2025-06-01')


# ===== 测试 WLS 索引对齐 =====
from utils.stats import WLS

def test_wls_index_align():
    """测试 WLS 索引对齐功能"""
    print('=' * 60)
    print('测试 WLS 索引对齐')
    print('=' * 60)

    # 1. 正常顺序 - 应该正常工作
    print('\n1. 正常顺序（索引一致）:')
    y_norm = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=[0, 1, 2, 3, 4])
    X_norm = pd.DataFrame({'x1': [0.1, 0.2, 0.3, 0.4, 0.5]}, index=[0, 1, 2, 3, 4])
    slope_norm, alpha_norm, resid_norm = WLS(y_norm, X_norm)
    print(f'  slope: {slope_norm.values}')
    print(f'  alpha: {alpha_norm:.4f}')
    print(f'  resid index: {resid_norm.index.tolist()}')
    print(f'  预期值约为 slope=10, alpha=0')
    print(f'  PASS')

    # 2. 索引顺序错乱 - 测试对齐
    print('\n2. 索引顺序不一致:')
    y_mis = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=[4, 3, 2, 1, 0])
    X_mis = pd.DataFrame({'x1': [0.1, 0.2, 0.3, 0.4, 0.5]}, index=[0, 1, 2, 3, 4])

    slope_mis, alpha_mis, resid_mis = WLS(y_mis, X_mis)

    # 对齐后实际数据应为 y=[1,2,3,4,5], X=[0.1,0.2,0.3,0.4,0.5]
    print(f'  slope: {slope_mis.values}')
    print(f'  alpha: {alpha_mis:.4f}')
    print(f'  resid index: {resid_mis.index.tolist()}')
    if abs(slope_mis.values[0] - slope_norm.values[0]) < 1e-6 and abs(alpha_mis - alpha_norm) < 1e-6:
        print(f'  PASS (与正常顺序结果一致)')
    else:
        print(f'  FAIL (与正常顺序结果不一致)')

    # 3. 索引部分重叠
    print('\n3. 索引部分重叠:')
    y_part = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    X_part = pd.DataFrame({'x1': [0.1, 0.2, 0.3, 0.4, 0.5]}, index=[0, 1, 2, 3, 4])
    slope_part, alpha_part, resid_part = WLS(y_part, X_part)
    print(f'  slope: {slope_part.values}')
    print(f'  alpha: {alpha_part:.4f}')
    print(f'  resid index: {resid_part.index.tolist()}')
    print(f'  预期只有3行 (索引 0,1,2)')
    if len(resid_part) == 3:
        print(f'  PASS')
    else:
        print(f'  FAIL')

    # 4. numpy array 输入（无索引） - 不应出错
    print('\n4. numpy array 输入:')
    y_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X_arr = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
    slope_arr, alpha_arr, resid_arr = WLS(y_arr, X_arr, backend='numpy')
    print(f'  slope: {slope_arr.values}')
    print(f'  alpha: {alpha_arr:.4f}')
    near_10 = abs(slope_arr.values[0] - 10) < 0.5
    print(f'  slope接近10: {near_10}')
    if near_10:
        print(f'  PASS')
    else:
        print(f'  FAIL')

    # 5. 索引顺序不一致 + numpy后端
    print('\n5. 索引顺序不一致 + numpy后端:')
    y_mis2 = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=[4, 3, 2, 1, 0])
    X_mis2 = pd.DataFrame({'x1': [0.1, 0.2, 0.3, 0.4, 0.5]}, index=[0, 1, 2, 3, 4])
    slope_mis2, alpha_mis2, resid_mis2 = WLS(y_mis2, X_mis2, backend='numpy')
    print(f'  slope: {slope_mis2.values}')
    print(f'  alpha: {alpha_mis2:.4f}')
    print(f'  resid index: {resid_mis2.index.tolist()}')
    if abs(slope_mis2.values[0] - slope_norm.values[0]) < 1e-6 and abs(alpha_mis2 - alpha_norm) < 1e-6:
        print(f'  PASS (与正常顺序结果一致)')
    else:
        print(f'  FAIL (与正常顺序结果不一致)')

    # 6. 索引完全无重叠
    print('\n6. 索引完全无重叠:')
    y_no = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
    X_no = pd.DataFrame({'x1': [0.1, 0.2, 0.3]}, index=[0, 1, 2])
    try:
        slope_no, alpha_no, resid_no = WLS(y_no, X_no)
        print(f'  FAIL (预期 ValueError，但未抛出)')
    except ValueError as e:
        print(f'  抛出 ValueError: {e}')
        print(f'  PASS')

    print('\n' + '=' * 60)
    print('测试完成')
    print('=' * 60)


if __name__ == '__main__':
    test_wls_index_align()






