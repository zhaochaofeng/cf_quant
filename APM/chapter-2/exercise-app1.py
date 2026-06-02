"""
《主动投资组合管理》第2章 应用练习 第1题
构建最小方差全额投资组合（组合C），检验式(2A-16): e = V h_c / σ²_c

数据：csi300 成分股（最新一期），近2年日度 $change（对数收益率）
"""

import numpy as np
import pandas as pd
from pathlib import Path

import qlib
from qlib.data import D

from utils import PickleIO

# ── 配置 ──────────────────────────────────────────────────────────────
START_DATE = "2024-06-01"
END_DATE = "2026-05-14"
OUTPUT_DIR = Path(__file__).parent / "output"

# ── Qlib 初始化 ─────────────────────────────────────────────────────────
qlib.init(provider_uri="~/.qlib/qlib_data/custom_data_hfq", kernels=1)

# ── Step 1: 获取 CSI300 成分股收益率数据 ──────────────────────────────
print("[Step 1] 获取 CSI300 成分股日度收益率数据...")

config = D.instruments(market="csi300")
instruments = D.list_instruments(config, start_time=END_DATE, end_time=END_DATE, as_list=True)

r = D.features(
    instruments=instruments[0:20],
    fields=["$change"],
    start_time=START_DATE,
    end_time=END_DATE,
)
# r: MultiIndex (datetime, instrument), column: $change

# 透视：index=datetime, columns=instrument, values=$change
R = r.unstack(level="instrument")["$change"]

# 删除全 NaN 的列（可能有退市/停牌股）
R = R.dropna(axis=1, how="any")
# 删除全 NaN 的行
R = R.dropna(axis=0, how="any")

n_stocks = R.shape[1]
n_days = R.shape[0]
print(f"  股票数: {n_stocks}, 交易日数: {n_days}")
PickleIO.write(R, OUTPUT_DIR / "R.pkl")

# ── Step 2: 协方差矩阵 V ─────────────────────────────────────────────
print("\n[Step 2] 计算协方差矩阵 V [N×N]...")

V = np.cov(R.T, ddof=1)  # [N×N]，无偏估计
PickleIO.write(V, OUTPUT_DIR / "V.pkl")
PickleIO.write(R.columns.tolist(), OUTPUT_DIR / "stock_list.pkl")

# ── Step 3: 组合 C 权重 h_c ──────────────────────────────────────────
print("\n[Step 3] 求解最小方差全额投资组合权重 h_c...")

N = V.shape[0]
e = np.ones(N)

# 解线性方程 V @ x = e，比 V^{-1} @ e 更数值稳定
v_inv_e = np.linalg.solve(V, e)
denom = e @ v_inv_e  # 标量 e^T V^{-1} e

h_c = v_inv_e / denom  # [N×1]
PickleIO.write(h_c, OUTPUT_DIR / "h_c.pkl")

print(f"  权重范围: [{h_c.min():.6f}, {h_c.max():.6f}]")
print(f"  做空比例: {h_c[h_c < 0].sum():.4f}")
print(f"  权重之和: {h_c.sum():.10f}")

# ── Step 4: 组合 C 方差 σ²_c ─────────────────────────────────────────
print("\n[Step 4] 计算组合 C 方差...")

var_c = h_c @ V @ h_c  # 标量
std_c = np.sqrt(var_c)
PickleIO.write(var_c, OUTPUT_DIR / "var_c.pkl")

print(f"  方差 σ²_c = {var_c:.10f}")
print(f"  标准差 σ_c = {std_c:.6f}  ({std_c * 100:.4f}%/day)")
print(f"  年化波动率 = {std_c * np.sqrt(252) * 100:.2f}%")

# ── Step 5: 每只股票对 C 的贝塔 β_c ──────────────────────────────────
print("\n[Step 5] 计算每只成分股对组合 C 的贝塔...")

beta_c = V @ h_c / var_c  # [N×1]
PickleIO.write(beta_c, OUTPUT_DIR / "beta_c.pkl")

# ── Step 6: 检验式(2A-16) ────────────────────────────────────────────
print("\n[Step 6] 检验式(2A-16): e = V h_c / σ²_c ...")

max_dev = np.max(np.abs(beta_c - e))
mean_dev = np.mean(np.abs(beta_c - e))
print(f"  max|β_c − 1| = {max_dev:.2e}")
print(f"  mean|β_c − 1| = {mean_dev:.2e}")

if np.allclose(beta_c, e, atol=1e-10):
    print("  ✓ 检验通过：β_c ≈ e，式(2A-16)成立")
else:
    # 不通过可能是因为数值误差或数据问题
    print(f"  △ 最大偏差 {max_dev:.2e}")

# ── 结果摘要 ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("结果摘要")
print("=" * 60)
top_long = np.argsort(h_c)[-5:][::-1]
top_short = np.argsort(h_c)[:5]
print("\n权重最大的5只（多头）:")
for i in top_long:
    print(f"  {R.columns[i]:12s}  h={h_c[i]:+.6f}")
print("\n权重最小的5只（空头）:")
for i in top_short:
    print(f"  {R.columns[i]:12s}  h={h_c[i]:+.6f}")

print(f"\n年化波动率: {std_c * np.sqrt(252) * 100:.2f}%")
print(f"\n中间结果已保存至: {OUTPUT_DIR}/")
print("  R.pkl       收益率矩阵 [T×N]")
print("  V.pkl       协方差矩阵 [N×N]")
print("  h_c.pkl     组合C权重 [N]")
print("  var_c.pkl   组合C方差 (标量)")
print("  beta_c.pkl  每只股票对C的贝塔 [N]")
print("  stock_list.pkl  成分股列表 [N]")
