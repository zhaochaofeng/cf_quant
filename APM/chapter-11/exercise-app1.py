"""
《主动投资组合管理》第11章 应用练习 第1题
计算至少两个预测信号的系数 c_g，分析相同 IC 下相对权重的含义

数据：qlib 全市场，2026-01-01 ~ 2026-05-31
信号：3 个 qlib 表达式因子 f1/f2/f3
残差收益率代理：ret = close(t+2)/close(t+1) - 1（t+2 前向收益）
残差波动率：omega_n = Std_TS{ret_n}（每股时间序列标准差）
"""

import numpy as np
import pandas as pd
from pathlib import Path

import qlib
from qlib.data import D

from utils import PickleIO

# ── 配置 ──────────────────────────────────────────────────────────────
START_DATE = "2026-01-01"
END_DATE = "2026-05-31"
OUTPUT_DIR = Path(__file__).parent / "output"

FIELDS = [
    "$close",
    "Cov($low/$close, EMA($close/Ref($close,1)-1, 20)/Std($close/Ref($close,1)-1, 20), 20)",
    "EMA(Power(($close-$open)/$open + ($high-Greater($open,$close))/$open, 3), 20)",
    "EMA(($close/Ref($close,1)-1)*($volume/Ref($volume,1)-1) - ($open-Ref($close, 1)) / (Ref($close, 1) + 1e-12), 20)",
]
SIGNAL_NAMES = ["f1", "f2", "f3"]

# ── Qlib 初始化 ─────────────────────────────────────────────────────────
qlib.init(provider_uri="~/.qlib/qlib_data/custom_data_hfq", kernels=1)

# ── Step 1: 数据准备 ──────────────────────────────────────────────────
print("[Step 1] 获取 qlib 全市场因子数据...")

config = D.instruments(market="all")
instruments = D.list_instruments(config, START_DATE, END_DATE)
df = D.features(instruments, FIELDS, START_DATE, END_DATE)
df.columns = ["close"] + SIGNAL_NAMES
df["ret"] = df["close"].groupby("instrument", group_keys=False).apply(
    lambda x: x.shift(-2) / x.shift(-1) - 1
)
df.dropna(inplace=True, how="any")

n_stocks = df.index.get_level_values("instrument").nunique()
n_days = df.index.get_level_values("datetime").nunique()
print(f"  股票数: {n_stocks}, 交易日数: {n_days}")
PickleIO.write(df, OUTPUT_DIR / "df.pkl")

# ── Step 2: 残差波动率 ω_n ────────────────────────────────────────────
print("\n[Step 2] 估计残差波动率 ω_n = Std_TS{ret_n}...")

omega = df["ret"].groupby("instrument").std(ddof=1)
PickleIO.write(omega, OUTPUT_DIR / "omega.pkl")

print(f"  ω_n 范围: [{omega.min():.6f}, {omega.max():.6f}]")
print(f"  ω_n 均值: {omega.mean():.6f}")

# ── Step 3: 计算 c_g ──────────────────────────────────────────────────
print("\n[Step 3] 逐日计算横截面 c_g，再对时间平均...")


def compute_cg(g_t, omega):
    """计算单日 c_g
    g_t:   pd.Series, 当日各股信号值 (MultiIndex: instrument, datetime)
    omega: pd.Series, 各股残差波动率 (index=instrument)
    """
    g = g_t.droplevel("datetime")       # index 降为 instrument
    w = omega.reindex(g.index)          # 对齐当日截面
    std_g = g.std(ddof=1)               # Std_CS{g_n}
    std_norm = (g / w).std(ddof=1)      # Std_CS{g_n / ω_n}
    return std_g / std_norm


cg = {}
for name in SIGNAL_NAMES:
    cg_t = df[name].groupby(level="datetime").apply(
        lambda s: compute_cg(s, omega)
    ).dropna()
    cg[name] = cg_t.mean()
    print(f"  {name}: c_g = {cg[name]:.6f}  (有效天数 {len(cg_t)}/{n_days})")

cg = pd.Series(cg, name="c_g")
PickleIO.write(cg, OUTPUT_DIR / "cg.pkl")

# ── Step 4: 相对权重分析 ──────────────────────────────────────────────
print("\n[Step 4] 相同 IC 假设下的相对权重（w_i/w_j = c_g_i/c_g_j）...")

for i in range(len(SIGNAL_NAMES)):
    for j in range(i + 1, len(SIGNAL_NAMES)):
        a, b = SIGNAL_NAMES[i], SIGNAL_NAMES[j]
        ratio = cg[a] / cg[b]
        print(f"  w({a})/w({b}) = {ratio:.4f}")

# ── 结果摘要 ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("结果摘要")
print("=" * 60)
for name in SIGNAL_NAMES:
    print(f"  {name}: c_g = {cg[name]:.6f}")
dominant = cg.idxmax()
print(f"\n相同 IC 下权重最大的信号: {dominant} (c_g 最大)")
print("结论: 相同 IC 不代表等权，信号相对权重按 c_g 比例分配")

print(f"\n中间结果已保存至: {OUTPUT_DIR}/")
print("  df.pkl    全市场因子数据 [close, f1, f2, f3, ret]")
print("  omega.pkl 各股残差波动率 [N]")
print("  cg.pkl    各信号 c_g [3]")