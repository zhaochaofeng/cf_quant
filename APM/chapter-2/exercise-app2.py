"""
《主动投资组合管理》第2章 应用练习 第2题
CAPM最优风险组合：最大化 f_P - λσ²_P 的全额投资有效组合

方法A：直接拉格朗日求解
方法B：两基金分离(式2A-45)
比较两种方法验证一致性

数据：csi300成分股，与第1题相同的股票集合
频率说明：V 和 σ² 为日度；f_Q=6% 为年化；λ=6/σ²_Q_annual 统一转换为日度等价形式
"""

import numpy as np
from pathlib import Path

import qlib
from qlib.data import D

from utils import PickleIO

# ── 配置 ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "output"
OUTPUT_DIR = Path(__file__).parent / "output2"
END_DATE = "2026-05-14"
TRADING_DAYS = 252

FQ_ANNUAL = 0.06       # CAPMMI 年化预期超额收益率 6%

# ── Qlib 初始化 ─────────────────────────────────────────────────────────
qlib.init(provider_uri="~/.qlib/qlib_data/custom_data_hfq", kernels=1)

# ── Step 1: 加载第1题结果 ────────────────────────────────────────────
print("[Step 1] 加载第1题计算结果...")

V = PickleIO.read(DATA_DIR / "V.pkl")          # 日度协方差矩阵
h_c = PickleIO.read(DATA_DIR / "h_c.pkl")       # 组合C权重
var_c = PickleIO.read(DATA_DIR / "var_c.pkl")   # 日度方差 σ²_c
stock_list = PickleIO.read(DATA_DIR / "stock_list.pkl")

N = V.shape[0]
e = np.ones(N)
std_c = np.sqrt(var_c)
print(f"  股票数: {N}")
print(f"  σ²_c(日度) = {var_c:.8f}")
print(f"  σ_c = {std_c * 100:.4f}%/day, 年化 {std_c * np.sqrt(TRADING_DAYS) * 100:.2f}%")
PickleIO.write(stock_list, OUTPUT_DIR / "stock_list.pkl")

# ── Step 2: 获取市值数据，构建组合Q ───────────────────────────────────
print("\n[Step 2] 获取流通市值 $circ_mv，构建 CAPMMI 权重...")

mkt_cap_r = D.features(
    instruments=stock_list,
    fields=["$circ_mv"],
    start_time=END_DATE,
    end_time=END_DATE,
)
mkt_cap = mkt_cap_r.unstack(level="instrument")["$circ_mv"].iloc[-1]
PickleIO.write(mkt_cap, OUTPUT_DIR / "mkt_cap.pkl")

h_Q = (mkt_cap / mkt_cap.sum()).values  # 归一化，单位无关（$circ_mv 单位为万元）
PickleIO.write(h_Q, OUTPUT_DIR / "h_Q.pkl")
print(f"  h_Q range: [{h_Q.min():.6f}, {h_Q.max():.6f}]")
print(f"  h_Q sum = {h_Q.sum():.10f}")

# ── Step 3: 组合Q方差 & 风险厌恶系数 ──────────────────────────────────
print("\n[Step 3] 计算组合Q方差和风险厌恶系数...")

var_Q = h_Q @ V @ h_Q              # 日度方差 σ²_Q
std_Q = np.sqrt(var_Q)
var_Q_annual = var_Q * TRADING_DAYS  # 年化方差
lamb = FQ_ANNUAL / var_Q_annual     # λ = 6 / σ²_Q(年化)
lamb_daily = lamb                   # λ 与频率无关

PickleIO.write(var_Q, OUTPUT_DIR / "var_Q.pkl")
PickleIO.write(var_Q_annual, OUTPUT_DIR / "var_Q_annual.pkl")
PickleIO.write(lamb, OUTPUT_DIR / "lambda.pkl")
print(f"  σ²_Q(日度) = {var_Q:.8f}")
print(f"  σ²_Q(年化) = {var_Q_annual:.6f}")
print(f"  σ_Q = {std_Q * 100:.4f}%/day, 年化 {std_Q * np.sqrt(TRADING_DAYS) * 100:.2f}%")
print(f"  λ = 6 / σ²_Q(年化) = {lamb:.4f}")

# ── Step 4: CAPM预期超额收益率向量 ────────────────────────────────────
print("\n[Step 4] 计算 β_Q 和 CAPM 预期超额收益率向量 f...")

beta_Q = V @ h_Q / var_Q            # 每只股票对 CAPMMI 的 β
PickleIO.write(beta_Q, OUTPUT_DIR / "beta_Q.pkl")

FQ_DAILY = FQ_ANNUAL / TRADING_DAYS  # 日度 f_Q
f = beta_Q * FQ_DAILY                # CAPM 日度预期超额收益率
PickleIO.write(f, OUTPUT_DIR / "f.pkl")

print(f"  β_Q range: [{beta_Q.min():.4f}, {beta_Q.max():.4f}]")
print(f"  β_Q 加权平均 = {h_Q @ beta_Q:.6f}  (应接近 1)")
print(f"  f_Q(日度) = {FQ_DAILY * 100:.4f}%/day")

# ── Step 5: 最优预期收益率 f* ────────────────────────────────────────
print("\n[Step 5] 效用最大化 → 最优预期收益率 f*...")

# 有效前沿参数：使用年化值（或日度值），结果 w 与频率无关
var_c_annual = var_c * TRADING_DAYS
f_C_annual = FQ_ANNUAL * var_c_annual / var_Q_annual  # 式(2A-35)
kappa_annual = (var_Q_annual - var_c_annual) / (FQ_ANNUAL - f_C_annual) ** 2  # 式(2A-48)
f_star_annual = f_C_annual + 1.0 / (2.0 * lamb * kappa_annual)  # 附录练习4

PickleIO.write(f_C_annual, OUTPUT_DIR / "f_C_annual.pkl")
PickleIO.write(kappa_annual, OUTPUT_DIR / "kappa_annual.pkl")
PickleIO.write(f_star_annual, OUTPUT_DIR / "f_star_annual.pkl")

f_C_daily = f_C_annual / TRADING_DAYS
f_star_daily = f_star_annual / TRADING_DAYS
print(f"  f_C = {f_C_annual * 100:.4f}%/年 = {f_C_daily * 100:.6f}%/day")
print(f"  κ(年化) = {kappa_annual:.4f}")
print(f"  f* = {f_star_annual * 100:.4f}%/年 = {f_star_daily * 100:.6f}%/day")

# ── Step 6: 方法A —— 直接拉格朗日 ────────────────────────────────────
print("\n[Step 6] 方法A: 直接拉格朗日求解 h_P*...")

# 使用日度 V 和日度 f，λ 不变（频率无关）
# V^{-1} @ f
v_inv_f = np.linalg.solve(V, f)
# 最优组合。(e @ v_inv_f) 返回的是标量，所以后面用 "* h_c"
h_P_A = h_c + (1.0 / (2.0 * lamb)) * (v_inv_f - (e @ v_inv_f) * h_c)
PickleIO.write(h_P_A, OUTPUT_DIR / "h_P_A.pkl")

print(f"  h_P*^(A) range: [{h_P_A.min():.6f}, {h_P_A.max():.6f}]")
print(f"  sum(h_P*^(A)) = {h_P_A.sum():.10f}")

# ── Step 7: 方法B —— 两基金分离(2A-45) ───────────────────────────────
print("\n[Step 7] 方法B: 两基金分离(式2A-45)求 h_P*...")

w = (FQ_ANNUAL - f_star_annual) / (FQ_ANNUAL - f_C_annual)
h_P_B = w * h_c + (1 - w) * h_Q
PickleIO.write(w, OUTPUT_DIR / "w.pkl")
PickleIO.write(h_P_B, OUTPUT_DIR / "h_P_B.pkl")

print(f"  w = {w:.6f}  (组合C权重)")
print(f"  1-w = {1-w:.6f}  (组合Q权重)")
print(f"  h_P*^(B) range: [{h_P_B.min():.6f}, {h_P_B.max():.6f}]")
print(f"  sum(h_P*^(B)) = {h_P_B.sum():.10f}")

# ── Step 8: 比较方法A与方法B ─────────────────────────────────────────
print("\n[Step 8] 比较方法A与方法B...")

diff = np.max(np.abs(h_P_A - h_P_B))
print(f"  max|h_P*^(A) - h_P*^(B)| = {diff:.2e}")
if np.allclose(h_P_A, h_P_B, atol=1e-10):
    print("  ✓ 两种方法一致，式(2A-45)的两基金分离性质成立")
else:
    print(f"  △ 存在差异 {diff:.2e}")

# 解析验证：CAPM假设下 V^{-1}f = (f_Q/σ²_Q) h_Q
# 由于 f = β_Q × f_Q(日度) = (V h_Q/var_Q) × f_Q(日度)
# V^{-1}f = V^{-1}(V h_Q/var_Q) × f_Q(日度) = (f_Q/var_Q) h_Q
v_inv_f_pred = (FQ_DAILY / var_Q) * h_Q
diff_pred = np.max(np.abs(v_inv_f - v_inv_f_pred))
print(f"  max|V⁻¹f - (f_Q/σ²_Q)h_Q| = {diff_pred:.2e}  (CAPM解析验证)")

# ── Step 9: 组合P*的贝塔和预期超额收益率 ─────────────────────────────
print("\n[Step 9] 计算组合P*的贝塔和预期超额收益率...")

h_P = h_P_B  # 两种方法一致，任选其一

# 式(2A-45) 代入 β_P = h_P^T V h_Q / σ²_Q 化简得:
beta_P = w * var_c / var_Q + (1.0 - w)
PickleIO.write(beta_P, OUTPUT_DIR / "beta_P.pkl")

f_P_daily = beta_P * FQ_DAILY
f_P_annual = f_P_daily * TRADING_DAYS
PickleIO.write(f_P_daily, OUTPUT_DIR / "f_P_daily.pkl")

# 直接计算交叉验证
beta_P_direct = h_P @ V @ h_Q / var_Q
f_P_daily_direct = h_P @ f

print(f"  β_P* = {beta_P:.6f}  (化简公式)")
print(f"  β_P* = {beta_P_direct:.6f}  (直接计算)")
print(f"  f_P* = {f_P_annual * 100:.4f}%/年 = {f_P_daily * 100:.6f}%/day  (β×f_Q)")
print(f"  f_P* = {f_P_daily_direct * TRADING_DAYS * 100:.4f}%/年  (h_P·f 直接计算)")
print(f"  f*  = {f_star_annual * 100:.4f}%/年  (效用最大化目标)")

# ── 结果摘要 ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("结果摘要")
print("=" * 60)
print(f"\n组合Q (CAPMMI 市值加权):")
print(f"  σ_Q = {std_Q * np.sqrt(TRADING_DAYS) * 100:.2f}% 年化")
print(f"  f_Q = 6.00%/年")

print(f"\n组合C (最小方差):")
print(f"  σ_C = {std_c * np.sqrt(TRADING_DAYS) * 100:.2f}% 年化")
print(f"  f_C = {f_C_annual * 100:.4f}%/年 (CAPM一致预期)")

print(f"\n组合P* (最优风险, λ = {lamb:.2f}):")
print(f"  构成: w(C)={w:.4f} + (1-w)(Q)={1-w:.4f}")
sigma_P = np.sqrt(h_P @ V @ h_P)
print(f"  β_P* = {beta_P:.4f}")
print(f"  f_P* = {f_P_annual * 100:.4f}%/年")
print(f"  σ_P* = {sigma_P * np.sqrt(TRADING_DAYS) * 100:.2f}% 年化")

print(f"\n问题 a) 答案:")
print(f"  组合P*的β = {beta_P:.4f}")
print(f"  组合P*的预期超额收益率 f_P* = {f_P_annual * 100:.4f}%/年")

print(f"\n问题 b) 验证:")
print(f"  方法A(拉格朗日) vs 方法B(2A-45): max|diff| = {diff:.2e}")
print(f"  P* = {w:.4f} × C + {1-w:.4f} × Q")

print(f"\n中间结果已保存至: {OUTPUT_DIR}/")
for name in sorted(OUTPUT_DIR.glob("*.pkl")):
    print(f"  {name.name}")
