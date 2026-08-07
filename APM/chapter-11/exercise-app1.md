# 应用练习第1题：计算系数 $c_g$

## 目标

为至少两个预测信号计算系数 $c_g$。这需要一组横截面预测信号和残差波动率。如果这两个预测信号具有相同的 $IC$，那么这对它们的相对权重意味着什么？

**待验证结论**：相同 IC 的两个信号，其相对权重与 $c_g$ 成正比（$w_1/w_2 = c_g^{(1)}/c_g^{(2)}$），而非等权分配。

---

## 理论回顾（§11.3 情形2）

### 情形2：时间序列信号波动率与资产波动率成比例

$$
\text{Std}_{TS}\{g_n\} = c_2 \cdot \omega_n \tag{11-8}
$$

其中常数 $c_2$ 可用横截面数据估计（式 11-10）：

$$
c_2 = \text{Std}_{CS}\left\{\frac{g_n}{\omega_n}\right\} \tag{11-10}
$$

### 精炼预测与系数 $c_g$（式 11-13 → 11-14）

$$
\phi_n = IC \cdot c_g \cdot z_{CS,n} \tag{11-14}
$$

其中系数 $c_g$ 定义为：

$$
c_g = \frac{\text{Std}_{CS}\{g_n\}}{\text{Std}_{CS}\{g_n/\omega_n\}}
$$

$c_g$ 具有日收益率量纲：分子 $\text{Std}_{CS}\{g_n\}$ 是原始信号的横截面波动，分母是波动率归一化后信号的横截面波动。$c_g$ 越大，单位 $IC \cdot z_{CS}$ 产生的 alpha 越大。

---

## Step 1: 数据准备（qlib）

用 3 个 qlib 表达式因子作为横截面预测信号 $f_1, f_2, f_3$，用 t+2 前向收益 `ret` 作为残差收益率的代理。

```python
import qlib
from qlib.data import D

start_date = '2026-01-01'
end_date = '2026-05-31'
fields = ['$close',
          'Cov($low/$close, EMA($close/Ref($close,1)-1, 20)/Std($close/Ref($close,1)-1, 20), 20)',
          'EMA(Power(($close-$open)/$open + ($high-Greater($open,$close))/$open, 3), 20)',
          'EMA(($close/Ref($close,1)-1)*($volume/Ref($volume,1)-1) - ($open-Ref($close, 1)) / (Ref($close, 1) + 1e-12), 20)']

qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq')
config = D.instruments(market='all')
instruments = D.list_instruments(config, start_date, end_date)
df = D.features(instruments, fields, start_date, end_date)
df.columns = ['close', 'f1', 'f2', 'f3']
df['ret'] = df['close'].groupby('instrument', group_keys=False).apply(lambda x: x.shift(-2)/x.shift(-1) - 1)
df.dropna(inplace=True, how='any')
```

数据格式：MultiIndex (datetime, instrument)，列 `[close, f1, f2, f3, ret]`。

## Step 2: 残差波动率 $\omega_n$

残差波动率 $\omega_n$ 由 `ret`（残差收益率代理）的**时间序列标准差**估计，每只股票一个常数：

$$
\omega_n = \text{Std}_{TS}\{\text{ret}_n\}
$$

```python
omega = df['ret'].groupby('instrument').std(ddof=1)
```

**简化说明**：此处用全窗口时间序列 std 作为 $\omega_n$。严格做法按滚动窗口重估（参照 `barra/alpha/config.py` 的 `RESIDUAL_VOL_WINDOW=400`），本练习中全窗口估计足以验证 $c_g$ 的相对大小。

## Step 3: 计算 $c_g$

对每个信号 $k$（f1/f2/f3）、每个交易日 $t$：

1. 横截面波动：$\text{Std}_{CS}\{g_n^{(k)}(t)\}$
2. 波动率归一化信号的横截面波动：$\text{Std}_{CS}\{g_n^{(k)}(t)/\omega_n\}$
3. 当日记为：$c_g^{(k)}(t) = \dfrac{\text{Std}_{CS}\{g_n^{(k)}(t)\}}{\text{Std}_{CS}\{g_n^{(k)}(t)/\omega_n\}}$
4. 对时间取平均，得该信号的 $c_g^{(k)}$

## Step 4: 相对权重的含义

两个信号 1、2 具有相同 IC 时，各自精炼 alpha：

$$
\phi_n^{(1)} = IC \cdot c_g^{(1)} \cdot z_{CS,n}^{(1)}, \quad
\phi_n^{(2)} = IC \cdot c_g^{(2)} \cdot z_{CS,n}^{(2)}
$$

合成 alpha $\phi_n = \phi_n^{(1)} + \phi_n^{(2)}$ 时，两信号对 alpha 的贡献比为：

$$
\frac{w_1}{w_2} = \frac{c_g^{(1)}}{c_g^{(2)}}
$$

**结论**：相同 IC 并不意味着等权。$c_g$ 大的信号在单位 z-score 下产生更多 alpha，获得更高权重。$c_g$ 衡量了信号横截面分散度在波动率归一化后的保留程度。

---

## 对应 numpy 实现

```python
import numpy as np
import pandas as pd

def compute_cg(g_t, omega):
    """计算单日 c_g
    g_t:   pd.Series, 当日各股信号值 (MultiIndex: instrument, datetime)
    omega: pd.Series, 各股残差波动率 (index=instrument)
    """
    g = g_t.droplevel('datetime')        # index 降为 instrument
    w = omega.reindex(g.index)            # 对齐当日截面
    std_g = g.std(ddof=1)                 # Std_CS{g_n}
    std_norm = (g / w).std(ddof=1)        # Std_CS{g_n / ω_n}
    return std_g / std_norm

for name in ['f1', 'f2', 'f3']:
    cg_t = df[name].groupby(level='datetime').apply(lambda s: compute_cg(s, omega))
    cg = cg_t.mean()
    print(f"{name}: c_g = {cg:.6f}")

# 权重比（假设 IC 相同）
cg_1 = df['f1'].groupby(level='datetime').apply(lambda s: compute_cg(s, omega)).mean()
cg_2 = df['f2'].groupby(level='datetime').apply(lambda s: compute_cg(s, omega)).mean()
print(f"权重比 w1/w2 = {cg_1 / cg_2:.4f}")
```

---

## 核心直觉

$c_g$ 是"信号横截面分散度 ÷ 波动率归一化后的分散度"。若信号在原始横截面上分散大、但归一化后分散小（即信号强弱主要由 $\omega_n$ 驱动），则 $c_g$ 大，单位 IC 产生更多 alpha。两个 IC 相同的信号，权重并非等分，而是按 $c_g$ 比例分配——这是情形 2 下"横截面标准分值已含波动率信息"的体现。