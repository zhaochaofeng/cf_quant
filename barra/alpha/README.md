# 多信号Alpha预测设计文档

## 1. 概述

本文档描述基于《主动投资组合管理》第10/11章的多信号Alpha预测框架。合并多因子信号为统一预测信号。

框架分两阶段：

1. **单信号 Alpha**：对每个信号 $k$，按式(11-15) 回归判定情形——**情形1**（信号波动率为常数）取 $\alpha_n^{(k)} = \omega_n \cdot IC_k \cdot z_{CS,n}^{(k)}$；**情形2**（信号波动率与资产波动率成比例）取 $\alpha_n^{(k)} = IC_k \cdot c_g^{(k)} \cdot z_{CS,n}^{(k)}$，其中 $c_g^{(k)}$ 由式(11-13) 估计。
2. **多信号合成**：当 $K>1$ 时，构造历史单信号 Alpha 向量，经 Cholesky 正交化（$\Sigma_\alpha = H^T H$）去除信号间冗余，再按各正交化分量的 IC（$\gamma_j$）加权求得综合 Alpha。

所有参数（$\omega_n$、$IC_k$、$c_g^{(k)}$、情形判定、正交化矩阵）均基于日频滚动窗口更新；新股因无历史残差数据，通过行业+市值回归估计 $\omega_n$。

---

## 2. 符号定义

| 符号 | 含义 |
|------|------|
| N | 资产数量 |
| K | 预测信号数量 |
| $g_n^{(k)}(t)$ | 资产 $n$ 在交易日 $t$ 的原始预测信号 $k$ |
| $z_{CS,n}^{(k)}(t)$ | 信号 $k$ 在交易日 $t$ 的横截面标准分值 |
| $\omega_n$ | 资产 $n$ 的残差波动率 |
| $IC_k$ | 信号 $k$ 的全局信息系数（日频截面相关序列经 EWMA 时间加权平均） |
| $c_g^{(k)}$ | 信号 $k$ 的情形2系数（式(11-13)），横截面分散度 ÷ 波动率归一化分散度，具有日收益率量纲 |
| $\alpha_n^{(k)}(t)$ | 仅由信号 $k$ 贡献的 Alpha 分量 |
| $\alpha_n(t)$ | 最终合成的 Alpha 预测 |
| $T$ | 历史时间窗口长度（交易日数） |

---

## 3. 输入数据

- 股票收盘价 $close_n(t)$，用于计算股票收益率 $r(t) = close_n(t+2)/close_n(t+1)-1$，进而计算 IC。获取方式：
```python
import qlib
from qlib.data import D
qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq')
df = D.features(['SZ000001'], fields=['$close'], start_time='2026-01-01', end_time='2026-01-31')
```

- 股票的预测信号 $\{g_n^{(k)}(t)\} \ (k=1,2,\dots,K)$。获取方式：
```python
import pandas as pd
df = pd.read_parquet('barra/factor_com/data/latest/alpha_exposure.parquet')
```
数据格式：MultiIndex(instrument, datetime)，列 = K个因子表达式
- 每个资产 $n$ 的残差收益率历史序列 $\{\theta_n(t)\}$。由风控模块 risk_control 模块预先计算，用于计算 $\omega_n$。获取方式：
```
直接读取: barra/risk_control/output/{dt}/model/residuals.parquet
数据格式：index ['instrument', 'datetime'], columns ['residual']
```
- 行业、流通市值（万元）数据。
获取方式：
```python
import qlib
from qlib.data import D
qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq')
instruments = D.instruments(market='csi300')
# $ind_one: 行业； circ_mv: 流通市值
df = D.features(instruments, fields=['$ind_one', '$circ_mv'], start_time='2025-01-01', end_time='2026-01-01')
```

---

## 4. 单信号Alpha计算

### 4.1 信号横截面标准化

对每个交易日 $t$ 和每个信号 $k$，计算横截面标准分值：

$$
z_{CS,n}^{(k)}(t) = \frac{g_n^{(k)}(t) - \mu_{CS}^{(k)}(t)}{\sigma_{CS}^{(k)}(t)}
$$  
 其中:  
$$
\mu_{CS}^{(k)}(t) = \frac{1}{N}\sum_{n=1}^{N} g_n^{(k)}(t), \quad 
\sigma_{CS}^{(k)}(t) = \sqrt{\frac{1}{N-1}\sum_{n=1}^{N}\left(g_n^{(k)}(t) - \mu_{CS}^{(k)}(t)\right)^2}
$$

该步骤不依赖历史数据，新股可直接参与。

### 4.2 情形判断

根据信号历史数据判断其属于情形1或情形2。通过 $\mathrm{Std}_{TS}(g)$ 与 $\omega_n$ 相关性程度判断。
$\mathrm{Std}_{TS}\{g_n(t)\}$ 的计算用过去 2 年的日频数据。

$$
\mathrm{Std}_{TS}\{g_n^{(k)}\} = a + b \cdot \omega_n + \epsilon_n
$$
 
其中，  $\mathrm{Std}_{TS}\{g_n^{(k)}\}$ 是资产 $n$ 的信号时间序列标准差。
若 $R^2 > 0.2$且 $b$ 显著，则判定为情形2，否则为情形1。该判断可预先完成并作为配置参数，按日频重估。

### 4.3 残差波动率 $\omega_n$ 估计

- **老股票**：用过去2年的日频残差收益率计算历史标准差：
  $$
  \omega_n = \mathrm{Std}\{\theta_n(t)\}
  $$
- **新股**：使用行业、市值作为自变量，对老股票样本的 $\omega_n$ 进行横截面回归：
  $$
  \omega_n = \beta_0 + \beta_1 \cdot \text{Industry}_n + \beta_2 \cdot \log(\text{MarketCap}_n) + \varepsilon_n
  $$
  对新股，用回归模型预测 $\hat{\omega}_n$ 作为其残差波动率。

### 4.4 全局IC估计

对每个信号 $k$，先对每个交易日 $t$ 计算横截面相关，得到**日频 IC 序列** $\{IC_k(t)\}$，再对时间做 **EWMA 指数加权平均**（半衰期约 60 交易日，权重 $w(t) = \lambda^{T-t}$）：

$$
IC_k = \frac{\sum_t w(t) \cdot \mathrm{Corr}_{CS}\left(z_{CS}^{(k)}(t),\ \theta_n(t)\right)}{\sum_t w(t)}, \quad w(t) = \lambda^{T-t}
$$

使用过去2年（500交易日）日数据，滚动窗口与 `config.py` 的 `ROLLING_WINDOW=500` 一致。日频序列 $\{IC_k(t)\}$ 同时是 4.5 节 IC 稳健性处理（ICIR、收缩）的输入。


### 4.5 IC 稳健性处理（时变性与变号）

#### 4.5.1 问题

IC 在历史上会漂移，甚至正负方向变动。直接以单标量 IC 作为权重会引入方向性噪声：变号因子在滚动窗口内被等权平均抹平，方向切换无法被及时跟踪。按 APM §11.7（IC 不确定性）对 IC 做稳健性处理，全部操作作用于 4.4 产出的日频 IC 序列 $\{IC_k(t)\}$。

#### 4.5.2 贝叶斯收缩（式 11-32）

$$
IC_k^* = \frac{IC_k}{1 + \dfrac{1}{T \cdot IC_k^2}}
$$

$T$ 大或 $|IC_k|$ 高 → 收缩系数接近 1，保留原值；$T$ 小或 $|IC_k|$ 低 → $IC_k^* \to 0$，变号噪声被压向零。

#### 4.5.3 ICIR 稳定性度量

对 4.4 的日频 IC 序列计算稳定性指标：

$$
ICIR_k = \frac{\bar{IC}_k}{\sigma_{IC,k}}
$$

其中 $\bar{IC}_k$、$\sigma_{IC,k}$ 为日频序列的均值与标准差（非池化标量）。$ICIR_k < 0.3$ 视为不可靠 → 激进收缩（压向 0）。变号因子 $\sigma_{IC}$ 大、ICIR 低，权重自动降低。

#### 4.5.4 方向一致性检验

对日频 IC 序列做 t 检验（$H_0: IC=0$），不能显著区分于 0 时直接置 0（式 11-32 的极限情形）。

#### 4.5.5 分情况处理

| 情况 | 处理 |
|---|---|
| IC 围绕 0 噪声震荡（变号） | 贝叶斯收缩式 11-32 压向 0；ICIR 低 → 权重调低 |
| IC 长期稳定但近期切换方向 | 4.4 的 EWMA 近期加权，快速跟踪 regime 切换 |
| IC 长期为正/负但绝对值小 | 收缩系数 < 0.5，alpha 输出缩小 |
| IC 显著且稳定 | 保留，收缩系数接近 1 |

### 4.6 情形2系数 $c_g$ 估计

当信号 $k$ 被判定为情形2（时间序列信号波动率与资产波动率成比例，式(11-8)），精炼预测需乘以系数 $c_g$（式(11-13)）：

$$
c_g^{(k)} = \frac{\text{Std}_{CS}\{g_n^{(k)}\}}{\text{Std}_{CS}\{g_n^{(k)} / \omega_n\}}
$$

逐日计算：对每个交易日 $t$，取当日横截面的 $g_n^{(k)}(t)$ 与 $\omega_n$，计算分子（信号横截面标准差）与分母（波动率归一化信号的横截面标准差），再对时间平均。$c_g$ 衡量信号横截面分散度在波动率归一化后的保留程度，具有日收益率量纲。

### 4.6 单信号Alpha公式

对每个交易日 $t$ 和每只资产 $n$，计算：

**情形1**（信号波动率为常数，式(11-7)）：

$$
\alpha_n^{(k)}(t) = \omega_n \cdot IC_k \cdot z_{CS,n}^{(k)}(t)
$$

**情形2**（信号波动率与资产波动率成比例，式(11-14)）：

$$
\alpha_n^{(k)}(t) = IC_k \cdot c_g^{(k)} \cdot z_{CS,n}^{(k)}(t)
$$

此时 $\alpha_n^{(k)}(t)$ 已具有日收益率量纲。两种情形下 alpha 量级一致（$c_g^{(k)} \approx \bar{\omega}$），可直接合并。

---

## 5. 多信号正交化与合成
  如果 K==1，则直接使用 $\alpha_n^{(1)}(t)$ 作为最终Alpha。跳过步骤5
### 5.1 构建历史Alpha矩阵

对历史每个交易日 $s$（$s=1,\dots,T$），构造向量：
$$
\mathbf{\alpha}(s) = \left(\alpha^{(1)}(s), \dots, \alpha^{(K)}(s)\right)^T
$$
其中 $\alpha^{(k)}(s)$ 是长度为 $N$ 的向量（所有资产在该交易日的Alpha分量）。将所有历史资产-交易日点 $(n,s)$ 的 $K$ 维Alpha值视为独立样本，共 $N \times T$ 个观测。

### 5.2 估计Alpha协方差矩阵

计算样本协方差矩阵 $\Sigma_\alpha \in \mathbb{R}^{K \times K}$：

$$
\Sigma_\alpha = \frac{1}{NT-1} \sum_{n=1}^{N} \sum_{s=1}^{T} \left( \mathbf{\alpha}_{n,s} - \bar{\mathbf{\alpha}} \right) \left( \mathbf{\alpha}_{n,s} - \bar{\mathbf{\alpha}} \right)^T
$$
其中 $\mathbf{\alpha}_{n,s}$ 是资产 $n$ 在交易日 $s$ 的 $K$ 维Alpha向量，$\bar{\mathbf{\alpha}}$ 是整体均值向量。使用过去3年日数据

### 5.3 Cholesky分解

由于 $\Sigma_\alpha$ 对称半正定，进行Cholesky分解：
$$
\Sigma_\alpha = H^T H
$$
其中 $H$ 是下三角矩阵。

### 5.4 正交化变换

对任意交易日（包括当前）的Alpha向量 $\mathbf{\alpha}$，计算正交化后的向量 $\mathbf{y}$：
$$
\mathbf{y} = (H^T)^{-1} \left( \mathbf{\alpha} - \bar{\mathbf{\alpha}} \right)
$$
由构造可知 $\mathrm{Var}\{\mathbf{y}\} = I_K$，各分量互不相关。

### 5.5 计算正交化信号的IC

用历史数据（过去3年日数据）估计每个正交化分量 $y_j$ 与未来t+2残差收益率 $\theta$ 的日频相关系数：

$$
\gamma_j = \mathrm{Corr}\left( y_j(t), \theta(t+2) \right)
$$
每日滚动更新。

### 5.6 合成最终Alpha

对当前交易日，最终Alpha为：
$$
\alpha_n = \sum_{j=1}^{K} \gamma_j \cdot y_{j,n}
$$
向量形式：
$$
\mathbf{\alpha} = \Gamma \cdot \mathbf{y}
$$
其中 $\Gamma = (\gamma_1, \dots, \gamma_K)$ 是 $1 \times K$ 行向量。

---

## 6. 新股处理

- **情形1信号**：新股无法计算历史 $\omega_n$，采用行业+市值回归模型预测 $\hat{\omega}_n$ 替代。
- **情形2信号**：新股无需历史 $\omega_n$ 即可计算 $z_{CS,n}^{(k)}$；系数 $c_g^{(k)}$ 为横截面常数，用当日全部股票（含老股票的 $\omega_n$）估计，新股沿用同一 $c_g^{(k)}$。
- **正交化矩阵 $H$ 和 IC 参数 $\gamma_j$**：均使用老股票历史数据估计，新股不参与参数估计，新股沿用同一变换。

---

## 7. 参数更新频率

所有参数均按**日频**滚动更新，使用过去3年（约750个交易日）的历史日数据：

| 参数 | 更新频率 | 说明 |
|------|------|------|
| $\omega_n$（老股票） | 每日   | 用过去3年日残差收益率滚动计算 |
| $\omega_n$（新股） | 每日   | 用行业+市值回归模型预测，模型每日重估 |
| $IC_k$ | 每日   | 用过去3年日数据滚动计算 |
| $c_g^{(k)}$ | 每日   | 逐日横截面计算后对时间平均 |
| $\Sigma_\alpha$、$H$、$\Gamma$ | 每日   | 用过去3年日数据滚动计算 |
| 情形判断 | 每日   | 滚动窗口回归，每日判断 |

---

## 8. 输出

- 每交易日每只资产的最终Alpha预测 $\alpha_n(t)$（日收益率量纲）
- 可选输出：各信号分量Alpha、正交化信号、IC等，用于归因和监控

---

## 9. 注意事项

1. **数据对齐**：所有历史数据需时间对齐，IC估计时注意前瞻偏差（使用 $t$ 日信号与 $t+2$ 日残差收益）。
2. **平稳性**：每日检查信号与残差波动率的关系，必要时动态调整情形分类。
3. **计算复杂度**：正交化涉及 $K \times K$ 矩阵分解，$K$ 通常较小（<100），每日计算开销可接受。
4. **稳健性**：当 $N$ 较小或历史期数不足时，可对协方差矩阵进行收缩（如Ledoit-Wolf），避免病态。
5. **新股回归模型**：行业变量采用哑变量编码，市值单位万元取对数，回归系数每日基于最新老股票样本更新。