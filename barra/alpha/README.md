# 多信号Alpha预测设计文档

## 1. 概述

本文档描述一个可编程实现的多信号Alpha预测框架。框架首先对每个信号独立计算其单信号Alpha（根据信号与残差波动率的关系选择情形1或情形2公式），然后通过正交化去除信号间冗余，最后按信息系数加权合成综合Alpha。所有参数均基于日频数据滚动更新，适用于包含新股的实际场景。

---

## 2. 符号定义

| 符号 | 含义 |
|------|------|
| N | 资产数量 |
| K | 预测信号数量 |
| g_n^{(k)}(t) | 资产 n 在交易日 t 的原始预测信号 k |
| z_{CS,n}^{(k)}(t) | 信号 k 在交易日 t 的横截面标准分值 |
| ω_n | 资产 n 的残差波动率 |
| IC_k | 信号 k 的全局信息系数（历史日频相关系数） |
| α_n^{(k)}(t) | 仅由信号 k 贡献的 Alpha 分量 |
| α_n(t) | 最终合成的 Alpha 预测 |
| T | 历史时间窗口长度（交易日数） |

---

## 3. 输入数据

- 每个信号 k(k=1,2,...,K) 的原始历史时间序列 {g_n^{(k)}(t)}(所有资产、所有交易日)。获取方式：
```python
import pandas as pd
from utils import sql_engine
engine = sql_engine()
sql = ''' 
    select qlib_code as instrument, day as datetime, score as g from monitor_return_rate 
where day>='2026-03-06' and day<='2026-03-06';
'''
df = pd.read_sql(sql, engine)
df.set_index(['instrument', 'datetime'], inplace=True)
```
- 每个资产 n 的残差收益率历史序列 {θ_n(t)}(由多因子模型预先计算，用于估计 IC 及计算ω_n)。获取方式：
```
直接读取: barra/risk_control/output/model/residuals.parquet
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

注：残差波动率 \(\omega_n\) 由 \(\theta_n(t)\) 派生，无需单独输入。

---

## 4. 单信号Alpha计算

### 4.1 信号横截面标准化

对每个交易日 \(t\) 和每个信号 \(k\)，计算横截面标准分值：

\[
z_{CS,n}^{(k)}(t) = \frac{g_n^{(k)}(t) - \mu_{CS}^{(k)}(t)}{\sigma_{CS}^{(k)}(t)}
\]
其中
\[
\mu_{CS}^{(k)}(t) = \frac{1}{N}\sum_{n=1}^{N} g_n^{(k)}(t), \quad 
\sigma_{CS}^{(k)}(t) = \sqrt{\frac{1}{N-1}\sum_{n=1}^{N}\left(g_n^{(k)}(t) - \mu_{CS}^{(k)}(t)\right)^2}
\]

该步骤不依赖历史数据，新股可直接参与。

### 4.2 情形判断

根据信号历史数据判断其属于情形1或情形2。
每个交易日t使用过去M=500个交易日的数据

\[
\mathrm{Std}_{TS}\{g_n^{(k)}\} = a + b \cdot \omega_n + \epsilon_n
\]
 
其中，  \(\mathrm{Std}_{TS}\{g_n^{(k)}\}\) 是资产 \(n\) 的信号时间序列标准差。
若 \(R^2 > 0.2)且 \(b\) 显著，则判定为情形2，否则为情形1。该判断可预先完成并作为配置参数，按日频重估。

### 4.3 残差波动率估计

- **老股票**：用过去2年的日频残差收益率计算历史标准差：
  \[
  \omega_n = \mathrm{Std}\{\theta_n(t)\}
  \]
- **新股**：使用行业、市值作为自变量，对老股票样本的 \(\omega_n\) 进行横截面回归：
  \[
  \omega_n = \beta_0 + \beta_1 \cdot \text{Industry}_n + \beta_2 \cdot \log(\text{MarketCap}_n) + \varepsilon_n
  \]
  对新股，用回归模型预测 \(\hat{\omega}_n\) 作为其残差波动率。

### 4.4 全局IC估计

对每个信号 \(k\)，用历史数据（所有资产、所有交易日）计算其横截面标准分值 \(z_{CS}^{(k)}\) 与未来一期残差收益率 \(\theta\) 的日频相关系数：

\[
IC_k = \mathrm{Corr}\left(z_{CS}^{(k)}(t), \theta(t+2)\right)
\]
其中 \(t+2\) 为t+2日的残差收益率（t+1收盘价买入，t+2收盘价卖出）。使用过去3年日数据

### 4.5 单信号Alpha公式

对每个交易日 \(t\) 和每只资产 \(n\)，计算：

\[
\alpha_n^{(k)}(t) = 
\begin{cases}
\omega_n \cdot IC_k \cdot z_{CS,n}^{(k)}(t), & \text{若信号 } k \text{ 属于情形1} \\
IC_k \cdot z_{CS,n}^{(k)}(t), & \text{若信号 } k \text{ 属于情形2}
\end{cases}
\]

此时 \(\alpha_n^{(k)}(t)\) 已具有日收益率量纲。

---

## 5. 多信号正交化与合成
  如果 K==1，则直接使用 \(\alpha_n^{(1)}(t)\) 作为最终Alpha。跳过步骤5
### 5.1 构建历史Alpha矩阵

对历史每个交易日 \(s\)（\(s=1,\dots,T\)），构造向量：
\[
\mathbf{\alpha}(s) = \left(\alpha^{(1)}(s), \dots, \alpha^{(K)}(s)\right)^T
\]
其中 \(\alpha^{(k)}(s)\) 是长度为 \(N\) 的向量（所有资产在该交易日的Alpha分量）。将所有历史资产-交易日点 \((n,s)\) 的 \(K\) 维Alpha值视为独立样本，共 \(N \times T\) 个观测。

### 5.2 估计Alpha协方差矩阵

计算样本协方差矩阵 \(\Sigma_\alpha \in \mathbb{R}^{K \times K}\)：

\[
\Sigma_\alpha = \frac{1}{NT-1} \sum_{n=1}^{N} \sum_{s=1}^{T} \left( \mathbf{\alpha}_{n,s} - \bar{\mathbf{\alpha}} \right) \left( \mathbf{\alpha}_{n,s} - \bar{\mathbf{\alpha}} \right)^T
\]
其中 \(\mathbf{\alpha}_{n,s}\) 是资产 \(n\) 在交易日 \(s\) 的 \(K\) 维Alpha向量，\(\bar{\mathbf{\alpha}}\) 是整体均值向量。使用过去3年日数据

### 5.3 Cholesky分解

由于 \(\Sigma_\alpha\) 对称半正定，进行Cholesky分解：
\[
\Sigma_\alpha = H^T H
\]
其中 \(H\) 是下三角矩阵。

### 5.4 正交化变换

对任意交易日（包括当前）的Alpha向量 \(\boldsymbol{\alpha}\)，计算正交化后的向量 \(\mathbf{y}\)：
\[
\mathbf{y} = (H^T)^{-1} \left( \mathbf{\alpha} - \bar{\mathbf{\alpha}} \right)
\]
由构造可知 \(\mathrm{Var}\{\mathbf{y}\} = I_K\)，各分量互不相关。

### 5.5 计算正交化信号的IC

用历史数据（过去3年日数据）估计每个正交化分量 \(y_j\) 与未来t+2残差收益率 \(\theta\) 的日频相关系数：

\[
\gamma_j = \mathrm{Corr}\left( y_j(t), \theta(t+2) \right)
\]
每日滚动更新。

### 5.6 合成最终Alpha

对当前交易日，最终Alpha为：
\[
\alpha_n = \sum_{j=1}^{K} \gamma_j \cdot y_{j,n}
\]
向量形式：
\[
\mathbf{\alpha} = \Gamma \cdot \mathbf{y}
\]
其中 \(\Gamma = (\gamma_1, \dots, \gamma_K)\) 是 \(1 \times K\) 行向量。

---

## 6. 新股处理

- **情形1信号**：新股无法计算历史 \(\omega_n\)，采用行业+市值回归模型预测 \(\hat{\omega}_n\) 替代。
- **情形2信号**：新股无需 \(\omega_n\)，直接使用 \(z_{CS,n}^{(k)}\)。
- **正交化矩阵 \(H\) 和 IC 参数 \(\gamma_j\)**：均使用老股票历史数据估计，新股不参与参数估计，新股沿用同一变换。

---

## 7. 参数更新频率

所有参数均按**日频**滚动更新，使用过去3年（约750个交易日）的历史日数据：

| 参数 | 更新频率 | 说明 |
|------|------|------|
| \(\omega_n\)（老股票） | 每日   | 用过去3年日残差收益率滚动计算 |
| \(\omega_n\)（新股） | 每日   | 用行业+市值回归模型预测，模型每日重估 |
| \(IC_k\) | 每日   | 用过去3年日数据滚动计算 |
| \(\Sigma_\alpha\)、\(H\)、\(\Gamma\) | 每日   | 用过去3年日数据滚动计算 |
| 情形判断 | 每日   | 滚动窗口回归，每日判断 |

---

## 8. 输出

- 每交易日每只资产的最终Alpha预测 \(\alpha_n(t)\)（日收益率量纲）
- 可选输出：各信号分量Alpha、正交化信号、IC等，用于归因和监控

---

## 9. 注意事项

1. **数据对齐**：所有历史数据需时间对齐，IC估计时注意前瞻偏差（使用 \(t\) 日信号与 \(t+2\) 日残差收益）。
2. **平稳性**：每日检查信号与残差波动率的关系，必要时动态调整情形分类。
3. **计算复杂度**：正交化涉及 \(K \times K\) 矩阵分解，\(K\) 通常较小（<100），每日计算开销可接受。
4. **稳健性**：当 \(N\) 较小或历史期数不足时，可对协方差矩阵进行收缩（如Ledoit-Wolf），避免病态。
5. **新股回归模型**：行业变量采用哑变量编码，市值单位万元取对数，回归系数每日基于最新老股票样本更新。