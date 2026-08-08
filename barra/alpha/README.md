# 多信号Alpha预测设计文档

## 1. 概述

本文档描述基于《主动投资组合管理》第10/11章的多信号Alpha预测框架。合并多因子信号为统一预测信号。

框架分两阶段：

1. **单信号 Alpha**：对每个信号 $k$，按式(11-15) 回归判定情形——**情形1**（信号波动率为常数）取 $\alpha_n^{(k)} = \omega_n \cdot IC_k \cdot z_{CS,n}^{(k)}$；**情形2**（信号波动率与资产波动率成比例）取 $\alpha_n^{(k)} = IC_k \cdot c_g^{(k)} \cdot z_{CS,n}^{(k)}$，其中 $c_g^{(k)}$ 由式(11-13) 估计。
2. **多信号合成**：当 $K>1$ 时，构造历史信号矩阵（横截面标准分值），经 Ledoit-Wolf 收缩与 Cholesky 正交化（$\Sigma_z^* = H^T H$）去除信号间冗余，按各正交化分量的 IC（$\gamma_j$）加权（等价于信号空间 $\Sigma_z^{-1}IC$ 最优组合），尺度因子（$\omega_n$ / $c_g^{(k)}$）在组合之后乘回求得综合 Alpha。

所有参数（$\omega_n$、$IC_k$、$c_g^{(k)}$、情形判定、正交化矩阵）均基于日频滚动窗口更新；新股因无历史残差数据，通过行业+市值回归估计 $\omega_n$。

---

## 2. 符号定义

| 符号 | 含义 |
|------|------|
| N | 资产数量 |
| K | 预测信号数量 |
| $g_n^{(k)}(t)$ | 资产 $n$ 在交易日 $t$ 的原始预测信号 $k$ |
| $\tilde{g}_n^{(k)}(t)$ | 中性化后预测信号（4.1.2 产出） |
| $\theta_n(t)$ | 资产 $n$ 在交易日 $t$ 的残差收益率（4.1.1 产出），t 日计算、覆盖 $[t+1,t+2]$ 的前向残差收益率 |
| $z_{CS,n}^{(k)}(t)$ | 信号 $k$ 在交易日 $t$ 的横截面标准分值 |
| $\omega_n$ | 资产 $n$ 的残差波动率 |
| $IC_k$ | 信号 $k$ 的全局信息系数（日频截面相关序列经 EWMA 时间加权平均） |
| $c_g^{(k)}$ | 信号 $k$ 的情形2系数（式(11-13)），横截面分散度 ÷ 波动率归一化分散度，具有日收益率量纲 |
| $\Sigma_z$ | 信号协方差矩阵（5.2 产出：逐日横截面协方差经 EWMA 时间平均） |
| $w_k$ | 信号 $k$ 的组合权重（5.7 产出：$w = \Sigma_z^{-1}IC$） |
| $\alpha_n^{(k)}(t)$ | 仅由信号 $k$ 贡献的 Alpha 分量（4.8 产出，含 $IC_k$） |
| $\tilde\alpha_n^{(k)}(t)$ | 不含 $IC_k$ 的尺度信号（5.7 产出：$\tilde\alpha^{(k)} = s_k \cdot z^{(k)}$，$s_k = \omega_n$ 情形1 / $c_g^{(k)}$ 情形2） |
| $\alpha_n(t)$ | 最终合成的 Alpha 预测 |
| $T$ | 历史时间窗口长度（交易日数），$T=500$（约2年） |

---

## 3. 输入数据

- 股票收盘价 $close_n(t)$，双用途：① 计算 t+2 前向收益率 $r_n(t) = close_n(t+2)/close_n(t+1)-1$，中性化后得到残差收益率 $\theta_n(t)$（见 4.1.1）；② 计算 IC。获取方式：
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
- 每个资产 $n$ 的残差收益率历史序列 $\{\theta_n(t)\}$。**在当前项目内计算**：读取 `$close` 计算收益率，再对收益率做中性化（见 4.1.1），用于 IC 计算与 $\omega_n$ 估计。获取方式：
```python
import qlib
from qlib.data import D
qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq')
config = D.instruments(market='csi300')
instruments = D.list_instruments(config, start_date, end_date)
df = D.features(instruments, ['$close', '$ind_one', '$circ_mv'], start_date, end_date)
# 收益率: r_n(t) = close_n(t+2)/close_n(t+1) - 1
# 中性化: 对行业哑变量 + log(circ_mv)（+可选 CNE6 风格因子）横截面回归取残差 → θ_n(t)
```
数据格式：`MultiIndex(instrument, datetime), column='residual'`
- 行业、流通市值（万元）数据，三用途：① 收益率中性化（4.1.1）自变量；② 因子中性化（4.1.2）自变量；③ 新股 $\omega_n$ 估计（4.4）自变量。获取方式：
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

### 4.1 中性化处理

对**收益率**与**因子**分别做横截面中性化，剔除行业、市值（及可选 CNE6 风险因子）的共同影响。中性化范围：行业 + 市值 = **必选**；CNE6 风险因子 = **可选**（配置开关，默认关闭）。回归方式：逐日横截面 WLS/OLS 取残差，权重可按需配置（默认等权）。公式中 $f^{risk}_{n}(t)$ 为资产 $n$ 的 CNE6 风格因子暴露向量，对应项仅在开启 CNE6 中性化时存在。

#### 4.1.1 收益率中性化（残差收益率）

对每个交易日 $t$，先计算 t+2 前向收益率，再对行业哑变量 + $\log(\text{CircMV})$（+ 可选 CNE6 风格因子）横截面回归取残差：

$$
r_n(t) = \frac{close_n(t+2)}{close_n(t+1)} - 1
$$

$$
\theta_n(t) = r_n(t) - \left(\hat\beta_0(t) + \hat\beta_1(t) \cdot \text{Industry}_n + \hat\beta_2(t) \cdot \log(\text{CircMV}_n) + \hat\beta_{risk}(t)^T \cdot f^{risk}_{n}(t)\right)
$$

产出 $\theta_n(t)$ 为残差收益率，供 IC 计算（4.5）与 $\omega_n$ 估计（4.4）使用。

#### 4.1.2 因子中性化

剔除行业、市值（及可选 CNE6 风险因子）对因子的共同影响，避免因子暴露与风险因子共线、IC 被共同因子污染。对每个信号 $k$，将 $g_n^{(k)}(t)$ 对行业哑变量 + $\log(\text{CircMV})$（+ 可选 CNE6 风格因子）横截面回归取残差：

$$
\tilde{g}_n^{(k)}(t) = g_n^{(k)}(t) - \left(\hat\beta_0(t) + \hat\beta_1(t) \cdot \text{Industry}_n + \hat\beta_2(t) \cdot \log(\text{CircMV}_n) + \hat\beta_{risk}(t)^T \cdot f^{risk}_{n}(t)\right)
$$

产出中性化后信号 $\tilde{g}_n^{(k)}(t)$，作为后续标准化（4.2）的输入。

### 4.2 信号横截面标准化

对每个交易日 $t$ 和每个信号 $k$，对**中性化后信号** $\tilde{g}_n^{(k)}(t)$ 计算横截面标准分值：

$$
z_{CS,n}^{(k)}(t) = \frac{\tilde{g}_n^{(k)}(t) - \mu_{CS}^{(k)}(t)}{\sigma_{CS}^{(k)}(t)}
$$  
 其中:  
$$
\mu_{CS}^{(k)}(t) = \frac{1}{N}\sum_{n=1}^{N} \tilde{g}_n^{(k)}(t), \quad 
\sigma_{CS}^{(k)}(t) = \sqrt{\frac{1}{N-1}\sum_{n=1}^{N}\left(\tilde{g}_n^{(k)}(t) - \mu_{CS}^{(k)}(t)\right)^2}
$$

该步骤不依赖历史数据，新股可直接参与。

### 4.3 情形判断

根据信号历史数据判断其属于情形1或情形2。通过 $\mathrm{Std}_{TS}(g)$ 与 $\omega_n$ 相关性程度判断。
$\mathrm{Std}_{TS}\{g_n(t)\}$ 的计算用过去 2 年的日频数据。

$$
\mathrm{Std}_{TS}\{g_n^{(k)}\} = a + b \cdot \omega_n + \epsilon_n
$$
 
其中，  $\mathrm{Std}_{TS}\{g_n^{(k)}\}$ 是资产 $n$ 的信号时间序列标准差。
若 $R^2 > 0.2$且 $b$ 显著，则判定为情形2，否则为情形1。

### 4.4 残差波动率 $\omega_n$ 估计

- **老股票**：用过去2年的日频残差收益率计算历史标准差：
  $$
  \omega_n = \mathrm{Std}\{\theta_n(t)\}
  $$
- **新股**：使用行业、市值作为自变量，对老股票样本的 $\omega_n$ 进行横截面回归：
  $$
  \omega_n = \beta_0 + \beta_1 \cdot \text{Industry}_n + \beta_2 \cdot \log(\text{MarketCap}_n) + \varepsilon_n
  $$
  对新股，用回归模型预测 $\hat{\omega}_n$ 作为其残差波动率。

### 4.5 全局IC估计

对每个信号 $k$，先对每个交易日 $t$ 计算横截面相关，得到**日频 IC 序列** $\{IC_k(t)\}$，再对时间做 **EWMA 指数加权平均**（半衰期约 60 交易日，权重 $w(t) = \lambda^{T-t}$）：

$$
IC_k = \frac{\sum_t w(t) \cdot \mathrm{Corr}_{CS}\left(z_{CS}^{(k)}(t),\ \theta_n(t)\right)}{\sum_t w(t)}, \quad w(t) = \lambda^{T-t}
$$

使用过去2年（500交易日）日数据，滚动窗口与 `config.py` 的 `ROLLING_WINDOW=500` 一致。IC 计算基于中性化后的信号（4.1.2 → z-score）与残差收益率 $\theta$（4.1.1 产出）。日频序列 $\{IC_k(t)\}$ 同时是 4.6 节 IC 稳健性处理（ICIR、收缩）的输入。

### 4.6 IC 稳健性处理（时变性与变号）

#### 4.6.1 问题

IC 在历史上会漂移，甚至正负方向变动。直接以单标量 IC 作为权重会引入方向性噪声：变号因子在滚动窗口内被等权平均抹平，方向切换无法被及时跟踪。按 APM §11.7（IC 不确定性）对 IC 做稳健性处理，全部操作作用于 4.5 产出的日频 IC 序列 $\{IC_k(t)\}$。

#### 4.6.2 贝叶斯收缩（式 11-32）

$$
IC_k^* = \frac{IC_k}{1 + \frac{1}{T \cdot IC_k^2}}
$$

$T$ 大或 $|IC_k|$ 高 → 收缩系数接近 1，保留原值；$T$ 小或 $|IC_k|$ 低 → $IC_k^* \to 0$，变号噪声被压向零。

#### 4.6.3 ICIR 稳定性度量

对 4.5 的日频 IC 序列计算稳定性指标：

$$
ICIR_k = \frac{\bar{IC}_k}{\sigma_{IC,k}}
$$

其中 $\bar{IC}_k$、$\sigma_{IC,k}$ 为日频序列的均值与标准差（非池化标量）。$ICIR_k < 0.3$ 视为不可靠 → 激进收缩（压向 0）。变号因子 $\sigma_{IC}$ 大、ICIR 低，权重自动降低。

#### 4.6.4 方向一致性检验

对日频 IC 序列做 t 检验（$H_0: IC=0$），不能显著区分于 0 时直接置 0（式 11-32 的极限情形）。

#### 4.6.5 分情况处理

| 情况 | 处理 |
|---|---|
| IC 围绕 0 噪声震荡（变号） | 贝叶斯收缩式 11-32 压向 0；ICIR 低 → 权重调低 |
| IC 长期稳定但近期切换方向 | 4.5 的 EWMA 近期加权，快速跟踪 regime 切换 |
| IC 长期为正/负但绝对值小 | 收缩系数 < 0.5，alpha 输出缩小 |
| IC 显著且稳定 | 保留，收缩系数接近 1 |

### 4.7 情形2系数 $c_g$ 估计

当信号 $k$ 被判定为情形2（时间序列信号波动率与资产波动率成比例，式(11-8)），精炼预测需乘以系数 $c_g$（式(11-13)）：

$$
c_g^{(k)} = \frac{\text{Std}_{CS}\{g_n^{(k)}\}}{\text{Std}_{CS}\{g_n^{(k)} / \omega_n\}}
$$

逐日计算：对每个交易日 $t$，取当日横截面的 $g_n^{(k)}(t)$ 与 $\omega_n$，计算分子（信号横截面标准差）与分母（波动率归一化信号的横截面标准差），再对时间平均。$c_g$ 衡量信号横截面分散度在波动率归一化后的保留程度，具有日收益率量纲。

### 4.8 单信号Alpha公式

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
> 若 $K=1$，则直接使用 4.8 节单信号 alpha $\alpha_n^{(1)}(t)$ 作为最终 Alpha，跳过本节。

### 5.1 构建历史信号矩阵

对窗口内每个交易日 $t$（$t=1,\dots,T$），构造当日信号矩阵：
$$
\mathbf{z}(t) = \left( z_1(t), \dots, z_{N_t}(t) \right) \in \mathbb{R}^{K \times N_t}
$$
其中第 $n$ 列 $z_n(t)$ 是资产 $n$ 在交易日 $t$ 的 $K$ 维信号向量（4.2 节横截面标准分值按资产取行），信号 $k$ 的横截面（长度 $N_t$）即第 $k$ 行 $z^{(k)}(t)^T$。**维度说明**：$\mathbf{z}(t)$ 为 $K \times N_t$ 矩阵——$K$ 个信号各占一行、$N_t$ 只股票各占一列。**估计样本仅含满足历史长度要求的老股票**（残差历史 $\ge$ `NEW_STOCK_MIN_DAYS`），新股不参与协方差与 $\gamma_j$ 参数估计（见 5.6 节）。

### 5.2 估计信号协方差矩阵

对每个交易日 $t$ 先计算当日横截面协方差 $\Sigma_{CS}(t)$，再对时间做 **EWMA 指数加权平均**（半衰期约 60 交易日，权重 $w(t) = \lambda^{T-t}$，与 4.5 节 IC 的 EWMA 一致）：

$$
\Sigma_{CS}(t) = \frac{1}{N_t-1}\sum_{n=1}^{N_t}\left( z_n(t) - \bar{z}_t \right)\left( z_n(t) - \bar{z}_t \right)^T
$$

$$
\Sigma_z = \frac{\sum_t w(t)\, \Sigma_{CS}(t)}{\sum_t w(t)}
$$

其中 $\Sigma_z \in \mathbb{R}^{K \times K}$ 是信号协方差矩阵，$z_n(t)$ 是资产 $n$ 在交易日 $t$ 的 $K$ 维信号向量，$\bar{z}_t$ 是当日横截面均值向量。**逐日截面估计**由构造保留日内资产间的截面相关结构（共同因子）；**EWMA 时间平均**降低窗口内相关性非平稳的影响。样本协方差可能近乎奇异，5.3 节施加收缩以保证 Cholesky 分解正定。

### 5.3 协方差矩阵收缩

样本协方差 $\Sigma_z$ 可能近乎奇异（$K$ 较大或信号高度相关），Cholesky 分解要求**正定**矩阵。施加 **Ledoit-Wolf 收缩**（或对角增广）保证正定与分解稳定：

$$
\Sigma_z^* = (1-\rho)\,\Sigma_z + \rho\,\mathrm{diag}(\Sigma_z)
$$

其中 $\rho \in [0,1]$ 为收缩强度，按 Ledoit-Wolf 解析式或交叉验证确定。收缩是有意的偏差-方差权衡：代价是正交化分量之间的截面协方差 $\approx I_K$（非精确，$y_n$ 为 $\mathbf{y}(t)$ 的第 $n$ 列、$K$ 维），收益是 $H$ 稳定、$\gamma_j$ 估计噪声降低。

### 5.4 Cholesky分解

对收缩后的 $\Sigma_z^*$ 进行Cholesky分解（式 11A-10）：
$$
\Sigma_z^* = H^T H
$$
其中 $H$ 是上三角矩阵。

### 5.5 正交化变换

对窗口内任意交易日（包括当前），对每只资产 $n$ 的信号向量 $z_n(t)$（$\mathbf{z}(t)$ 的第 $n$ 列，$K$ 维）逐列施加变换（式 11A-11）：
$$
\mathbf{y}(t) = (H^T)^{-1} \left( \mathbf{z}(t) - \bar{\mathbf{z}} \right) \in \mathbb{R}^{K \times N_t}
$$
其中第 $n$ 列 $y_n(t) = (H^T)^{-1}\left(z_n(t) - \bar{z}\right)$，$\bar{z}$ 是窗口内信号向量均值（$K$ 维，逐列广播）。由构造可知各正交化分量之间的截面协方差 $\approx I_K$，各分量互不相关（$y_n$ 为 $\mathbf{y}(t)$ 的第 $n$ 列、$K$ 维）。

### 5.6 计算正交化信号的IC

按 4.5 节方法对每个正交化分量 $y_j$（$\mathbf{y}(t)$ 的第 $j$ 行，$N_t$ 维横截面向量）计算其与残差收益率的**日频截面相关序列**，再经 **EWMA 指数加权平均**（半衰期约 60 交易日，权重 $w(t)=\lambda^{T-t}$）：

$$
\gamma_j = \frac{\sum_t w(t)\, \mathrm{Corr}_{CS}\left( y_j(t),\ \theta(t) \right)}{\sum_t w(t)}, \quad w(t)=\lambda^{T-t}
$$

**维度说明**：$\gamma_j$ 是标量——$y_j(t)$ 与 $\theta(t)$ 都是 $N_t$ 维横截面向量，二者相关系数为标量，EWMA 平均后仍为标量。对 $j=1,\dots,K$ 得 $K$ 维列向量 $\gamma = (\gamma_1, \dots, \gamma_K)^T \in \mathbb{R}^K$，每个分量对应一个正交信号源的 IC，与 5.7 节的权重 $w = H^{-1}\gamma^*$（$K$ 维）维度一致。

时点对齐与 4.5 节一致：$y_j(t)$ 对 $t+2$ 前向残差收益率 $\theta(t)$（覆盖 $[t+1,t+2]$）。$\gamma_j$ 随后应用 4.6 节的稳健性处理——贝叶斯收缩（式 11-32）与 ICIR 门槛（$ICIR<0.3$ 激进收缩压向 0）——得 $\gamma_j^*$，每日滚动更新。

### 5.7 合成最终Alpha

由式 11A-12（$\mathrm{Cov}\{r,z\} = \mathrm{Cov}\{r,y\}H$）得 z 空间最优权重 $w = H^{-1}\gamma^*$（$K$ 维，**已含全部信号 IC 信息**）。该权重也可由 11A-13 直接推出：对任意资产 $n$（$\mathbf{z}(t)$ 的第 $n$ 列），记 $z = z_n(t)$、$y = y_n(t)$（均 $K$ 维列向量），正交分量上的最优预测为 $\phi = \omega \cdot \gamma^T y$（11A-13），代入正交变换 $y = (H^T)^{-1}(z - \bar{z})$（11A-11，见 5.5 节）：

$$
\phi = \omega \cdot \gamma^T y = \omega \cdot \gamma^T (H^T)^{-1}(z - \bar{z}) = \omega \cdot (H^{-1}\gamma)^T (z - \bar{z})
$$

其中 $(H^T)^{-1} = (H^{-1})^T$。与 $\phi = \omega \cdot w^T (z - \bar{z})$ 对比得 $w = H^{-1}\gamma$：按正交分量 IC 加权与按 z 空间权重 $w$ 加权是同一预测的两种坐标表示，预测值不变。最终 Alpha 将权重应用于**不含 IC 的尺度信号** $\tilde\alpha_n^{(k)} = s_{k,n} \cdot z_n^{(k)}(t)$（$s_{k,n} = \omega_n$ 情形1 / $c_g^{(k)}$ 情形2，见 4.8 节），避免 IC 双重计数：

$$
\alpha_n(t) = \sum_{k=1}^{K} w_k \cdot \tilde\alpha_n^{(k)}(t)
= \sum_{k=1}^{K} w_k \cdot s_{k,n} \cdot z_n^{(k)}(t)
$$

- **与单信号 IC 的关系**：$w = \Sigma_z^{-1} IC$，各信号 IC 已由权重 $w$ 吸收；4.8 节单信号 alpha $\alpha_n^{(k)} = s_{k,n} \cdot IC_k \cdot z_n^{(k)}$ 仅是 **K=1 特例**（此时 $\Sigma_z=[1]$、$w_1 = IC_1$，上式自动还原）。两信号情形权重闭式（11A-9）：
  $$
  w_1 \propto \frac{IC_1 - \rho_{12}\, IC_2}{1-\rho_{12}^2}, \quad
  w_2 \propto \frac{IC_2 - \rho_{12}\, IC_1}{1-\rho_{12}^2}
  $$
- **顺序不变性**：$w = \Sigma_z^{-1} IC$ 理论上与信号排列顺序无关，Cholesky 仅是计算手段；中间量 $y$、$\gamma_j$ 的解释随顺序而定，故信号顺序需**固定**（如按因子类别）。
- **尺度一致性**：全情形1信号时上式退化为 $\alpha_n = \omega_n \cdot \sum_k w_k z_n^{(k)}$（等价于式 11A-13 的 $\omega \cdot \gamma^T y$ 加常数项）；混合情形下各信号尺度在正交化层之外单独乘回，正交化不吸收尺度差异。

---

## 6. 新股处理

- **中性化**：新股具备行业/市值数据，可直接参与 4.1 两类中性化的逐日横截面回归，无需历史数据。
- **情形1信号**：新股无法计算历史 $\omega_n$，采用行业+市值回归模型预测 $\hat{\omega}_n$ 替代。
- **情形2信号**：新股无需历史 $\omega_n$ 即可计算 $z_{CS,n}^{(k)}$；系数 $c_g^{(k)}$ 为横截面常数，用当日全部股票（含老股票的 $\omega_n$）估计，新股沿用同一 $c_g^{(k)}$。
- **协方差矩阵 $\Sigma_z$、正交化矩阵 $H$ 和 IC 参数 $\gamma_j$**：均使用满足历史长度要求的老股票估计（5.1 节样本定义），新股不参与参数估计，新股沿用同一变换与权重。

---

## 7. 参数更新频率

所有参数均按**日频**滚动更新，使用过去 500 交易日（约2年）的历史日数据：

| 参数 | 更新频率 | 说明                  |
|------|------|---------------------|
| 中性化回归系数（收益率/因子） | 每日   | 逐日横截面回归取残差，无持久参数    |
| $\omega_n$（老股票） | 每日   | 用过去500交易日日残差收益率滚动计算     |
| $\omega_n$（新股） | 每日   | 用行业+市值回归模型预测，模型每日重估 |
| $IC_k$ | 每日   | 用过去500交易日日数据滚动计算        |
| $c_g^{(k)}$ | 每日   | 逐日横截面计算后对时间平均       |
| $\Sigma_z$、$H$、$w$ | 每日   | 逐日横截面 + EWMA + Ledoit-Wolf 收缩，用过去500交易日日数据滚动计算        |
| 情形判断 | 每日   | 滚动窗口回归，每日判断         |

---

## 8. 输出

- 每交易日每只资产的最终Alpha预测 $\alpha_n(t)$（日收益率量纲）
- 可选输出：各信号分量Alpha、正交化信号、IC等，用于归因和监控

---

## 9. 注意事项

1. **数据对齐**：所有历史数据需时间对齐；IC 估计使用 $t$ 日信号与 $t+2$ 日残差收益（4.1.1），避免前瞻偏差。
2. **平稳性**：每日检查信号与残差波动率的关系，必要时动态调整情形分类。
3. **计算复杂度**：正交化涉及 $K \times K$ 矩阵分解，$K$ 通常较小（<100），每日计算开销可接受。
4. **协方差收缩**：样本协方差可能近乎奇异，5.3 节 Ledoit-Wolf 收缩为流程必需步骤，保证 Cholesky 分解正定。
5. **新股回归模型**：行业变量采用哑变量编码，市值单位万元取对数，回归系数每日基于最新老股票样本更新。
6. **顺序**：先中性化（4.1）后标准化（4.2），不可颠倒。
7. **信息一致**：收益率中性化、因子中性化、新股 $\omega_n$ 估计共用行业/市值自变量，确保口径一致。
8. **CNE6 可选**：开启 CNE6 中性化会增加回归变量数，需注意共线性与自由度。
9. **残差来源**：残差收益率由本项目内计算（4.1.1），不再依赖 risk_control 模块输出。
10. **正交化空间**：多信号正交化在信号空间（z）进行，尺度因子（$\omega_n$ / $c_g^{(k)}$）在组合之后单独乘回（5.7），不得先乘尺度再正交化。
11. **信号顺序**：$w = \Sigma_z^{-1}IC$ 理论上与信号顺序无关，但中间量 $\gamma_j$ 随顺序而定，信号顺序需固定（如按因子类别）。