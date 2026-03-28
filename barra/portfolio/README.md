# 主动投资组合构建工程化设计文档

## 1. 输入参数

- **股票池**：\(N\) 只股票，基准权重向量 \(w_b \in \mathbb{R}^N\)（基准外股票权重为0），基准为沪深300指数(csi300)，股票n的权重=股票n的流通市值/指数所有股票流通市值之和。数据获取方式：
```python
import qlib
from qlib.data import D
qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq')
instruments = D.instruments(market='csi300')
# circ_mv: 流通市值；$close: 收盘价
df = D.features(instruments, fields=['$circ_mv', '$close'], start_time='2025-01-01', end_time='2026-01-01')
```
- **当前持仓**：\(w_{\text{cur}}\)，默认 w_{\text{cur}} 是一个长度与w_b相同的零向量。作为输入参数，表示当前的股票持仓权重。
- **初始金额**：\(\text{PortfolioValue}\)，默认为1亿元。作为输入参数，表示初始的组合资产净值。
- **阿尔法**：\(\alpha \in \mathbb{R}^N\)，由预测模型提供，获取方式：
```
    直接读取：barra/alpha/output/alpha_{%Y%m%d}.parquet
```
- **风险模型**：多因子模型 \(V = X F X^T + \Delta\)
  - \(X \in \mathbb{R}^{N \times K}\)：因子暴露矩阵
  - \(F \in \mathbb{R}^{K \times K}\)：因子收益率协方差矩阵（正定）
  - \(\Delta = \text{diag}(\sigma^2_1,\dots,\sigma^2_N)\)：特异风险方差对角阵
```
    数据获取方式：读取文件
    X: barra/risk_control/output/debug/exposure_matrix.parquet
    F: barra/risk_control/output/model/factor_covariance.parquet
    Delta: barra/risk_control/output/model/specific_risk.parquet
```
- **风险厌恶系数**：\(\lambda\)（以百分数单位，例如 \(\lambda=0.05\) 对应目标主动风险5%）。作为输入参数
- **交易成本**：
  - 买入成本率 \(c_b\)（如0.03%），卖出成本率 \(c_s\)（如0.13%），c_b和c_s 都是长度为N的常向量。作为输入参数
- **当前主动头寸**：\(h_{\text{cur}} = w_{\text{cur}} - w_b\)，已知
---

## 2. 优化问题构建

### 2.1 决策变量
- 主动头寸向量 \(h \in \mathbb{R}^N\)（最终要得到的）
- 辅助变量 \(b, s \in \mathbb{R}^N_{\ge 0}\)，表示买入、卖出权重变化量

### 2.2 交易量关系
\[
\Delta h = h - h_{\text{cur}} = b - s
\]

### 2.3 目标函数（最大化附加值）
\[
\max \; \alpha^T h - \lambda \, h^T V h - \left( c_b^T b + c_s^T s \right)
\]

### 2.4 约束条件
- **现金中性**（全额投资）：
  \[
  \sum_{n=1}^N h_n = 0
  \]
- **禁止卖空**（可选）：
  \[
  h_{n}\geq -w_{b,n}
  \]
  - 对基准内股票 \( (w_{b,n} > 0) \)：允许低配但权重不能为负（即 \( w_{b,n} + h_n \geq 0 \)）
  - 对基准外股票 \( (w_{b,n} = 0) \)：\( h_n \geq 0 \)
- **个股主动头寸上限**（可选，默认5%[相对于基准的主动头寸上限]）：
  \[
  h_n \le U_n
  \]
- **换手率约束**（可选，默认10%）：
  \[
  \sum (b_n + s_n) \le T_{\max}
  \]
- **非负性**：
  \[
  b \ge 0,\; s \ge 0
  \]

### 2.5 等价形式
上述问题是一个**凸二次规划**（最大化凹函数等价于最小化凸函数）。将其转化为标准最小化形式：
\[
\min \; \frac{1}{2} h^T (2\lambda V) h - \alpha^T h + c_b^T b + c_s^T s
\]
约束为线性，通过cvxpy 进行求解。

---

## 3. 求解理论最优组合

求解上述QP，得到：
- 理论最优主动头寸 \(h^*\)
- 对应的主动风险 \(\psi^* = \sqrt{h^{*T} V h^*}\)
- 边际贡献 \(MCVA_n^* = \alpha_n - 2\lambda (V h^*)_n\)

此时 \(h^*\) 是假设成本为线性的条件下的最优解。在实际交易中，可以包含如下情况：
 - 交易成本可能包含固定部分（如最低佣金5元）
 - 市场冲击成本可能是非线性的（如大单交易成本更高）
 - 线性假设会低估小额交易的实际成本

从而，优化器可能生成大量小额交易。为此引入**无交易区域**来进行后处理。

---

## 4. 无交易区域与最终头寸确定

### 4.1 边际贡献与成本区间
定义股票 \(n\) 的边际贡献：
\[
MCVA_n(h) = \alpha_n - 2\lambda (V h)_n
\]
在无交易成本时，最优解满足 \(MCVA_n(h^*) = 0\)。存在成本时，最优解应落在区间 \([-SC_n, PC_n]\) 内，其中：
- \(PC_n = c_b\)（买入成本率）
- \(SC_n = c_s\)（卖出成本率）

### 4.2 判断是否需要交易
计算当前头寸下的边际贡献 \(MCVA_n^{\text{cur}} = \alpha_n - 2\lambda (V h_{\text{cur}})_n\)。
- 若 \(-SC_n \le MCVA_n^{\text{cur}} \le PC_n\)：不交易，\(h_n^{\text{target}} = h_{\text{cur},n}\)
- 若 \(MCVA_n^{\text{cur}} > PC_n\)：需买入，调整至边界 \(MCVA_n = PC_n\)
- 若 \(MCVA_n^{\text{cur}} < -SC_n\)：需卖出，调整至边界 \(MCVA_n = -SC_n\)

### 4.3 边界头寸的线性近似（单股票调整）
假设只调整股票 \(n\)，其他股票头寸固定，则 \(MCVA_n\) 对 \(h_n\) 是线性的：
\[
MCVA_n(h_n) = MCVA_n^{\text{cur}} - 2\lambda V_{nn} (h_n - h_{\text{cur},n})
\]

令其等于边界值，解得：

**买入情形**（\(MCVA_n^{\text{cur}} > PC_n\)）：
\[
h_n^{\text{target}} = h_{\text{cur},n} + \frac{MCVA_n^{\text{cur}} - PC_n}{2\lambda V_{nn}}
\]

**卖出情形**（\(MCVA_n^{\text{cur}} < -SC_n\)）：
\[
h_n^{\text{target}} = h_{\text{cur},n} + \frac{MCVA_n^{\text{cur}} + SC_n}{2\lambda V_{nn}}
\]


边界头寸线性近似的详细推导如下：
#### 1. 边际贡献的定义

根据式 (14-7) 及后续推导，股票 \(n\) 对附加值的边际贡献为：

\[
MCVA_n(h) = \alpha_n - 2\lambda (V h)_n
\]

其中 \((V h)_n = \sum_{j=1}^N V_{nj} h_j\)，\(V_{nn}\) 是股票 \(n\) 的方差。

---

#### 2. 分离第 \(n\) 项与其他项

将 \((V h)_n\) 拆分为第 \(n\) 项与其余项之和：

\[
(V h)_n = V_{nn} h_n + \sum_{j \ne n} V_{nj} h_j
\]

因此：

\[
MCVA_n(h) = \alpha_n - 2\lambda \left( V_{nn} h_n + \sum_{j \ne n} V_{nj} h_j \right)
\]

---

#### 3. 固定其他股票头寸

假设我们只调整股票 \(n\) 的头寸，而其他股票的头寸 \(h_j\)（\(j \ne n\)）保持不变，固定在当前值 \(h_{\text{cur},j}\)。

定义常数：

\[
C_n = \sum_{j \ne n} V_{nj} h_{\text{cur},j}
\]

则：

\[
MCVA_n(h_n) = \alpha_n - 2\lambda (V_{nn} h_n + C_n)
\]

这是一个关于 \(h_n\) 的**线性函数**。

---

#### 4. 当前状态的边际贡献

在当前头寸 \(h_n = h_{\text{cur},n}\) 下：

\[
MCVA_n^{\text{cur}} = \alpha_n - 2\lambda (V_{nn} h_{\text{cur},n} + C_n)
\]

---

#### 5. 推导增量关系

将 \(MCVA_n(h_n)\) 减去 \(MCVA_n^{\text{cur}}\)：

\[
\begin{aligned}
MCVA_n(h_n) - MCVA_n^{\text{cur}} &= \left[ \alpha_n - 2\lambda (V_{nn} h_n + C_n) \right] - \left[ \alpha_n - 2\lambda (V_{nn} h_{\text{cur},n} + C_n) \right] \\
&= -2\lambda V_{nn} (h_n - h_{\text{cur},n})
\end{aligned}
\]

因此：

\[
MCVA_n(h_n) = MCVA_n^{\text{cur}} - 2\lambda V_{nn} (h_n - h_{\text{cur},n})
\]

这就是图片中的线性关系。

---

#### 6. 求解目标头寸

- 买入情形

当 \(MCVA_n^{\text{cur}} > PC_n\) 时，需要增加头寸（\(h_n > h_{\text{cur},n}\)），使边际贡献下降到买入成本边界 \(PC_n\)。

令：

\[
MCVA_n(h_n^{\text{target}}) = PC_n
\]

代入线性关系：

\[
PC_n = MCVA_n^{\text{cur}} - 2\lambda V_{nn} (h_n^{\text{target}} - h_{\text{cur},n})
\]

移项：

\[
2\lambda V_{nn} (h_n^{\text{target}} - h_{\text{cur},n}) = MCVA_n^{\text{cur}} - PC_n
\]

解得：

\[
h_n^{\text{target}} = h_{\text{cur},n} + \frac{MCVA_n^{\text{cur}} - PC_n}{2\lambda V_{nn}}
\]

- 卖出情形

当 \(MCVA_n^{\text{cur}} < -SC_n\) 时，需要减少头寸（\(h_n < h_{\text{cur},n}\)），使边际贡献上升到卖出成本边界 \(-SC_n\)。

令：

\[
MCVA_n(h_n^{\text{target}}) = -SC_n
\]

代入线性关系：

\[
-SC_n = MCVA_n^{\text{cur}} - 2\lambda V_{nn} (h_n^{\text{target}} - h_{\text{cur},n})
\]

移项：

\[
2\lambda V_{nn} (h_n^{\text{target}} - h_{\text{cur},n}) = MCVA_n^{\text{cur}} + SC_n
\]

解得：

\[
h_n^{\text{target}} = h_{\text{cur},n} + \frac{MCVA_n^{\text{cur}} + SC_n}{2\lambda V_{nn}}
\]

---

这个推导假设只调整单只股票，其他股票头寸不变，因此是“近似”。在实际迭代中，需要多轮调整以收敛到全局均衡。

### 4.4 迭代求解
由于调整一只股票会影响其他股票的边际贡献，需**迭代**：

1. 初始化 \(h = h_{\text{cur}}\)
2. 重复：
   - 计算所有 \(MCVA_n(h)\)
   - 对于每只股票，按上述规则计算 \(h_n^{\text{target}}\)
   - 更新 \(h = h^{\text{target}}\)
   - 若所有股票 \(MCVA_n\) 均在区间内，或变化小于阈值，停止

---

## 5. 生成买卖指令

最终得到 \(h^{\text{final}}\)，计算交易量：
\[
\Delta h_n = h_n^{\text{final}} - h_{\text{cur},n}
\]
- 若 \(\Delta h_n > 0\)：买入
- 若 \(\Delta h_n < 0\)：卖出

交易金额（元）：
\[
\text{Amount}_n = |\Delta h_n| \times \text{PortfolioValue}
\]

其中，PortfolioValue 是 组合的总资产净值，即股票持仓市值与现金之和。

交易股数：
\[
\text{Shares}_n = \frac{\text{Amount}_n}{P_n}
\]
其中 \(P_n\) 为股票当前价格。

---

## 6. 算法流程总结

**输入**：
- \(\alpha \in \mathbb{R}^N\)：阿尔法向量
- \(V \in \mathbb{R}^{N \times N}\)：协方差矩阵
- \(\lambda\)：风险厌恶系数（百分数单位）
- \(c_b, c_s \in \mathbb{R}^N\)：买入/卖出成本率向量
- \(h_{\text{cur}} \in \mathbb{R}^N\)：当前主动头寸
- \(w_b \in \mathbb{R}^N\)：基准权重向量
- \(\text{净值}\)：组合总资产净值（元）
- \(P \in \mathbb{R}^N\)：股票当前价格（元/股）

**输出**：
- 交易指令：每只股票的**方向**（买入/卖出）、**交易股数**

---

### 算法步骤

1. **（可选）求解理论最优组合**  
   构建二次规划问题，求解得到理论最优主动头寸 \(h^*\)（可作为迭代初始值的参考，非必需）

2. **初始化**  
   设当前主动头寸 \(h = h_{\text{cur}}\)

3. **迭代求解无交易区域**  
   重复以下步骤，直到所有股票的边际贡献均落在无交易区域内，或头寸变化小于收敛阈值：
   
   \[
   \begin{aligned}
   &\text{对于每只股票 } n = 1, 2, \dots, N: \\
   &\quad MCVA_n = \alpha_n - 2\lambda (V h)_n \\
   &\quad \text{若 } MCVA_n > c_{b,n}: \\
   &\qquad \Delta h_n = \frac{MCVA_n - c_{b,n}}{2\lambda V_{nn}} \\
   &\qquad h_n \leftarrow h_n + \Delta h_n \\
   &\quad \text{否则若 } MCVA_n < -c_{s,n}: \\
   &\qquad \Delta h_n = \frac{MCVA_n + c_{s,n}}{2\lambda V_{nn}} \\
   &\qquad h_n \leftarrow h_n + \Delta h_n \\
   &\quad \text{否则：} \\
   &\qquad \text{不调整}
   \end{aligned}
   \]
   
   每轮迭代结束后，施加现金中性约束：
   \[
   h \leftarrow h - \frac{1}{N}\sum_{n=1}^N h_n
   \]
   
   收敛条件：
   - 所有 \(MCVA_n \in [-c_{s,n}, c_{b,n}]\)，或
   - \(\|h^{(k)} - h^{(k-1)}\| < \varepsilon\)（\(\varepsilon\) 为预设阈值）

4. **计算交易量**  
   \[
   \Delta h_n = h_n - h_{\text{cur},n}, \quad n = 1, \dots, N
   \]

5. **生成买卖指令**  
   对于每只股票 \(n\)：
   - 若 \(|\Delta h_n| < \delta\)（\(\delta\) 为最小交易阈值），则跳过
   - 若 \(\Delta h_n > 0\)：买入
     \[
     \text{交易金额} = \Delta h_n \times \text{净值}, \quad \text{交易股数} = \frac{\text{交易金额}}{P_n}
     \]
   - 若 \(\Delta h_n < 0\)：卖出
     \[
     \text{交易金额} = |\Delta h_n| \times \text{净值}, \quad \text{交易股数} = \frac{\text{交易金额}}{P_n}
     \]

## 7. 注意事项
- 计算过程对齐索引 
- 保留中间计算过程，以便问题排查和结果验证，其中包括最新的持仓数据
---

