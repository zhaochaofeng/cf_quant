# 应用练习第2题：构建 CAPM 最优风险组合

## 问题

在 CAPM 预期超额收益率（正比于每只资产关于 CAPMMI 的贝塔，假设 CAPMMI 预期超额收益率为 6\%）的基础上，构建全额投资有效组合，风险厌恶系数 $\lambda = 6 / \sigma_Q^2$。

**a）** 该组合的贝塔和预期超额收益率是多少？

**b）** 将该组合与由式(2A-45)描述的 $\text{组合C}$ 和 $\text{组合Q}$ 的线性组合作比较。本题中 $\text{组合Q}$ 是 CAPMMI。

---

## 预备：有效前沿关系（正文附录）

$\text{任意全额投资有效组合}$ $P$ $\text{的方差和预期超额收益率满足式(2A-47)}$：

$$
\sigma_P^2 = \sigma_C^2 + \kappa(f_P - f_C)^2
$$

$\text{其中}$ $\kappa$ $\text{由式(2A-48)给出}$：

$$
\kappa = \frac{\sigma_Q^2 - \sigma_C^2}{(f_Q - f_C)^2}
$$

$\text{且由式(2A-35)}$：$f_C / \sigma_C^2 = f_Q / \sigma_Q^2$。

---

## Step 1: 准备数据

$\text{承接第1题结果}$，已有协方差矩阵 $V$、$\text{组合C}$ 权重 $h_c$、$\text{组合C}$ 方差 $\sigma_C^2$。

$\text{新增组合Q}$（CAPMMI，即市值加权指数）的权重：

$$
h_{Q,i} = \frac{\text{市值}_i}{\sum_j \text{市值}_j}
$$

$\text{组合Q}$ 的方差：

$$
\sigma_Q^2 = h_Q^T V h_Q
$$

$\text{风险厌恶系数}$：

$$
\lambda = \frac{6}{\sigma_Q^2}
$$

$f_Q = 6\%$（CAPMMI 预期超额收益率）。

---

## Step 2: CAPM 预期超额收益率向量

$\text{每只股票对}$ CAPMMI $\text{的贝塔}$：

$$
\beta_Q = \frac{V h_Q}{\sigma_Q^2}
$$

CAPM $\text{预期超额收益率}$：

$$
f = \beta_Q \times f_Q
$$

---

## Step 3: 效用最大化 → 最优预期收益率 $f^*$

$\text{利用附录练习第4题的结论}$，$\text{最大化}$ $f_P - \lambda\sigma_P^2$ $\text{的全额投资组合}$ $P^*$ $\text{的预期超额收益率满足}$：

$$
f^* = f_C + \frac{1}{2\lambda\kappa}
$$

$\text{推导}$：$\text{将有效前沿关系}$ $\sigma_P^2 = \sigma_C^2 + \kappa(f_P - f_C)^2$ $\text{代入效用函数}$：

$$
U = f_P - \lambda[\sigma_C^2 + \kappa(f_P - f_C)^2]
$$

$\text{一阶条件}$：

$$
\frac{dU}{df_P} = 1 - 2\lambda\kappa(f_P - f_C) = 0
$$

$\text{解得}$ $f^* = f_C + 1/(2\lambda\kappa)$。

---

## Step 4: 方法 A —— 直接拉格朗日求解

$\text{从原优化问题出发}$：

$$
\max_h \quad f^T h - \lambda h^T V h, \quad \text{s.t.} \quad e^T h = 1
$$

$\text{拉格朗日函数}$：

$$
L = f^T h - \lambda h^T V h - \gamma(e^T h - 1)
$$

$\text{一阶条件}$：

$$
\frac{\partial L}{\partial h} = f - 2\lambda V h - \gamma e = 0 \quad\Rightarrow\quad h = \frac{1}{2\lambda} V^{-1}(f - \gamma e)
$$

$\text{展开}$：

$$
h = \frac{1}{2\lambda} V^{-1} f - \frac{\gamma}{2\lambda} V^{-1} e \tag{1}
$$

$\text{其中}$ $\gamma$ $\text{为拉格朗日乘子}$。$\text{代入约束}$ $e^T h = 1$ $\text{解}$ $\gamma$：

$$
e^T h = \frac{1}{2\lambda} e^T V^{-1} f - \frac{\gamma}{2\lambda} e^T V^{-1} e = 1
$$

$$
\frac{\gamma}{2\lambda} e^T V^{-1} e = \frac{1}{2\lambda} e^T V^{-1} f - 1
$$

$$
\gamma = \frac{e^T V^{-1} f - 2\lambda}{e^T V^{-1} e} \tag{2}
$$

$\text{将(2)代入(1)}$：

$$
\begin{aligned}
h &= \frac{1}{2\lambda} V^{-1} f - \frac{1}{2\lambda} \cdot \frac{e^T V^{-1} f - 2\lambda}{e^T V^{-1} e} \cdot V^{-1} e \\
&= \frac{1}{2\lambda} V^{-1} f - \frac{e^T V^{-1} f}{2\lambda} \cdot \frac{V^{-1} e}{e^T V^{-1} e} + \frac{2\lambda}{2\lambda} \cdot \frac{V^{-1} e}{e^T V^{-1} e}
\end{aligned}
$$

$\text{识别}$ $h_C = V^{-1} e / (e^T V^{-1} e)$：

$$
\begin{aligned}
h_{P^*}^{(A)} &= \frac{1}{2\lambda} V^{-1} f - \frac{e^T V^{-1} f}{2\lambda} \cdot h_C + h_C \\
&= h_C + \frac{1}{2\lambda} \left[ V^{-1} f - (e^T V^{-1} f) h_C \right]
\end{aligned}
$$

---

## Step 5: 方法 B —— 两基金分离 (2A-45)

$\text{将}$ $f^*$ $\text{代入式(2A-45)}$，$\text{有效前沿组合是}$ $h_C$ $\text{和}$ $h_Q$ $\text{的线性组合}$：

$$
h_{P^*}^{(B)} = \frac{f_Q - f^*}{f_Q - f_C} \; h_C + \frac{f^* - f_C}{f_Q - f_C} \; h_Q
$$

$\text{记}$ $w = (f_Q - f^*) / (f_Q - f_C)$，$\text{则}$ $h_{P^*}^{(B)} = w h_C + (1-w) h_Q$。

---

## Step 6: 比较方法 A 与 方法 B（问题 b）

### 6a. 验证方式

$\text{数值验证}$：

$$
\max \left| h_{P^*}^{(A)} - h_{P^*}^{(B)} \right| \approx 0
$$

### 6b. 解析证明：$h_{P^*}^{(A)} = h_{P^*}^{(B)}$

$\text{关键一步}$——CAPM $\text{假设下}$ $V^{-1}f$ $\text{的化简}$：

$\text{由}$ $f = \beta_Q \times f_Q = (V h_Q / \sigma_Q^2) \times f_Q$：

$$
V^{-1} f = V^{-1} \left( \frac{V h_Q}{\sigma_Q^2} f_Q \right) = \frac{f_Q}{\sigma_Q^2} h_Q \tag{1}
$$

$\text{左乘}$ $e^T$：

$$
e^T V^{-1} f = \frac{f_Q}{\sigma_Q^2} e^T h_Q = \frac{f_Q}{\sigma_Q^2} \tag{2}
$$

$\text{（因为}$ $e^T h_Q = 1$，$h_Q$ $\text{全额投资）。}$

$\text{将(1)(2)代入方法A}$：

$$
\begin{aligned}
h_{P^*}^{(A)} &= h_C + \frac{1}{2\lambda} \left[ \frac{f_Q}{\sigma_Q^2} h_Q - \frac{f_Q}{\sigma_Q^2} h_C \right] \\
&= h_C + \frac{f_Q}{2\lambda\sigma_Q^2} (h_Q - h_C) \\
&= \left(1 - \frac{f_Q}{2\lambda\sigma_Q^2}\right) h_C + \left(\frac{f_Q}{2\lambda\sigma_Q^2}\right) h_Q \qquad (3)
\end{aligned}
$$

$\text{接下来将方法B化为相同形式}$。$\text{由}$ $w = (f_Q - f^*) / (f_Q - f_C)$ $\text{和}$ $f^* = f_C + 1/(2\lambda\kappa)$：

$$
\begin{aligned}
w &= 1 - \frac{f_Q - f_C}{2\lambda(\sigma_Q^2 - \sigma_C^2)} \\
&= 1 - \frac{1}{2\lambda(\sigma_Q^2 - \sigma_C^2)} \cdot f_Q \cdot \frac{\sigma_Q^2 - \sigma_C^2}{\sigma_Q^2} \\
&= 1 - \frac{f_Q}{2\lambda\sigma_Q^2}
\end{aligned}
$$

$\text{其中利用了}$ $f_Q - f_C = f_Q \cdot (\sigma_Q^2 - \sigma_C^2) / \sigma_Q^2$（$\text{由式2A-35：} f_C/\sigma_C^2 = f_Q/\sigma_Q^2$）。

$\text{因此}$ $1 - w = f_Q / (2\lambda\sigma_Q^2)$，$\text{代入(3)即得}$：

$$
h_{P^*}^{(A)} = w h_C + (1-w) h_Q = h_{P^*}^{(B)} \quad \blacksquare
$$

---

## Step 7: 组合的贝塔

$$
\beta_{P^*} = \frac{h_{P^*}^T V h_Q}{\sigma_Q^2}
$$

$\text{推导}$：$\text{由贝塔定义}$ $\beta_{P^*} = \text{Cov}(r_{P^*}, r_Q) / \text{Var}(r_Q)$：

$$
\text{Cov}(r_{P^*}, r_Q) = \text{Cov}(h_{P^*}^T r, h_Q^T r) = h_{P^*}^T V h_Q
$$

$$
\text{Var}(r_Q) = h_Q^T V h_Q = \sigma_Q^2
$$

$\text{利用}$ $V h_C = \sigma_C^2 e$（$\text{式2A-16}$）$\text{和}$ $e^T h_Q = 1$（$\text{全额投资}$），$\text{代入}$ $h_{P^*} = w h_C + (1-w) h_Q$：

$$
h_{P^*}^T V h_Q = w \cdot h_C^T V h_Q + (1-w) \cdot \sigma_Q^2
= w \sigma_C^2 \cdot e^T h_Q + (1-w) \sigma_Q^2
= w \sigma_C^2 + (1-w) \sigma_Q^2
$$

$\text{因此}$：

$$
\beta_{P^*} = w \frac{\sigma_C^2}{\sigma_Q^2} + (1-w)
$$

---

## Step 8: 预期超额收益率

$$
f_{P^*} = \beta_{P^*} \times f_Q
$$

$\text{可验证}$ $f_{P^*} = f^*$（$\text{与Step 3效用最大化结果一致}$）。

---

## 计算路径汇总

| $\text{步骤}$ | $\text{公式}$ | $\text{依赖}$ |
|---|---|---|
| $\sigma_C^2, \sigma_Q^2$ | $\text{接第1题} + h_Q^T V h_Q$ | $V, h_C, h_Q$ |
| $f_C$ | $f_Q \cdot \sigma_C^2 / \sigma_Q^2$ | $\text{式2A-35}$ |
| $\kappa$ | $(\sigma_Q^2 - \sigma_C^2) / (f_Q - f_C)^2$ | $\text{式2A-48}$ |
| $f^*$ | $f_C + 1/(2\lambda\kappa)$ | $\text{附录练习4}$ |
| $w$ | $(f_Q - f^*) / (f_Q - f_C)$ | — |
| $h_{P^*}^{(A)}$ | $h_C + \frac{1}{2\lambda}[V^{-1}f - (e^T V^{-1} f) h_C]$ | $\text{直接拉格朗日}$ |
| $h_{P^*}^{(B)}$ | $w h_C + (1-w) h_Q$ | $\text{式2A-45}$ |
| $\text{比较}$ | $\max|h_{P^*}^{(A)} - h_{P^*}^{(B)}| \approx 0$ | — |
| $\beta_{P^*}$ | $w \sigma_C^2 / \sigma_Q^2 + (1-w)$ | — |
| $f_{P^*}$ | $\beta_{P^*} \times f_Q$ | — |

---

## 与第1题的对比

| $\text{项目}$ | $\text{第1题}$ | $\text{第2题}$ |
|---|---|---|
| $\text{输入}$ | $\text{仅协方差矩阵} V$ | $V$ + CAPM $\text{预期收益率} f$ |
| $\text{组合}$ | $\text{最小方差组合} C$ | $\text{最优风险组合} P^*$ |
| $\text{目标}$ | $\min h^T V h$ | $\max f_P - \lambda \sigma_P^2$ |
| $\text{需要新增}$ | — | $\text{市值数据} h_Q$、$\lambda$ |
