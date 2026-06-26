# 因子计算知识库

## MIDCAP 因子市值加权标准化：作用、计算与正交性分析

### 一、背景与动机

MIDCAP 因子旨在捕捉市值规模与股票收益之间的**非线性关系**。其原始构建逻辑为：

1. 取 Size 因子暴露（即 LNCAP，流通市值的自然对数）的**立方**：$(\ln Cap)^3$
2. 将 $(\ln Cap)^3$ 对 LNCAP 进行**截面回归**，取**残差** $\epsilon_i$
3. 对残差进行**去极值（Winsorize）**处理
4. 进行**标准化**，得到最终因子暴露 $\tilde B_i$

其中，回归方式和标准化方式直接影响正交性性质，是本文讨论的核心。

---

### 二、正交条件由权重方案决定

在回归阶段，残差 $\epsilon_i$ 满足的正交条件**由权重方案决定，而非由回归方法名称决定**：

| 回归方式 | 权重 $w_i$ | 正交条件 |
|---------|-----------|---------|
| 等权回归（$w_i \equiv 1$） | 等权 | $\Sigma \epsilon_i = 0$，$\Sigma \, \text{lncap}_i \cdot \epsilon_i = 0$ |
| 市值加权回归（$w_i = mktcap_i$） | 市值权重 | $\Sigma w_i \epsilon_i = 0$，$\Sigma w_i \cdot \text{lncap}_i \cdot \epsilon_i = 0$ |
| 截距项含常数 1，正交化落在拟合自变量 X（含常数或 LNCAP）的张成空间补空间 | 任意权重 | $\Sigma w_i \cdot X_i \cdot \epsilon_i = 0$ |

换句话说：**WLS（加权最小二乘）的残差（$ \epsilon_i $ ）与自变量($X_i$)是加权正交的，不是等权正交的。**

---

### 三、标准化公式

设 $\epsilon_i$ 为第 $i$ 只股票经正交化后得到的原始残差，$w_i$ 为该股票的**自由流通市值权重**。

若采用**等权标准化**：

$$
\tilde B_i = \frac{\epsilon_i - \mu}{\sigma}, \quad
\mu = \frac{1}{N}\sum_i \epsilon_i, \quad
\sigma = \sqrt{\frac{1}{N-1}\sum_i (\epsilon_i - \mu)^2}
$$

若采用**市值加权标准化**（BARRA CNE6 标准做法）：

$$
\tilde B_i = \frac{\epsilon_i - \mu_w}{\sigma}, \quad
\mu_w = \frac{\sum_i w_i \epsilon_i}{\sum_i w_i}, \quad
\sigma = \sqrt{\frac{1}{N-1}\sum_i (\epsilon_i - \mu_w)^2}
$$

---

### 四、正交性分析

考虑等权回归后，对 $\epsilon_i$ 进行市值加权标准化，分析其对 LNCAP 正交性的影响。

标准化后的因子 $\tilde B_i = (\epsilon_i - \mu_w)/\sigma$ 与 $\ln Cap_i$ 的内积：

$$
\sum_i \ln Cap_i \cdot \tilde B_i
= \sum_i \ln Cap_i \cdot \left( \frac{\epsilon_i - \mu_w}{\sigma} \right)
= \frac{1}{\sigma} \left( \sum_i \ln Cap_i \cdot \epsilon_i - \mu_w \sum_i \ln Cap_i \right)
$$

等权回归保证 $\sum_i \ln Cap_i \cdot \epsilon_i = 0$，代入得：

$$
\sum_i \ln Cap_i \cdot \tilde B_i
= -\frac{\mu_w}{\sigma} \sum_i \ln Cap_i \neq 0 \quad (\text{因为 } \mu_w \neq 0，除非市值加权均值为零)
$$

若采用**等权标准化**（即 $\mu = 0$，由等权回归保证 $\Sigma \epsilon_i = 0$），则正交性保留：

$$
\sum_i \ln Cap_i \cdot \tilde B_i
= \frac{1}{\sigma} \left(0 - 0 \cdot \sum_i \ln Cap_i \right) = 0
$$

---

### 五、另两种加权方案的证明

#### 方案 A：等权回归 + 市值加权标准化

**市值加权组合暴露为零**：

等权回归残差 $\epsilon_i$ 满足 $\Sigma \epsilon_i = 0$（含截距项），但**不保证** $\Sigma w_i \epsilon_i = 0$。

$$
\begin{aligned}
\sum_i w_i \tilde B_i
&= \sum_i w_i \cdot \frac{\epsilon_i - \mu_w}{\sigma} \\
&= \frac{1}{\sigma} \left( \sum_i w_i \epsilon_i - \mu_w \sum_i w_i \right) \\
&= \frac{1}{\sigma} \left( \mu_w \sum_i w_i - \mu_w \sum_i w_i \right) \quad (\mu_w \equiv \frac{\sum_i w_i \epsilon_i}{\sum_i w_i}) \\
&= 0
\end{aligned}
$$

**与 LNCAP 正交性被破坏**（已在第四节证明）：

$$
\sum_i \ln Cap_i \cdot \tilde B_i = -\frac{\mu_w}{\sigma} \sum_i \ln Cap_i \neq 0
$$

结论：标准化过程中的**平移项** $\mu_w$ 同时实现了市值加权暴露为零（期望），但破坏了等权正交（代价）。

#### 方案 B：市值加权回归 + 市值加权标准化

市值加权回归残差 $\epsilon_i$ 满足**加权正交条件（一元加权二乘估计目标函数分别截距及斜率参数求导）**：

$$
\sum_i w_i \epsilon_i = 0, \quad \sum_i w_i \cdot \ln Cap_i \cdot \epsilon_i = 0
$$

因为 $\Sigma w_i \epsilon_i = 0$，所以市值加权均值 $\mu_w = 0$，标准化退化为：

$$
\tilde B_i = \frac{\epsilon_i - 0}{\sigma} = \frac{\epsilon_i}{\sigma}
$$

**市值加权组合暴露为零**：

$$
\sum_i w_i \tilde B_i = \frac{1}{\sigma} \sum_i w_i \epsilon_i = 0
$$

**与 LNCAP 的加权正交成立**：

$$
\sum_i w_i \cdot \ln Cap_i \cdot \tilde B_i = \frac{1}{\sigma} \sum_i w_i \cdot \ln Cap_i \cdot \epsilon_i = 0
$$

**但等权正交不成立**：$\Sigma \ln Cap_i \cdot \epsilon_i$ 不一定为零，因为回归时的正交条件是加权而非等权的。

方案 B 的自洽性最强：回归和标准化使用同一套权重，正交性和组合暴露同时满足。这也是 BARRA 的**理论最优方案**。

---

### 六、三种方案对比

| 方案 | 回归权重 | 标准化权重 | 与 LNCAP 正交性 | 市值加权组合暴露 |
|------|---------|-----------|----------------|---------------|
| 等权回归 + 等权标准化 | 等权 | 等权 | ✅ 严格正交 | ❌ 非零 |
| 等权回归 + 市值加权标准化 | 等权 | 市值加权 | ❌ 被破坏 | ✅ 零 |
| 市值加权回归 + 市值加权标准化 | 市值加权 | 市值加权 | ✅ 市值加权正交（非等权） | ✅ 零 |

---

### 七、关键结论

1. **WLS 残差满足加权正交，非等权正交**：$\Sigma w_i X_i \epsilon_i = 0$，而非 $\Sigma X_i \epsilon_i = 0$。
2. **标准化方式决定最终正交性**：等权标准化保留了等权正交；市值加权标准化将其破坏。
3. **BARRA 选择后者**：即使正交性被轻微破坏（VIF ≈ 1.40），但使市值加权市场组合因子暴露为零这一全局约束的优先级更高（在Barra框架中，所有风格因子最终都经过市值加权标准化，以确保市值加权市场组合对所有风格因子的暴露为零）。
