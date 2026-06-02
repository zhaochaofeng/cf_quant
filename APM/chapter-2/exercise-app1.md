# 应用练习第1题：构建最小方差组合 C

## 目标

只考虑 MMI 中的 20 只股票，构建最小方差全额投资组合（组合 C）。每只成分股对该组合的贝塔是多少？检验式(2A-16)：$e = V h_c / \sigma_c^2$。

---

## Step 1: 协方差矩阵

- 数据：MMI 20 只股票的月度超额收益率，T ≈ 60 个月
- 收益率矩阵 $R$: [T × 20]
- 协方差矩阵 $V$: [20 × 20]，对称正定

$$
V = \frac{1}{T-1} \sum_{t=1}^T (r_t - \bar{r})(r_t - \bar{r})^T
$$

## Step 2: 组合 C 权重

- 全 1 向量 $e = (1,1,\dots,1)^T$，维度 [20×1]
- 约束：$\sum h_{c,i} = 1$（全额投资），允许做空
- 最小化：$h_c^T V h_c$

$$
h_c = \frac{V^{-1} e}{e^T V^{-1} e}
$$

## Step 3: 组合 C 方差

$$
\sigma_c^2 = h_c^T V h_c = \frac{1}{e^T V^{-1} e}
$$

## Step 4: 每只股票对 C 的贝塔

$$
\beta_c = \frac{V h_c}{\sigma_c^2}
$$

$\beta_c$: [20×1]，每个元素对应一只股票的贝塔。

## Step 5: 检验式(2A-16)

式(2A-16) 声称：$e = V h_c / \sigma_c^2$

即每只股票对组合 C 的贝塔都等于 1：$\beta_c = e$。

验证方法：计算 `max(|β_c − 1|)`，理论上应接近 0。

---

## 对应 numpy 实现

```python
import numpy as np

V = np.cov(R.T)                     # R: [T×20]
e = np.ones(20)
v_inv_e = np.linalg.solve(V, e)     # 解线性方程 V·x = e，比 inv(V)@e 更稳定
h_c = v_inv_e / (e @ v_inv_e)       # 组合 C 权重
var_c = 1 / (e @ v_inv_e)           # 组合 C 方差
beta_c = V @ h_c / var_c            # 每只股票对 C 的贝塔

assert np.allclose(beta_c, e)       # 检验式(2A-16)
```

---

## 核心直觉

组合 C 的构建不依赖任何预期收益率信息，仅基于协方差矩阵。它在所有全额投资组合中风险最小。式(2A-16) 说明每只股票对 C 的贝塔都等于 1，即组合 C 处于贝塔中性的对称位置。
