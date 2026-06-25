# 理论笔记

本文件记录 `utils/` 模块中涉及到的单个理论知识点，每个知识点以二级标题区分。

---

## MAD 去极值常数 1.4826
MAD（Median Absolute Deviation）中位数绝对偏差法。

代码位置：`preprocess.py` 中 `winsorize(data, method='median')`。

### 常数来源

在正态分布假设下，建立 MAD 与标准差 $\sigma$ 的换算关系。

设 $X \sim N(\mu, \sigma^2)$，定义总体 MAD：

$$
M = \text{Median}(|X - \mu|)
$$

根据中位数定义 $P(|X - \mu| \le M) = 0.5$，令 $Z = (X - \mu)/\sigma \sim N(0, 1)$，标准化后：

$$
P(|Z| \le M/\sigma) = 2\Phi(M/\sigma) - 1 = 0.5 \quad\Rightarrow\quad \Phi(M/\sigma) = 0.75
$$

查表得 $\Phi^{-1}(0.75) \approx 0.6745$，因此：

$$
\sigma = \frac{M}{0.6745} \approx 1.4826 \times M
$$

### 稳健性原理

| 统计量 | 崩溃点 | 含义 |
|-------|-------|------|
| 均值 | 0% | 1 个异常值即可使估计量失效 |
| 中位数 | 50% | 需过半数据被污染才会失效 |

MAD 的影响函数有界，标准差的影响函数无界——这是使用 MAD 而非标准差做去极值的根本原因。

### 金融数据适用性

正态分布假设在实际中不严格成立（金融数据呈尖峰厚尾），但业界（Barra 等）统一沿用 $1.4826$，保证因子间可比性和模型一致性。

### 代码公式

```
MAD(X) = Median(|X - Median(X)|)
lower_bound = Median(X) - k × 1.4826 × MAD(X)
upper_bound = Median(X) + k × 1.4826 × MAD(X)
```
