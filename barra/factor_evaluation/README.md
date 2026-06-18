# 因子评价体系

基于《主动投资组合管理》(APM) 第 12、13 章理论，构建因子和 Alpha 预测信号的量化评估体系。

## 一、模型设定

- **理论依据**：APM 第 12 章（信息分析）定义 IC/ICIR 和信息时间尺度，第 13 章（信息时间尺度）定义信号衰减与半衰期
- **评价维度**：三层递进——截面预测能力（IC）→ 实际分组收益（分层）→ 衰减速度（半衰期）
- **因子类型**：风险因子（Barra CNE6 结构性暴露）和 Alpha 因子（预测信号），风险因子不评价半衰期（结构因子不衰减）
- **收益口径**：超额收益率（个股收益 - 基准收益），基准为沪深 300
- **Alpha 中性化**：可选的将 Alpha 对风险因子正交化，分离纯 Alpha 信息

## 二、符号定义

| 符号 | 含义 |
|------|------|
| N | 每期股票数量（横截面） |
| T | 交易日数（时间序列） |
| K | 因子数量（风险因子列数 + Alpha 因子列数） |
| r_{t} | t 时刻个股收益率向量 (N×1) |
| r_{t}^{b} | t 时刻基准（沪深 300）收益率 |
| r_{t}^{e} | t 时刻超额收益率 = r_{t} - r_{t}^{b} |
| f_{t}^{(k)} | 因子 f 在 t 时刻的横截面暴露向量 (N×1) |
| IC_{t} | t 时刻截面信息系数（Pearson 相关系数） |
| RIC_{t} | t 时刻截面秩信息系数（Spearman 相关系数） |
| IR | 信息比率 = mean(IC) / std(IC) |

## 三、核心流程

```
输入（close + 因子数据）
  → 前向超额收益率计算                     ← Step 1: 预处理
  → 对每个因子逐列评价：
    → Layer 1: IC / ICIR / RIC / RICIR     ← Step 2: 截面预测能力
    → Layer 2: 分组等权收益 + 多空对冲     ← Step 3: 分层收益
    → Layer 3: IC 衰减 → 半衰期            ← Step 4: 衰减速度（仅 Alpha 因子）
  → [可选] Alpha 对风险因子正交化后复评    ← Step 5: 中性化
  → 输出结果 → MySQL + parquet
```

## 四、前向超额收益率计算

对每个滞后周期 k，计算超额前向收益：

$$
r^{e}_{t \to t+k} = \left( \frac{P_{t+k+1}}{P_{t+k}} - 1 \right) - \left( \frac{P^{b}_{t+k+1}}{P^{b}_{t+k}} - 1 \right)
$$

其中 $P_t$ 为个股收盘价，$P^b_t$ 为基准收盘价。使用 `close(t+k+1) / close(t+k) - 1` 而非 `close(t+k) / close(t) - 1`，原因是 A 股 T+1 制度——t 时刻的信号最早可以在 t+1 时刻执行。

实现方式（引擎内部，纯 pandas）：

```python
ret_df = pd.DataFrame(index=close.index)
close_gb = close.groupby(level='instrument')
for k in sorted(lags):
    ret_df[f'forward_ret_{k}'] = close_gb.shift(-k-1) / close_gb.shift(-k) - 1

# 扣除基准收益
for k in sorted(lags):
    bench_ret = benchmark_close.shift(-k-1) / benchmark_close.shift(-k) - 1
    ret_df[f'forward_ret_{k}'] -= bench_ret.reindex(dates).values
```

`groupby('instrument').shift(-k)` 确保不跨股票移位。计算在引擎初始化时一次性完成，所有因子共享同一份收益数据。

## 五、第一层：截面 IC

### 5.1 定义

截面 IC 衡量因子暴露与同期超额收益的横截面相关性：

$$
IC_t = corr(f_t, r^{e}_{t \to t+k}), \quad
RIC_t = rankcorr(f_t, r^{e}_{t \to t+k})
$$

- **IC**：Pearson 相关系数，度量线性相关
- **RIC**：Spearman 秩相关系数，度量单调性（对极端值不敏感）

### 5.2 汇总统计

- **IC 均值**：$\bar{IC} = \frac{1}{T}\sum_{t=1}^{T} IC_t$
- **IC 标准差**：$\sigma_{IC} = std(\{IC_t\})$
- **ICIR（信息比率）**：$ICIR = \bar{IC} / \sigma_{IC}$，衡量因子预测的稳定性

IC 和 RIC 各自计算其均值、标准差和 IR，得到 ICIR 和 RICIR 两组指标。

### 5.3 实现

委托给 `qlib.contrib.eva.alpha.calc_ic()`，不自行实现 groupby+corr：

```python
from qlib.contrib.eva.alpha import calc_ic as qlib_calc_ic
ic, ric = qlib_calc_ic(df[factor_col], df[ret_col])
```

返回两个 Series，索引为 datetime。上层计算汇总统计。

**设计决策**：使用 qlib 而非手动实现，原因是 qlib 内部处理了 MultiIndex 对齐和截面分组逻辑，经过生产验证。

## 六、第二层：分层收益率

### 6.1 分组收益

每日按因子值排序将股票分为 $G$ 等权分位数组（$G$ 默认为 5），组内等权平均收益率：

```python
for dt in dates:
    g = df.xs(dt, level='datetime')
    labels = pd.qcut(g[factor_col], n_groups, labels=False, duplicates='drop')
    row = g.groupby(labels)[ret_col].mean()
```

- 使用 `pd.qcut` 保证每组股票数量均衡
- `duplicates='drop'` 处理因子值大量重复导致分位数边界模糊的情况
- 返回 DataFrame(index=datetime, columns=[0..G-1])

### 6.2 多空对冲收益

顶部组（G5）买入 - 底部组（G1）卖出的零成本组合收益：

```python
from qlib.contrib.eva.alpha import calc_long_short_return as qlib_ls
long_short, avg_return = qlib_ls(df[factor_col], df[ret_col], quantile=1.0/G)
```

`long_short` 为每日多空收益序列，`avg_return` 为顶部组减全市场平均的序列。两者均由 qlib 实现，该模块仅做包装。

## 七、第三层：信号半衰期

### 7.1 理论

假设 IC 衰减服从指数模型（APM 第 13 章）：

$$
|IC(k)| \approx |IC(1)| \cdot e^{-\lambda k}, \quad k = 1, 2, ...
$$

其中 $\lambda$ 为衰减率。半衰期为 IC 降至初始值一半所需的滞后周期数：

$$
t_{1/2} = \frac{\ln 2}{\lambda}
$$

### 7.2 实现

仅对 Alpha 因子计算（风险因子为结构性暴露，不衰减）。

对几何间隔的滞后期 $k \in L$（由引擎预计算，如 1, 2, 4, 8, 16, 21, ..., 100），分别计算 RIC 均值，然后做对数线性 WLS 回归：

```python
# |IC(k)| = |IC(1)| * exp(-lambda * k)
# log|IC(k)| = log|IC(1)| - lambda * k

log_ic = log(|RIC_mean(k)|)
k_vals = lags[valid]
b, _, _ = WLS(y=log_ic, X=k_vals.reshape(-1, 1), intercept=True, weight=1)
slope = b[0]  # slope = -lambda
half_life = -ln(2) / slope
```

**决策理由**：
- 使用 WLS 而非简单 OLS：WLS 已在 `utils.stats` 中实现，与项目中其他回归一致
- 使用几何间隔滞后期而非连续滞后期：减少计算量（每期需调一次 qlib `calc_ic`），且在短端有更多采样点符合指数衰减的早期特征
- 使用 RIC 而非 IC：秩相关系数对极端值不敏感，衰减曲线更平滑
- 斜率大于等于 0 时认为不衰减（返回 NaN），因模型假设衰减为负

滞后期由引擎的 `_build_decay_lags()` 生成：

```python
# gamma=1.1, max_lag=100 → (1, 2, 4, 8, 16, 21, 28, 31, ..., 100)
def _build_decay_lags(max_lag, gamma=1.1):
    lags = [1]
    while lags[-1] * gamma <= max_lag:
        lags.append(ceil(lags[-1] * gamma))
    if lags[-1] < max_lag:
        lags.append(max_lag)
    return tuple(lags)
```

## 八、风险因子中性化

### 8.1 理论依据

APM 附录 Eq(12A-7)~(12A-11) 将信息评价与回归统一：设 $Y = [X, a]$ 为风险因子矩阵 $X$ 与 Alpha 信号 $a$ 拼接成的设计矩阵，因子组合矩阵 $H$ 满足 $H^\top Y = I$（Eq 12A-11），即 Alpha 对应的组合对 $a$ 有单位暴露、对风险因子零暴露。

将 Alpha 对风险因子做截面回归取残差，等价于施加上述零暴露条件。

### 8.2 实现

对每个交易日做横截面 WLS 回归，取残差作为中性化后的 Alpha 值：

```python
from utils.preprocess import neutralize

alpha_neutralized = neutralize(
    y=alpha_series[valid],
    x=risk_factors.loc[valid],
    weight=1,
    intercept=False,          # 不影响残差，仅影响系数可解释性
    level='datetime',         # 逐日期截面回归
)
```

- `intercept`：不影响回归残差（中性化结果）。风险因子包含完备的行业哑变量，其列空间已包含常数向量，加/不加截距的投影矩阵相同，残差（中性化后的 Alpha）完全一致。`intercept` 仅影响回归系数的可解释性：
  - `intercept=False`：行业因子系数解释为该行业的绝对超额收益
  - `intercept=True`：行业因子系数解释为相对参考行业的偏离
- `level='datetime'`：对每个 date 独立做横截面 WLS 回归

输出中区分 `raw`（未中性化）和 `neutralized`（中性化后）两组结果，便于对比。

## 九、算法流程总结

**输入**：
- `close`：个股收盘价 Series，MultiIndex(instrument, datetime)
- `risk_factors`：风险因子暴露 DataFrame，MultiIndex，每列一个风险因子（可选）
- `alpha_factors`：Alpha 因子 DataFrame，MultiIndex，每列一个 Alpha 因子（可选）
- `benchmark_close`：基准（沪深 300）收盘价 Series，MultiIndex
- `ic_periods`：IC 计算周期，如 (1,)
- `n_groups`：分层组数，默认 5
- `max_decay_lag`：半衰期最大滞后期，默认 100
- `neutralize`：是否对 Alpha 做风险因子中性化，默认 False

**输出**：
- 嵌套字典结构，每个因子包含 Layer 1 + Layer 2 + （Alpha 可选）Layer 3 结果
- 同时持久化到 MySQL `factor_evaluation` 表和本地 parquet 文件

### 算法步骤

1. 初始化：校验输入、检查因子列名重叠
2. 计算全部滞后期的前向超额收益率（一次性，所有因子共享）
3. 对每个风险因子：
   - Layer 1：IC/RIC 时间序列 + ICIR/RICIR
   - Layer 2：分 G 组等权收益 + 多空对冲收益
4. 对每个 Alpha 因子：
   - 若 neutralize=True 且有风险因子：对 Alpha 做截面正交化，得到 raw 和 neutralized 两套结果
   - Layer 1：IC/RIC 时间序列 + ICIR/RICIR
   - Layer 2：分 G 组等权收益 + 多空对冲收益
   - Layer 3：IC 衰减序列 → 指数拟合 → 半衰期
5. 持久化：写入 MySQL 表 + 输出 parquet/result.pkl

## 十、输出

### MySQL 表

表名 `factor_evaluation`，字段信息见 `factor_eval.sql`：

| 字段 | 类型 | 说明 |
|------|------|------|
| day | DATE | 计算日期 |
| name | VARCHAR(50) | 因子名称 |
| type | VARCHAR(50) | 因子类型：risk / alpha |
| IC | DECIMAL(10,6) | Normal IC |
| ICIR | DECIMAL(10,6) | IC 信息比率 |
| RIC | DECIMAL(10,6) | Rank IC |
| RICIR | DECIMAL(10,6) | Rank IC 信息比率 |
| long_short | DECIMAL(10,6) | 多空组合超额收益率均值 |
| avg_return | DECIMAL(10,6) | 市场平均超额收益率均值 |
| half_life | DECIMAL(10,2) | 半衰期（天），仅 Alpha 因子 |
| id | INT (PK) | 自增主键 |

唯一键 `(day, name, type)`，使用 `ON DUPLICATE KEY UPDATE` 实现幂等的每日更新。

### Parquet 文件

`run()` 执行过程中将中间数据和最终结果写入指定输出目录：

```
{output}/
├── ret_df.parquet          # 前向超额收益率（所有滞后期）
├── close.parquet           # 对齐后的收盘价
├── risk_factors.parquet    # 对齐后的风险因子
├── alpha_factors.parquet   # 对齐后的 Alpha 因子
├── bench_close.parquet     # 基准收盘价
└── result.pkl              # 完整的嵌套结果字典（PickleIO）
```

## 注意事项

1. **数据对齐**：`run.py` 中通过 index 三路取交集保证 close、risk_factors、alpha_factors 的 instrument 和 datetime 一致。但对齐不代表数据完整，缺失率过高的因子（有效数据 < 50%）在中性化步骤会抛异常

2. **前向收益的时间边界**：计算 `forward_ret_k` 时需要 `shift(-k-1)` 的数据，即接近数据末尾的日期缺失 k+1 天的前向收益。这些日期在 IC 计算中自动产生 NaN，不构成偏差

3. **半衰期拟合的稳健性**：要求至少有 3 个有效滞后期 |RIC| > 0。当 IC 不衰减（斜率 ≥ 0）或数据不足时返回 NaN。项目日志中记录了这些情况，属于期望行为

4. **多空收益的经济学含义**：5 组等权分组的 G5-G1 多空收益反映因子的单调区分能力，但不等同于实际可交易策略（未考虑交易成本、冲击成本、T+1 限制）

5. **中性化与截距无关**：`intercept` 参数不影响中性化结果（残差），仅影响回归系数的可解释性。风险因子中的行业哑变量列空间已包含常数向量，加/不加截距的投影矩阵相同

6. **计算周期匹配**：`max_decay_lag` 不应超过数据的时间窗口，否则对应该滞后期 k 的 `forward_ret_k` 全为 NaN，Layer 3 半衰期拟合会因有效滞后期不足 3 个而跳过。当前 `max_decay_lag=100` 配合 24 个月历史窗口（约 500 个交易日）收益充足
