# CNE6 函数式因子计算优化方案

## 当前架构

38 个 CNE6 因子均以独立函数实现，接受统一的 MultiIndex `df` (`instrument, datetime`)，返回单列因子 DataFrame。

- **编排层**：`barra/risk_control/factor_exposure.py` 中的 `FactorExposureBuilder`，一次性加载 `raw_data`（17 个 Qlib 字段），逐因子调用
- **多进程**：生产环境通过 `multiprocessing` 分发，每因子一个 worker 进程
- **当前饱和点**：CAPM 回归（`rolling_regress` → 逐窗口逐股票串行 `lstsq`）和年度财务数据 PRef 查询

---

## 优化机会

### P0 — 高收益、低改动

#### 1. CAPM 3 因子合并为原子计算单元

**问题**：多进程下 BETA/HSIGMA/HALPHA 各占一个 worker，各自独立调用 `capm_regress(504, 252)`。模块级缓存 `_capm_cache` 在进程间无效，同一计算执行 3 次。

**文件**：`data/factor/volatility.py` (BETA, HSIGMA), `data/factor/momentum.py` (HALPHA)

**方案**：新增 `CapmFactors` 类，一次回归产出 3 个因子：

```python
class CapmFactors:
    def __init__(self, df):
        df = df.sort_index()
        b, a, s = capm_regress(df['$change'], window=504, half_life=252)
        self.BETA = pd.DataFrame({'BETA': b}).dropna()
        self.HSIGMA = pd.DataFrame({'HSIGMA': s}).dropna()
        self.HALPHA = pd.DataFrame({'HALPHA': a}).dropna()
```

编排层将 `[BETA, HSIGMA, HALPHA]` 三个任务合并为一个 `CapmFactors` 任务，分配单个 worker。

**收益**：CAPM 因子计算时间从 3T → T（**3x 加速**）

**风险**：编排层 `FactorExposureBuilder` 需支持因子组调度；因子名仍为 BETA/HSIGMA/HALPHA，输出格式不变

---

#### 2. `rolling_with_func` 直接使用 pandas 内置 rolling 方法

**问题**：`rolling().apply(func, raw=True)` 中 pandas 无法识别 `sum` 语义，走通用 apply 路径，比 `rolling().sum()` 慢 10-100 倍。

**文件**：`data/factor/utils.py`

**方案**：在 `rolling_with_func` 内部增加分发：

```python
def rolling_with_func(series, window, half_life=None, func_name='sum'):
    if func_name == 'sum' and half_life is None:
        return series.groupby(level='instrument').rolling(window=window).sum()
    if func_name == 'std' and half_life is None:
        return series.groupby(level='instrument').rolling(window=window).std()
    # ... 原逻辑
```

**收益**：受影响的因子中 sum/std 操作 **10-100x 加速**

**风险**：需确认调用方均无 `half_life`（线性 sum 不加权符合语义）

---

### P1 — 高收益、中改动

#### 3. 年度财务数据字段预计算缓存

**问题**：`remap_lyr` / `get_annual_data` / `get_annual_data2` 对相同字段重复查询 qlib PRef/P()。例如 `total_assets_q` 被 6 个因子分别查询。

| 字段 | 调用次数 | 调用方 |
|------|---------|--------|
| `total_assets_q` | 6 | MLEV, DTOA, BTOP, EM, ATO, ROA |
| `revenue_q` | 4 | VSAL, ATO, GP, GPM |
| `n_income_attr_p_q` | 3 | VERN, ETOP, ROA |
| `n_cashflow_act_q` | 3 | VFLO, ABS, ACF |

**方案**：编排层 `FactorExposureBuilder` 预先计算所有唯一字段的年度数据，缓存在内存中；因子从缓存获取，不再各自查询。

```python
# 编排层
ANNUAL_FIELDS = ['total_assets_q', 'revenue_q', 'n_income_attr_p_q', ...]
annual_cache = {}
for field in set(ANNUAL_FIELDS):
    annual_cache[field] = get_annual_data2(df[PRef_col], field)

# 因子层通过参数接收缓存
def MLEV(df, annual_cache=None):
    ta_annual = annual_cache['total_assets_q']
    ...
```

**收益**：年度数据查询量减少 ~80%（20 次 → ~5 次唯一查询）

**风险**：因子接口变更（新增 `annual_cache` 参数），需全量因子适配；PRef 字段（资产负债表 `_q`）与 P(\_a) 字段需分别处理

---

### P2 — 高收益、高改动

#### 4. `rolling_regress` 向量化

**问题**：`utils.py:208` 中 `rolling_regress` 对每只股票的每个窗口独立调用 `lstsq`。核心浪费在于 `(X^T W X)^{-1} X^T W` 投影算子对所有股票相同（基准收益率和权重仅依赖时间），但当前重复计算了 N_stocks × N_windows 次。

当前结构：
```python
for stock in instruments:
    for i in range(window-1, len(dates)):
        X_w = add_intercept(x[i-window+1:i+1])
        y_w = y_stock[i-window+1:i+1]
        w = half_life_weights(window)
        result = lstsq(X_w, y_w, w)  # 逐窗口逐股票
```

**方案**：预计算每个窗口的投影算子 `P_w = (X^T W X)^{-1} X^T W`（仅 T 次，与股票数无关），然后对每只股票做 `P_w @ y_w` 矩阵乘。

```python
# 预计算投影算子（T 次）
P = torch.linalg.lstsq(X_weighted, X).solution  # 或 numpy

# 批量应用（N 次矩阵乘）
for stock in instruments:
    beta[stock] = np.einsum('wij,wj->wi', P_windows, y_matrix)
```

**收益**：CAPM 回归核心计算 **10-50x 加速**

**风险**：工程量大，需处理 NaN、上市退市日期、内存布局；依赖 numpy/torch 向量化

---

### P3 — 中低收益、低改动

#### 5. `calc_growth_rate_slope` 简化

**问题**：对 5 个数据点调用完整 WLS 回归，单变量回归可用 `cov(x, y) / var(x)` 直接计算斜率。

**文件**：`data/factor/utils.py`

**方案**：

```python
def calc_growth_rate_slope(values, ...):
    x = np.arange(len(values))
    slope = np.cov(x, values)[0, 1] / np.var(x)  # 替代 WLS
    return slope
```

**收益**：EGRO/SGRO/AGRO/IGRO/CXGRO 中此部分 **5-10x 加速**

**风险**：需验证 results 精度一致（OLS vs WLS 结果在等权时一致）

---

#### 6. `map_annual_to_daily` 向量化

**问题**：当前 `_get_map_series` 中 `annual_df.apply(lambda row: ...)` 逐行循环生成映射。

**文件**：`data/factor/utils.py`

**方案**：用 `np.where(dt.month <= 4, dt.year - 2, dt.year - 1)` 直接生成映射，避免逐行 apply。

**收益**：年度数据回填操作 **2-5x 加速**

---

## 实施优先级

| 优先级 | 优化项 | 预估收益 | 改动范围 | 建议顺序 |
|--------|--------|---------|---------|---------|
| P0 | CAPM 3 因子合并 | 3x | 2 文件 + 编排层 ~50 行 | ① |
| P0 | rolling_with_func sum 替换 | 10-100x | 1 文件 ~10 行 | ② |
| P1 | 年度字段缓存 | ~80% 查询减少 | 编排层 + 15 因子 ~200 行 | ③ |
| P2 | rolling_regress 向量化 | 10-50x | 1 文件 ~200 行 | ④ |
| P3 | growth_rate_slope 简化 | 5-10x | 1 文件 ~10 行 | ⑤ |
| P3 | map_annual_to_daily 向量化 | 2-5x | 1 文件 ~20 行 | ⑥ |

## 不做

- **模块级缓存扩展**：多进程环境下不生效，不做。需要缓存在编排层统一管理。
- **OOP 继承重构**：当前函数式设计已论证优于 `barra_cne6_factor.py` 的 OOP 方案，不引入复杂继承。
- **LACK 因子合并**：LTHALPHA 使用 `capm_regress(1040, 260)`，与 BETA/HSIGMA/HALPHA 参数不同，无法合并。
