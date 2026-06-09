# 因子评价体系

基于《主动投资组合管理》第12、13章理论，构建因子和Alpha预测信号的量化评估体系。

## 首期指标范围

```
第一层：IC / ICIR / RIC / RICIR
第二层：分层收益率 / 多空对冲组合收益
第三层：信号半衰期
```

## 目录结构

```
barra/factor_evaluation/
├── __init__.py
├── config.py                 # 配置常量（IC周期、分组数、半衰期阈值等）
├── engine.py                 # FactorEvalEngine - 总入口
├── layer1_ic.py              # 第一层：IC/ICIR/RIC/RICIR
├── layer2_stratified.py      # 第二层：分层收益 + 多空对冲
├── layer3_decay.py           # 第三层：信号半衰期
└── test_engine.py            # 手动验证脚本（不提交 git）
```

## 核心设计

### FactorEvalEngine 构造函数

```python
class FactorEvalEngine:
    def __init__(
        self,
        close: pd.Series,
        risk_factors: Optional[pd.DataFrame] = None,
        alpha_factors: Optional[pd.DataFrame] = None,
    ):
```

**输入约定**:
- 统一使用 `MultiIndex(instrument, datetime)` —— 与 Barra 体系、qlib 原生格式一致
- `close`: MultiIndex(instrument, datetime) 的 Series，股票收盘价
- `risk_factors`: MultiIndex(instrument, datetime) 的 DataFrame，每一列是一个风险因子（如 LNCAP、BETA、MOMENTUM...）。可以为 None
- `alpha_factors`: MultiIndex(instrument, datetime) 的 DataFrame，每一列是一个 alpha 因子。可以为 None
- `risk_factors` 和 `alpha_factors` **不能同时为 None**
- 两者都有时，两套因子的列名**不能有交集**

### 前向收益率计算（引擎内部，纯 pandas）

```
forward_ret_k = close.groupby('instrument').shift(-k) / close - 1   (k = 1, 5, 10, 21)
```

`groupby('instrument').shift(-k)` 确保不跨股票移位。

### 评价流程

```
风险因子评价:
  for each column in risk_factors:
      → Layer1: IC / ICIR / RIC / RICIR
      → Layer2: 分5组等权收益 + 多空(G5-G1)收益
      → Layer3: IC衰减 → 半衰期

Alpha 因子评价:
  for each column in alpha_factors:
      [if risk_factors is not None and neutralize=True]:
          对 alpha 做风险因子正交化（逐截面回归取残差）
      → Layer1: IC / ICIR / RIC / RICIR
      → Layer2: 分5组等权收益 + 多空(G5-G1)收益
      → Layer3: IC衰减 → 半衰期
```

### 风险因子中性化（neutralize）

**理论依据（Eq 12A-7 ~ 12A-11）**：

APM 技术附录"与回归的关系"将信息评价与回归统一：`r = Y·b + ε`，`Y = [X, a]`（风险因子矩阵 + alpha信号）。因子组合矩阵 H 满足 `Hᵀ·Y = I`（Eq 12A-11），即 alpha 对应的组合对 a 有单位暴露，对 X **零暴露**。

将 alpha 对风险因子做截面回归取残差，等价于 Eq(12A-11) 中的零暴露条件。

**实现**（直接调 `utils.preprocess.neutralize()`）：

```python
from utils.preprocess import neutralize

alpha_neutralized = neutralize(
    y=alpha_col.values,
    x=risk_factors.values,
    weight=1,
    intercept=False,          # 不加截距，保留 alpha 截面均值中的信息
)
```

`neutralize()` 内部调用 `utils.stats.WLS()` 做加权最小二乘，返回残差。不加截距项是因为截距会吸收 alpha 的截面均值（如全市场看多/看空倾向也是信息）。

输出中区分 `raw` 和 `neutralized` 两组结果。

### run() 方法签名

```python
def run(
    self,
    neutralize: bool = False,
    ic_periods: tuple = (1,),
    n_groups: int = 5,
    max_decay_lag: int = 21,
) -> dict:
```

**返回结构**:

```
{
    'risk_factors': {
        '<factor_name>': {
            'layer1': {'ic': Series, 'ric': Series, 'icir': float, 'ricir': float},
            'layer2': {'group_returns': DataFrame, 'long_short': Series},
            'layer3': {'half_life': float, 'ic_decay': Series},
        },
        ...
    },
    'alpha_factors': {
        '<factor_name>': {
            'raw': {
                'layer1': {...},
                'layer2': {...},
                'layer3': {...},
            },
            'neutralized': {               # neutralize=True 时有
                'layer1': {...},
                'layer2': {...},
                'layer3': {...},
            },
        },
        ...
    },
}
```

## 各层模块接口（委托给 qlib + utils/）

### layer1_ic.py

```python
from qlib.contrib.eva.alpha import calc_ic as qlib_calc_ic

class CrossSectionalIC:
    """委托给 qlib.contrib.eva.alpha.calc_ic()"""

    @staticmethod
    def calc_ic(df, factor_col, ret_col) -> dict:
        """调用 qlib_calc_ic(pred, label)，返回 {'ic': Series, 'ric': Series}"""

    @staticmethod
    def calc_summary(ic_series: pd.Series) -> dict:
        """返回 {'ic_mean': float, 'ic_std': float, 'icir': float}
           ICIR = ic_mean / ic_std"""
```

不自己实现 groupby+corr，直接调 qlib 的 `calc_ic()`。

### layer2_stratified.py

```python
from qlib.contrib.eva.alpha import calc_long_short_return as qlib_ls

class StratifiedReturn:
    def __init__(self, n_groups: int = 5):
        ...

    def compute(self, df, factor_col, ret_col) -> dict:
        """每日按 factor 排序分 n_groups 组，组内等权平均收益。
           分组逻辑手动实现（groupby date + pd.qcut），不依赖 qlib plotly 代码。
           多空收益可用 qlib_ls() 交叉验证。
           返回 {'group_returns': DataFrame, 'long_short': Series}"""
```

分组计算核心：`df.groupby(level='datetime').apply(lambda g: g.groupby(pd.qcut(g[factor], n_groups, labels=False))[ret_col].mean())`

### layer3_decay.py

```python
from utils.stats import get_exp_weight
from qlib.contrib.eva.alpha import calc_ic as qlib_calc_ic

class SignalDecay:
    @staticmethod
    def calc_half_life(df, factor_col, ret_prefix, max_lag) -> dict:
        """对 lag=1..max_lag 分别调 qlib_calc_ic() 得到 IC(k) 序列。
           半衰期 = IC(k) 首次降至 IC(1) * 0.5 时的 k（线性插值）。
           get_exp_weight() 用于验证指数衰减假设。
           返回 {'half_life': float, 'ic_decay': Series}"""
```

- 本身不重新实现 IC 计算，循环调 layer1 的 `calc_ic()`
- `get_exp_weight(window, half_life)` 用于辅助验证衰减模式是否服从指数衰减

## 工具复用（仅用 qlib + utils/）

### qlib 提供

| 功能 | 来源 | 用法 |
|------|------|------|
| 每日 IC/RIC | `qlib.contrib.eva.alpha.calc_ic()` | `ic, ric = calc_ic(pred, label)` → 两个 Series |
| 多空收益 | `qlib.contrib.eva.alpha.calc_long_short_return()` | `ls, avg = calc_long_short_return(pred, label, quantile=0.2)` |
| 风险评估 | `qlib.contrib.evaluate.risk_analysis()` | `risk_analysis(r, freq='day')` → 年化收益/波动/IR/最大回撤 |
| 数据获取 | `qlib.data.D.features()` / `D.instruments()` / `D.calendar()` | 因子值、收盘价、交易日历 |

### cf_quant/utils/ 提供

| 功能 | 来源 | 用法 |
|------|------|------|
| alpha 中性化 | `utils.preprocess.neutralize()` | `neutralize(y, x, weight=1, intercept=False)` |
| 因子去极值 | `utils.preprocess.winsorize()` | `winsorize(data, method='std', k=3)` |
| 因子标准化 | `utils.preprocess.standardize()` | `standardize(data, method='zscore')` |
| WLS 回归 | `utils.stats.WLS()` | `WLS(y, X, intercept=True)` → (params, intercept, resid) |
| 半衰期权重 | `utils.stats.get_exp_weight()` | `get_exp_weight(window, half_life)` → 指数衰减权重数组 |
| 日志 | `utils.LoggerFactory` | `LoggerFactory.get_logger(__name__)` |
| 超额收益 | `utils.trans.calculate_excess_returns()` | 个股收益减基准收益 |
| 交易日 | `utils.utils.is_trade_day()` / `get_trade_cal_inter()` | 交易日判断与获取 |

## 实现顺序

1. `config.py`
2. `layer1_ic.py`
3. `layer2_stratified.py`
4. `layer3_decay.py`
5. `engine.py` — 整合三层 + neutralize 逻辑
6. `test_engine.py` — 用真实因子数据跑通验证

## 测试数据

### 风险因子数据

`barra/risk_control/output/2026-04-24/debug/raw_factors.parquet`：
- 19 个 Barra CNE6 因子列: LNCAP, MIDCAP, BETA, HSIGMA, DASTD, CMRA, STOM, STOA, ATVR, STREV, SEASON, INDMOM, RSTR, HALPHA, ATO, IGRO, ETOP, CETOP, LTHALPHA
- MultiIndex(instrument, datetime), ~300 只沪深300成分股, 2018-05-08 ~ 2026-05-08

### 收盘价获取

两种方式，引擎不绑定具体来源：

```python
# 方式1: 从 Barra debug 输出文件直接读取
raw_data = pd.read_parquet('barra/risk_control/output/<date>/debug/raw_data.parquet')
close = raw_data['$close']

# 方式2: 从 qlib 获取
import qlib
from qlib.data import D
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')
close = D.features(D.instruments('csi300'), ['$close'], start_time='...', end_time='...')['$close']
```

## 验证

1. 用 raw_factors.parquet + raw_data.parquet('$close') 调 `engine.run()`，检查各层输出维度
2. Layer1 ICIR 交叉验证
3. Layer2 多空收益符号与因子经济学含义一致
4. Layer3 半衰期在合理范围（通常 5~60 个交易日）
5. neutralize=True 时 alpha 因子与风险因子的截面相关性应接近 0
