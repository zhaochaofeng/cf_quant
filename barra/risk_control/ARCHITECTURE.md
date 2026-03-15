# Barra CNE6 风险模型系统 - 代码组织结构

## 系统概述

本项目实现了完整的 **Barra CNE6 多因子风险模型**，用于量化投资组合的风险控制和归因分析。系统基于qlib框架，针对8GB内存环境进行了优化，支持沪深300成分股的实时风险监控。

---

## 目录结构

```
barra/risk_control/
│
├── README.md                          # 技术设计文档（算法原理）
├── README_MEMORY_OPTIMIZATION.md      # 内存优化指南
├── TEST_README.md                     # 测试文档
│
├── __init__.py                        # 包初始化，导出核心组件
├── config.py                          # 配置参数（因子定义、行业映射）
│
├── # 数据层
├── data_loader.py                     # 数据加载器（qlib集成）
├── portfolio.py                       # 组合管理（基准/持仓/权重计算）
│
├── # 因子层
├── factor_exposure.py                 # 因子暴露矩阵构建（标准版）
├── factor_exposure_optimized.py       # 因子暴露矩阵构建（内存优化版）
│
├── # 模型层
├── cross_sectional.py                 # 横截面回归（标准版）
├── cross_sectional_optimized.py       # 横截面回归（内存优化版）
├── covariance.py                      # 因子协方差矩阵估计
├── specific_risk.py                   # 特异风险矩阵估计（ARMA+面板回归）
├── risk_model.py                      # 资产协方差矩阵计算
│
├── # 归因层
├── risk_attribution.py                # 风险归因分析（MCAR/RCAR/FMCAR/FRCAR）
│
├── # 输出层
├── output.py                          # CSV输出管理
├── output_manager.py                  # 输出管理器
│
├── # 控制层
├── barra_engine.py                    # 主引擎（标准版）
├── barra_engine_optimized.py          # 主引擎（内存优化版）
│
├── # 工具层
├── memory_utils.py                    # 内存优化工具集
│
├── # 脚本层
├── run_daily.py                       # 每日风险计算脚本
├── run_monthly.py                     # 每月模型更新脚本
├── example.py                         # 使用示例
├── example_optimized.py               # 内存优化版示例
│
├── # 测试层
├── test_config.py                     # 测试配置
├── test_base.py                       # 测试基类和工具
├── test_modules.py                    # 模块化单元测试
├── test_integration.py                # 集成测试
└── test_runner.py                     # 主测试运行脚本
```

---

## 核心模块详解

### 1. 配置模块

#### `config.py`
**职责**: 集中管理所有配置参数

**关键内容**:
```python
# CNE6 风格因子定义（38个）
CNE6_STYLE_FACTORS = {
    'size': ['LNCAP', 'MIDCAP'],
    'volatility': ['BETA', 'HSIGMA', 'DASTD', 'CMRA'],
    'liquidity': ['STOM', 'STOQ', 'STOA', 'ATVR'],
    ...
}

# 申万一级行业映射（31个）
INDUSTRY_MAPPING = {
    '801780': '银行',
    '801180': '房地产',
    ...
}

# 模型参数
MODEL_PARAMS = {
    'history_window': 120,  # 10年历史数据
    'arma_p': 1,            # ARMA(p,q)
    'arma_q': 1,
}
```

**使用场景**:
- 因子名称统一管理
- 行业分类标准化
- 模型超参数配置

---

### 2. 数据层

#### `data_loader.py`
**职责**: 从qlib加载各类市场数据

**核心类**: `DataLoader`

**主要方法**:
| 方法 | 功能 | 返回值 |
|------|------|--------|
| `get_instruments()` | 获取股票列表 | List[str] |
| `load_returns()` | 加载收益率 | DataFrame ['return'] |
| `load_market_cap()` | 加载市值 | DataFrame ['circ_mv', 'total_mv'] |
| `load_industry()` | 加载行业分类 | DataFrame ['industry_code'] |
| `load_factor_data()` | 加载原始因子数据 | DataFrame (30+列) |

**数据流向**:
```
qlib数据源 → DataLoader → 各模块使用
```

---

#### `portfolio.py`
**职责**: 管理投资组合和基准

**核心类**: `PortfolioManager`

**主要功能**:
1. **基准权重计算**: 沪深300市值加权
2. **随机组合生成**: 等权重随机选择n只股票
3. **主动权重计算**: h_PA = h_p - h_b
4. **组合因子暴露**: x_p = X^T * h_p

**关键公式**:
```python
# 主动权重
h_pa = portfolio_weights - benchmark_weights

# 组合因子暴露
x_p = exposure_matrix.T @ portfolio_weights
```

---

### 3. 因子层

#### `factor_exposure.py` / `factor_exposure_optimized.py`
**职责**: 构建因子暴露矩阵，执行完整的预处理流程

**核心类**: `FactorExposureBuilder`

**处理流程**:
```
原始数据
    ↓
[1] 计算原始因子（38个CNE6因子并行计算）
    ↓
[2] 去极值（中位数MAD方法）
    ↓
[3] 中性化（行业/市值中性化）
    ↓
[4] 正交化（逐步回归取残差）
    ↓
[5] 标准化（Z-Score）
    ↓
[6] 合并行业因子（One-Hot编码）
    ↓
因子暴露矩阵 X_t (N×K)
```

**内存优化策略**:
- `float64` → `float32` (节省50%内存)
- 分批处理（100只股票/批）
- 增量计算（磁盘缓存）
- 及时垃圾回收

---

### 4. 模型层

#### `cross_sectional.py` / `cross_sectional_optimized.py`
**职责**: 横截面回归估计因子收益率

**核心类**: `CrossSectionalRegression`

**模型公式**:
```
r_t = X_t × b_t + u_t

其中:
- r_t: 股票超额收益率 (N×1)
- X_t: 因子暴露矩阵 (N×K)
- b_t: 因子收益率 (K×1) ← 待估计
- u_t: 特异收益率 (N×1)
```

**加权最小二乘 (WLS)**:
```python
# 市值平方根加权
W_t = diag(sqrt(MV_1), sqrt(MV_2), ..., sqrt(MV_N))

# 估计公式
b̂_t = (X_t^T × W_t × X_t)^(-1) × X_t^T × W_t × r_t
```

---

#### `covariance.py`
**职责**: 估计因子协方差矩阵

**核心类**: `FactorCovarianceEstimator`

**算法**:
1. **样本协方差矩阵** (默认)
   ```
   F = 1/(T-1) × Σ(b_t - b̄)(b_t - b̄)^T
   ```

2. **指数加权协方差** (EWMA)
   - 半衰期36个月
   - 近期数据权重更高

3. **Ledoit-Wolf收缩估计**
   - 提高数值稳定性
   - 防止过拟合

---

#### `specific_risk.py`
**职责**: 估计特异风险矩阵（ARMA + 面板回归）

**核心类**: `SpecificRiskEstimator`

**模型步骤**:
```
Step 1: 分解特异方差
    S(t) = mean(u_n^2(t))           # 平均特异方差
    v_n(t) = u_n^2(t)/S(t) - 1      # 相对偏离

Step 2: ARMA预测 S(t+1)
    S(t) = c + Σφ_i×S(t-i) + Σθ_j×ε(t-j) + ε(t)

Step 3: 面板回归预测 v_n(t+1)
    v_n(t) = Σβ_k,n(t) × λ_k(t) + ε_n(t)

Step 4: 合成未来特异方差
    û_n^2(t+1) = Ŝ(t+1) × [1 + v̂_n(t+1)]
```

---

#### `risk_model.py`
**职责**: 计算资产协方差矩阵

**核心类**: `AssetCovarianceCalculator`

**核心公式**:
```
V = X × F × X^T + Δ

其中:
- V: 资产协方差矩阵 (N×N)
- X: 因子暴露矩阵 (N×K)
- F: 因子协方差矩阵 (K×K)
- Δ: 特异风险对角矩阵 (N×N)
```

---

### 5. 归因层

#### `risk_attribution.py`
**职责**: 计算风险贡献指标

**核心类**: `RiskAttributionAnalyzer`

**风险指标**:

| 指标 | 公式 | 含义 |
|------|------|------|
| **MCAR** | V × h_PA / ψ_p | 股票边际风险贡献 |
| **RCAR** | h_PA ⊙ MCAR | 股票风险贡献 |
| **FMCAR** | F × x_PA / ψ_p | 因子边际风险贡献 |
| **FRCAR** | x_PA ⊙ FMCAR | 因子风险贡献 |

**理论验证**:
```
ΣRCAR_n = ψ_p                (主动风险)
ΣFRCAR_k ≈ ψ_p - 特异风险    (因子解释部分)
```

---

### 6. 输出层

#### `output.py`
**职责**: 管理风险指标的CSV输出

**核心类**: `RiskOutputManager`

**输出文件**:

1. **股票风险文件** (`stock_risk_YYYYMMDD.csv`)
   ```csv
   instrument,mcar,rcar,calc_date
   000001.SZ,0.004967,0.000324,2024-03-01
   ...
   ```

2. **因子风险文件** (`factor_risk_YYYYMMDD.csv`)
   ```csv
   factor_name,fmcar,frcar,factor_type,calc_date
   LNCAP,0.01523,0.000612,规模,2024-03-01
   BTOP,-0.001383,-0.000385,价值,2024-03-01
   银行,0.006477,-0.000677,行业,2024-03-01
   ...
   ```

**特性**:
- 自动创建输出目录
- 保留6位小数精度
- UTF-8编码
- 按类型分组排序

---

### 7. 控制层

#### `barra_engine.py` / `barra_engine_optimized.py`
**职责**: 主控制引擎，整合所有模块

**核心类**: `BarraRiskEngine`

**工作流程**:

```python
# 1. 初始化
engine = BarraRiskEngine(
    calc_date='2024-03-14',
    portfolio_input='random',  # 或权重字典
    output_dir='output',
    memory_threshold_gb=6.0
)

# 2. 月频更新（每月运行一次）
engine.run_monthly_update(
    start_date='2014-03-01',
    end_date='2024-03-01',
    stock_batch_size=100,   # 分批处理
    date_batch_size=10
)

# 3. 日频计算（每日运行）
engine.run_daily_risk()

# 4. 保存结果
engine.save_results()

# 5. 生成报告
engine.print_risk_report()
```

---

### 8. 工具层

#### `memory_utils.py`
**职责**: 内存优化工具集

**核心功能**:

| 工具类/函数 | 功能 |
|------------|------|
| `MemoryMonitor` | 实时监控内存使用 |
| `convert_to_float32()` | 数据类型转换 |
| `optimize_dataframe_memory()` | DataFrame内存优化 |
| `suggest_workers_by_memory()` | 根据内存建议进程数 |
| `chunk_list()` | 列表分批处理 |
| `chunk_dataframe_generator()` | DataFrame分批生成器 |
| `clear_variables()` | 强制垃圾回收 |

**内存优化效果**:
- Float32转换: **节省50%内存**
- 分批处理: 峰值内存降低至 **3-4GB** (8GB系统)
- 增量模式: 支持磁盘缓存

---

### 9. 脚本层

#### `run_daily.py`
**用途**: 每日风险计算入口

```bash
python barra/risk_control/run_daily.py \
    --date 2024-03-14 \
    --portfolio random
```

#### `run_monthly.py`
**用途**: 每月模型更新入口

```bash
python barra/risk_control/run_monthly.py \
    --end-date 2024-03-01 \
    --history-months 120
```

---

### 10. 测试层

#### 测试体系结构

```
test_runner.py          # 主测试入口
test_config.py          # 测试配置
test_base.py            # 测试基类（断言、报告）
├── test_modules.py     # 单元测试（6个模块）
└── test_integration.py # 集成测试（5个场景）
```

**测试覆盖**:
- ✓ 数据加载（qlib集成）
- ✓ 组合管理（基准/持仓）
- ✓ 因子计算（去极值/中性化/正交化）
- ✓ 协方差估计（样本/收缩）
- ✓ 风险归因（MCAR/RCAR/FMCAR/FRCAR）
- ✓ 数值正确性（RCAR之和=主动风险）
- ✓ 内存优化（float32节省50%）

---

## 数据流图

```
┌─────────────────────────────────────────────────────────────┐
│                        输入层                               │
├─────────────────────────────────────────────────────────────┤
│  • 股票持仓 (portfolio_weights)                            │
│  • 市场基准 (benchmark_weights, 默认沪深300)                │
│  • qlib数据源 (price, volume, financial reports)           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      数据层                                 │
├─────────────────────────────────────────────────────────────┤
│  DataLoader.load_factor_data()                             │
│  DataLoader.load_returns()                                 │
│  DataLoader.load_market_cap()                              │
│  DataLoader.load_industry()                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      因子层                                 │
├─────────────────────────────────────────────────────────────┤
│  FactorExposureBuilder.build_exposure_matrix()             │
│    ├── calculate_raw_factors()     # 38个CNE6因子          │
│    ├── winsorize_factors()         # 去极值                │
│    ├── neutralize_factors()        # 行业/市值中性化      │
│    ├── orthogonalize_factors()     # 正交化                │
│    ├── standardize_factors()       # 标准化                │
│    └── merge_industry_factors()    # 合并31个行业因子     │
└────────────────────┬────────────────────────────────────────┘
                     │ X_t (N×K) 因子暴露矩阵
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      模型层                                 │
├─────────────────────────────────────────────────────────────┤
│  横截面回归: CrossSectionalRegression.fit()                │
│    └── b_t (因子收益率)                                    │
│                                                            │
│  协方差估计: FactorCovarianceEstimator                     │
│    └── F (因子协方差矩阵)                                  │
│                                                            │
│  特异风险: SpecificRiskEstimator                           │
│    └── Δ (特异风险矩阵)                                    │
│                                                            │
│  资产协方差: AssetCovarianceCalculator                     │
│    └── V = XFX^T + Δ                                       │
└────────────────────┬────────────────────────────────────────┘
                     │ V (资产协方差矩阵)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      归因层                                 │
├─────────────────────────────────────────────────────────────┤
│  RiskAttributionAnalyzer.analyze_risk()                    │
│    ├── calculate_mcar()     # 股票边际风险贡献            │
│    ├── calculate_rcar()     # 股票风险贡献                │
│    ├── calculate_fmcar()    # 因子边际风险贡献            │
│    └── calculate_frcar()    # 因子风险贡献                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      输出层                                 │
├─────────────────────────────────────────────────────────────┤
│  RiskOutputManager.save_stock_risk()                       │
│    └── stock_risk_YYYYMMDD.csv                             │
│                                                            │
│  RiskOutputManager.save_factor_risk()                      │
│    └── factor_risk_YYYYMMDD.csv                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 类关系图

```
BarraRiskEngine (主引擎)
    ├── DataLoader (数据加载)
    ├── PortfolioManager (组合管理)
    ├── FactorExposureBuilder (因子暴露)
    ├── CrossSectionalRegression (横截面回归)
    ├── FactorCovarianceEstimator (协方差估计)
    ├── SpecificRiskEstimator (特异风险)
    ├── AssetCovarianceCalculator (资产协方差)
    ├── RiskAttributionAnalyzer (风险归因)
    └── RiskOutputManager (输出管理)
```

---

## 使用示例

### 基础用法

```python
from barra.risk_control import BarraRiskEngine

# 初始化引擎
engine = BarraRiskEngine(
    calc_date='2024-03-14',
    portfolio_input='random',  # 随机50只股票
    output_dir='output'
)

# 运行全流程
engine.run_monthly_update('2014-03-01', '2024-03-01')
engine.run_daily_risk()
engine.save_results()
```

### 高级用法（内存优化）

```python
from barra.risk_control import BarraRiskEngine

# 8GB内存优化配置
engine = BarraRiskEngine(
    calc_date='2024-03-14',
    portfolio_input={'000001.SZ': 0.5, '000002.SZ': 0.5},
    memory_threshold_gb=6.0,
    use_incremental=True,      # 启用磁盘缓存
    stock_batch_size=100,      # 分批处理
    n_jobs=2                   # 并行进程数
)

# 运行
engine.run_monthly_update(
    '2014-03-01', '2024-03-01',
    stock_batch_size=100,
    date_batch_size=10
)
```

### 独立模块使用

```python
from barra.risk_control import FactorExposureBuilder
from barra.risk_control import RiskAttributionAnalyzer

# 单独使用因子暴露构建器
builder = FactorExposureBuilder()
exposure = builder.build_exposure_matrix(
    raw_data, industry_df, market_cap_df
)

# 单独使用风险归因
analyzer = RiskAttributionAnalyzer()
results = analyzer.analyze_risk(
    asset_cov, factor_cov, exposure,
    portfolio_weights, benchmark_weights
)
```

---

## 开发指南

### 添加新因子

1. 在 `data/factor/` 目录下实现因子计算函数
2. 在 `config.py` 的 `CNE6_STYLE_FACTORS` 中添加因子名称
3. 在 `factor_exposure.py` 的 `FACTOR_FUNCTIONS` 字典中注册

### 修改模型参数

编辑 `config.py` 中的 `MODEL_PARAMS`:

```python
MODEL_PARAMS = {
    'history_window': 120,  # 修改历史窗口
    'arma_p': 2,            # 修改ARMA阶数
    'arma_q': 2,
}
```

### 扩展测试

在 `test_modules.py` 或 `test_integration.py` 中添加:

```python
def test_new_feature(self):
    """测试新功能"""
    # 测试代码
    pass
```

---

## 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **支持股票数** | 300只 | 沪深300成分股 |
| **支持因子数** | 69个 | 38风格 + 31行业 |
| **历史数据窗口** | 120个月 | 10年 |
| **内存占用** | 3-4GB | 峰值（8GB系统） |
| **计算时间** | 1-2分钟 | 日频风险计算 |
| **月频更新** | 30-60分钟 | 含因子暴露计算 |

---

## 依赖项

```
必需:
- pandas >= 1.3.0
- numpy >= 1.20.0
- statsmodels >= 0.13.0
- qlib >= 0.8.0

可选:
- joblib (并行计算)
- psutil (内存监控)
```

---

## 相关文档

- [技术设计文档](README.md) - 算法原理和数学公式
- [内存优化指南](README_MEMORY_OPTIMIZATION.md) - 8GB内存优化策略
- [测试文档](TEST_README.md) - 测试框架使用说明
- [Barra CNE6因子说明](https://www.example.com) - 外部参考

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0.0 | 2024-03-14 | 初始版本，完整实现Barra CNE6模型 |
| v1.1.0 | 2024-03-14 | 添加内存优化版本，支持8GB内存 |

---

## 作者与维护

**开发团队**: cf_quant项目组  
**最后更新**: 2024-03-14  
**许可证**: MIT

---

## 问题反馈

如有问题，请检查:
1. 测试报告文件 (`test_output/complete_test_report_*.txt`)
2. 错误日志中的traceback信息
3. qlib数据完整性
4. 内存使用情况
