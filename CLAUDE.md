# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

定量金融研究平台 (Quantitative Finance Research Platform)，基于 Qlib 框架的 A 股多因子选股、投资组合优化系统。核心流水线：

**数据 -> 因子 -> 策略(ML模型) -> Alpha预测 -> 风险模型 -> 组合优化 -> 交易指令**

## Quick Start

```bash
conda activate python3
python workflow_by_code.py          # Qlib 完整流程: 训练+回测
python strategy/framework.py train  # 模型训练框架 (支持 gbm/transformer)
bash test.sh                         # 手动测试
```

- 无 pytest/unittest，使用 `test.py`, `test2.py` 等手动验证脚本
- 配置文件: `config.yaml` (数据库/API密钥, 不提交), `features.yaml` (字段定义)

## Code Architecture

### 1. Data Layer (`data/`)

数据获取与因子构建流水线。数据源: TuShare, BaoStock, JoinQuant。

```
data/
├── process_data.py          # 基类: Base -> ProcessData -> TSCommonData/TSTradeDailyData/BaoTradeDailyData
├── factor/factor.py         # Qlib 因子构建 + IC 计算 + MySQL 写入
├── factor/factor_func.py    # 技术指标因子函数 (MACD, BOLL, KDJ, RSI, etc.)
├── factor/                  # Barra CNE6 风格因子: size, value, momentum, volatility, liquidity, quality, growth
└── trade_daily/valuation/income/balance/cashflow/  # 各维度的 ETL 脚本
```

### 2. Strategy Layer (`strategy/`)

Qlib 驱动的 ML 模型训练与预测。

```
strategy/
├── framework.py             # 通用模型训练框架 (Qlib Experiment + Fire CLI)
├── model.py                 # LGBModel2, TransformerModel2 定义
├── dataset.py               # 自定义数据集处理 (ExpAlpha158, AlphaExpandHandler)
├── lightGBM/                # 滚动训练: lightgbm_alpha_rolling.py (Qlib RollingStrategy + OnlineManager)
└── master/                  # 多模型集成策略
```

关键模式: `prepare_data_config()` -> `prepare_model_config()` -> `model.fit(dataset)` -> `backtest()`

### 3. Barra Risk Model (`barra/`)

三模块架构，每个模块独立运行 `run_daily.py`，数据通过文件/MQ传递。

```
barra/
├── alpha/                   # Alpha预测引擎
│   └── alpha_engine.py      # AlphaEngine: 数据加载 -> 标准化 -> 残差波动率 -> 情形判断 -> IC估计 -> Alpha合成
├── risk_control/            # 风险模型 (Barra CNE6)
│   ├── barra_engine.py      # BarraEngine: 驱动整个风险模型流水线
│   ├── factor_exposure.py   # 因子暴露计算
│   ├── covariance.py        # 因子协方差矩阵 (EWMA/Newey-West)
│   ├── cross_sectional.py   # 横截面WLS回归: fit_multi_periods() 估计因子收益率
│   └── specific_risk.py     # 特异风险估计
└── portfolio/               # 组合优化
    ├── portfolio_engine.py  # PortfolioEngine: 数据加载 -> 协方差 -> QP优化 -> 无交易区域迭代 -> 交易指令
    ├── optimizer.py         # cvxpy QP求解: min 0.5*h'Qh - α'h + 交易成本
    ├── no_trade_zone.py     # 无交易区域迭代算法 (边际贡献法)
    └── trade_generator.py   # 交易指令生成
```

### 4. Utils (`utils/`)

共享工具模块:

- `utils.py` — 数据库连接 (MySQL/Redis), 数据API (TuShare/BaoStock/JQ), 交易日工具, 重试装饰器
- `preprocess.py` — winsorize, standardize, neutralize
- `stats.py` — WLS 回归
- `convex.py` — constrained_wls (cvxpy带线性约束的WLS)
- `logger.py` — LoggerFactory
- `dt.py` — DateTimeUtils
- `multiprocess.py` — multiprocessing_wrapper
- `backtest.py` — RollingPortAnaRecord
- `qlib_ops.py` — PTTM
- `qlib_processor.py` — Winsorize processor

### Data Flow

```
TuShare/BaoStock/JQ
  -> data/process_data.py (ETL to MySQL)
    -> data/factor/factor.py (因子构建, IC评估)
      -> strategy/ (ML模型训练 & 预测)
        -> barra/alpha/ (Alpha合成)
          -> barra/risk_control/ (风险模型)
            -> barra/portfolio/ (组合优化 -> 交易指令)
```

### Key Design Patterns

- **Pipeline模式**: AlphaEngine / PortfolioEngine / BarraEngine 编排多步骤流水线
- **Mixin继承**: factor.py 中 Size / Volatility / Momentum 等通过多继承组合
- **LazyProperty**: 因子按需计算，结果缓存在实例 __dict__ 中
- **配置驱动**: Qlib 任务用嵌套 dict 配置，通过 init_instance_by_config 实例化
- **Qlib Experiment**: 用 MLflow 管理实验记录 (mlruns/)

## Code Style

- 4 spaces indentation, ~100 chars per line
- `snake_case` for functions/vars, `PascalCase` for classes
- 类型注解, Google-style docstrings (注释混合中英文)
- pandas MultiIndex (instrument, datetime) 是核心数据格式
- 导入顺序: 标准库 -> 第三方 (numpy, pandas, qlib) -> 本地模块
- 日志使用 `LoggerFactory.get_logger(__name__)`

## Hard Rule
- 提交代码到 git 前必须询问
