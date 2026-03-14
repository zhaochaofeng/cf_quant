# Barra CNE6 风险模型测试文档

## 测试概述

本测试套件使用qlib真实数据对Barra CNE6风险模型进行完整的功能验证和数值正确性测试。

## 测试文件结构

```
barra/risk_control/
├── test_config.py              # 测试配置
├── test_base.py                # 测试基类和工具
├── test_modules.py             # 模块化单元测试
├── test_integration.py         # 集成测试
├── test_runner.py              # 主测试运行脚本
└── test_output/                # 测试结果输出目录
    ├── complete_test_report_*.txt    # 综合测试报告
    ├── complete_test_summary_*.csv   # 测试摘要
    └── complete_test_details_*.json  # 详细结果
```

## 环境要求

```bash
# 激活conda环境（已安装statsmodels）
conda activate python311-tf210

# 确保已安装依赖
pip install pandas numpy statsmodels

# qlib数据已准备好
# 数据路径: ~/.qlib/qlib_data/custom_data_hfq
```

## 运行测试

### 方法1：运行所有测试（推荐）

```bash
# 在cf_quant目录下运行
python barra/risk_control/test_runner.py
```

### 方法2：单独运行模块化单元测试

```bash
# 测试各个独立模块
python barra/risk_control/test_modules.py
```

测试内容包括：
- 数据加载模块（DataLoader）
- 组合管理模块（PortfolioManager）
- 因子暴露模块（FactorExposureBuilder）
- 协方差估计模块（FactorCovarianceEstimator）
- 风险归因模块（RiskAttributionAnalyzer）
- 输出模块（RiskOutputManager）

### 方法3：单独运行集成测试

```bash
# 测试完整流程
python barra/risk_control/test_integration.py
```

测试内容包括：
- 全流程小规模测试
- 数据流测试（qlib → 因子暴露）
- 数值正确性验证
- 输出文件验证
- 内存优化测试

## 测试配置

在 `test_config.py` 中可以修改测试参数：

```python
TEST_CONFIG = {
    'test_start_date': '2024-01-01',    # 测试开始日期
    'test_end_date': '2024-03-01',      # 测试结束日期
    'calc_date': '2024-03-01',          # 计算日期
    'test_n_stocks': 50,                 # 测试股票数量
    'history_months': 12,                # 历史数据月数
    'n_jobs': 2,                         # 并行进程数
    'memory_threshold_gb': 6.0,          # 内存阈值
}
```

## 测试结果

### 输出文件格式

测试完成后会生成以下文件：

1. **文本报告** (`complete_test_report_YYYYMMDD_HHMMSS.txt`)
   - 完整的测试结果文本
   - 每个测试的通过/失败状态
   - 执行时间和错误信息

2. **CSV摘要** (`complete_test_summary_YYYYMMDD_HHMMSS.csv`)
   - 测试名称、通过状态、执行时间
   - 便于Excel分析

3. **JSON详细结果** (`complete_test_details_YYYYMMDD_HHMMSS.json`)
   - 包含所有测试的详细信息
   - 便于程序化处理

### 测试内容详解

#### 1. 数据加载模块测试
- ✓ 获取股票列表
- ✓ 加载收益率数据
- ✓ 加载市值数据
- ✓ 加载行业数据
- ✓ 数据格式验证

#### 2. 组合管理模块测试
- ✓ 获取基准权重（沪深300市值加权）
- ✓ 生成随机组合
- ✓ 计算主动权重
- ✓ 验证权重归一化

#### 3. 因子暴露模块测试
- ✓ 原始因子计算（LNCAP、BTOP等）
- ✓ 去极值处理（中位数MAD方法）
- ✓ 行业/市值中性化
- ✓ 正交化处理
- ✓ 标准化处理

#### 4. 协方差矩阵估计测试
- ✓ 样本协方差估计
- ✓ 协方差矩阵正定性
- ✓ 相关系数矩阵
- ✓ 收缩估计（可选）

#### 5. 风险归因模块测试
- ✓ MCAR计算
- ✓ RCAR计算
- ✓ FMCAR计算
- ✓ FRCAR计算
- ✓ RCAR之和 = 主动风险验证
- ✓ FRCAR性质验证

#### 6. 输出模块测试
- ✓ 股票风险CSV保存
- ✓ 因子风险CSV保存
- ✓ 文件格式验证
- ✓ 数值精度验证（6位小数）

#### 7. 数值正确性验证
- ✓ 总风险计算
- ✓ 主动风险计算
- ✓ 协方差矩阵特征值
- ✓ 风险分解一致性

#### 8. 内存优化测试
- ✓ float32转换（节省50%内存）
- ✓ 内存监控
- ✓ 并行进程数自动调整

## 测试验证要点

### 数值验证

1. **权重归一化**
   - 组合权重之和 = 1.0
   - 基准权重之和 = 1.0
   - 主动权重之和 ≈ 0.0

2. **风险指标范围**
   - 总风险: (0, 1)
   - 主动风险: (0, 1)
   - MCAR: (-1, 1)
   - RCAR: (-0.1, 0.1)

3. **一致性验证**
   - RCAR之和 = 主动风险
   - 协方差矩阵正定（所有特征值>0）
   - 相关系数 ∈ [-1, 1]

### 文件格式验证

1. **股票风险文件** (`stock_risk_YYYYMMDD.csv`)
   - 必需列: instrument, mcar, rcar, calc_date
   - 每只股票一行
   - 数值保留6位小数

2. **因子风险文件** (`factor_risk_YYYYMMDD.csv`)
   - 必需列: factor_name, fmcar, frcar, factor_type, calc_date
   - 每个因子一行
   - 按factor_type分组排序

## 常见问题

### Q1: 测试运行时间过长？

A: 可以调整 `test_config.py` 中的参数：
```python
TEST_CONFIG = {
    'test_n_stocks': 20,        # 减少测试股票数
    'n_jobs': 4,                 # 增加并行进程
}
```

### Q2: 内存不足？

A: 
1. 减少测试股票数：`test_n_stocks = 20`
2. 降低并行进程：`n_jobs = 1`
3. 启用增量模式：`use_incremental = True`

### Q3: qlib数据加载失败？

A: 检查：
1. qlib是否正确安装
2. 数据路径是否正确：`~/.qlib/qlib_data/custom_data_hfq`
3. 数据是否完整（至少包含2024年数据）

### Q4: 某些因子计算失败？

A: 可能原因：
1. 原始数据字段缺失
2. 股票停牌导致数据不足
3. 计算日期非交易日

## 测试结果解读

### 通过标准

- 所有模块测试通过
- 数值计算误差 < 1e-6
- 输出文件格式正确
- 内存使用在阈值内

### 失败处理

如果测试失败：

1. 查看文本报告中的错误信息
2. 检查JSON详细结果中的traceback
3. 验证测试配置是否正确
4. 确认qlib数据完整性

## 扩展测试

### 添加新的测试用例

在 `test_modules.py` 或 `test_integration.py` 中添加：

```python
def test_your_feature(self):
    """测试新功能"""
    print("\n测试新功能...")
    
    # 1. 准备数据
    # ...
    
    # 2. 执行测试
    # ...
    
    # 3. 验证结果
    self.assert_not_none(result)
    self.assert_in_range(value, min_val, max_val)
    
    return {'detail': 'value'}
```

然后在 `run_all_tests()` 中添加：
```python
self.run_test("新功能测试", self.test_your_feature)
```

## 联系与支持

如有问题，请检查：
1. 测试日志文件
2. 错误堆栈信息
3. 配置文件设置

---

**文档版本**: 1.0  
**最后更新**: 2024-03-14
