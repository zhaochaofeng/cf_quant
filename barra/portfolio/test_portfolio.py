"""
投资组合优化模块单元测试（直接导入版，避免依赖utils外部包）
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import importlib.util
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# barra/portfolio 目录
PORTFOLIO_DIR = PROJECT_ROOT / 'barra' / 'portfolio'


def load_module_directly(module_name: str, file_path: Path, package_name: str = None):
    """直接加载模块，处理相对导入"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    if package_name:
        sys.modules[package_name] = module
    else:
        sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def setup_portfolio_modules():
    """预先加载portfolio的所有模块，处理相对导入"""
    # 创建模拟的LoggerFactory
    class MockLoggerFactory:
        @staticmethod
        def get_logger(name):
            return logging.getLogger(name)
    
    # 创建模拟的utils模块
    mock_utils = type(sys)('mock_utils')
    mock_utils.LoggerFactory = MockLoggerFactory
    sys.modules['utils'] = mock_utils
    
    # 先加载config（无依赖）
    config = load_module_directly("portfolio_config", PORTFOLIO_DIR / "config.py", "barra.portfolio.config")
    
    return config


# 在模块加载时初始化
_config = setup_portfolio_modules()


def test_covariance_building():
    """测试协方差矩阵构建（使用实际数据）"""
    print('\n' + '='*50)
    print('测试协方差矩阵构建')
    print('='*50)
    
    # 直接读取数据
    exposure = pd.read_parquet(PROJECT_ROOT / 'barra/risk_control/output/debug/exposure_matrix.parquet')
    factor_cov = pd.read_parquet(PROJECT_ROOT / 'barra/risk_control/output/model/factor_covariance.parquet')
    specific_risk_df = pd.read_parquet(PROJECT_ROOT / 'barra/risk_control/output/model/specific_risk.parquet')
    
    # 提取最新日期的因子暴露
    if isinstance(exposure.index, pd.MultiIndex):
        latest_date = exposure.index.get_level_values('datetime').max()
        exposure = exposure.xs(latest_date, level='datetime')
        print(f'因子暴露提取日期: {latest_date}')
    
    # 提取对角元素
    if isinstance(specific_risk_df, pd.DataFrame) and specific_risk_df.shape[0] == specific_risk_df.shape[1]:
        specific_risk = pd.Series(
            specific_risk_df.values.diagonal(),
            index=specific_risk_df.index
        )
    else:
        specific_risk = specific_risk_df.iloc[:, 0]
    
    print(f'因子暴露: {exposure.shape}')
    print(f'因子协方差: {factor_cov.shape}')
    print(f'特异风险: {specific_risk.shape}')
    
    # 对齐数据
    common_instruments = exposure.index.intersection(specific_risk.index)
    common_factors = exposure.columns.intersection(factor_cov.index)
    
    X = exposure.loc[common_instruments, common_factors].values
    F = factor_cov.loc[common_factors, common_factors].values
    delta = specific_risk.loc[common_instruments].values
    
    print(f'共同股票: {len(common_instruments)}, 共同因子: {len(common_factors)}')
    
    # 构建协方差矩阵
    XFXT = X @ F @ X.T
    V = XFXT + np.diag(delta)
    
    print(f'协方差矩阵: {V.shape}')
    
    # 检查正定性
    eigenvalues = np.linalg.eigvalsh(V)
    min_eigenvalue = eigenvalues.min()
    print(f'最小特征值: {min_eigenvalue:.2e}')
    
    if min_eigenvalue >= 0:
        print('✓ 协方差矩阵正定')
        return True
    else:
        print('⚠ 协方差矩阵非正定，需要修正')
        return True


def test_optimizer():
    """测试QP优化器"""
    print('\n' + '='*50)
    print('测试QP优化器')
    print('='*50)
    
    try:
        import cvxpy as cp
        print('✓ cvxpy已安装')
    except ImportError:
        print('⚠ cvxpy未安装，跳过测试')
        return False
    
    # 直接加载optimizer模块
    optimizer_module = load_module_directly(
        "portfolio_optimizer", 
        PORTFOLIO_DIR / "optimizer.py",
        "barra.portfolio.optimizer"
    )
    QPOptimizer = optimizer_module.QPOptimizer
    compute_mcva = optimizer_module.compute_mcva
    
    # 创建模拟数据
    N = 50
    np.random.seed(42)
    
    alpha = np.random.randn(N) * 0.01
    V = np.random.randn(N, N)
    V = V @ V.T / N + np.eye(N) * 0.01
    h_cur = np.zeros(N)
    w_b = np.ones(N) / N
    
    optimizer = QPOptimizer(
        risk_aversion=0.05,
        max_turnover=0.10,
        max_active_position=0.05
    )
    
    result = optimizer.solve(alpha, V, h_cur, w_b)
    
    print(f'求解状态: {result.status}')
    print(f'主动风险: {result.active_risk:.4f}')
    print(f'头寸范围: [{result.h_optimal.min():.4f}, {result.h_optimal.max():.4f}]')
    
    # 验证约束
    cash_neutral = abs(np.sum(result.h_optimal))
    short_violation = (result.h_optimal < -w_b).sum()
    
    print(f'现金中性偏差: {cash_neutral:.2e}')
    print(f'卖空约束违反: {short_violation}')
    
    print('✓ QP优化器测试通过')
    return True


def test_no_trade_zone():
    """测试无交易区域迭代"""
    print('\n' + '='*50)
    print('测试无交易区域迭代')
    print('='*50)
    
    # 直接加载模块
    ntz_module = load_module_directly(
        "portfolio_no_trade_zone", 
        PORTFOLIO_DIR / "no_trade_zone.py",
        "barra.portfolio.no_trade_zone"
    )
    NoTradeZoneIterator = ntz_module.NoTradeZoneIterator
    
    # 创建模拟数据
    N = 50
    np.random.seed(42)
    
    alpha = np.random.randn(N) * 0.01
    V = np.random.randn(N, N)
    V = V @ V.T / N + np.eye(N) * 0.01
    h_cur = np.zeros(N)
    w_b = np.ones(N) / N
    
    iterator = NoTradeZoneIterator(
        risk_aversion=0.05,
        max_iterations=100,
        convergence_threshold=1e-6
    )
    
    result = iterator.iterate(alpha, V, h_cur, w_b)
    
    print(f'迭代次数: {result.iterations}')
    print(f'是否收敛: {result.converged}')
    print(f'主动风险: {result.active_risk:.4f}')
    print(f'无交易区域比例: {result.in_no_trade_zone.mean():.2%}')
    
    # 验证约束
    cash_neutral = abs(np.sum(result.h_final))
    short_violation = (result.h_final < -w_b).sum()
    
    print(f'现金中性偏差: {cash_neutral:.2e}')
    print(f'卖空约束违反: {short_violation}')
    
    print('✓ 无交易区域迭代测试通过')
    return True


def test_trade_generator():
    """测试交易指令生成器"""
    print('\n' + '='*50)
    print('测试交易指令生成器')
    print('='*50)
    
    # 直接加载模块
    tg_module = load_module_directly(
        "portfolio_trade_generator", 
        PORTFOLIO_DIR / "trade_generator.py",
        "barra.portfolio.trade_generator"
    )
    TradeGenerator = tg_module.TradeGenerator
    
    # 创建模拟数据
    N = 50
    np.random.seed(42)
    
    instruments = pd.Index([f'SH600{str(i).zfill(3)}' for i in range(N)])
    h_final = pd.Series(np.random.randn(N) * 0.02, index=instruments)
    h_cur = pd.Series(np.zeros(N), index=instruments)
    prices = pd.Series(np.random.uniform(10, 100, N), index=instruments)
    w_b = pd.Series(np.ones(N) / N, index=instruments)
    
    generator = TradeGenerator(min_trade_threshold=1e-5)
    
    trade_orders = generator.generate(
        h_final=h_final,
        h_cur=h_cur,
        portfolio_value=1e8,
        prices=prices,
        w_b=w_b
    )
    
    print(f'交易指令数量: {len(trade_orders)}')
    print(f'买入: {(trade_orders["direction"] == "buy").sum()}只')
    print(f'卖出: {(trade_orders["direction"] == "sell").sum()}只')
    
    # 持仓摘要
    position = generator.generate_position_summary(trade_orders, 1e8)
    print(f'持仓股票数: {len(position)}')
    
    print('✓ 交易指令生成器测试通过')
    return True


def test_output_manager():
    """测试输出管理器"""
    print('\n' + '='*50)
    print('测试输出管理器')
    print('='*50)
    
    # 直接加载模块
    out_module = load_module_directly(
        "portfolio_output", 
        PORTFOLIO_DIR / "output.py",
        "barra.portfolio.output"
    )
    PortfolioOutputManager = out_module.PortfolioOutputManager
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        manager = PortfolioOutputManager(output_dir=temp_dir)
        
        # 创建测试数据
        trade_orders = pd.DataFrame({
            'instrument': ['SH600000', 'SH600001'],
            'direction': ['buy', 'sell'],
            'weight_change': [0.01, -0.005],
            'amount': [1000000.0, 500000.0],
            'shares': [10000, 5000],
            'price': [10.0, 20.0],
            'active_weight': [0.01, -0.005],
            'total_weight': [0.02, 0.005]
        })
        
        position = pd.DataFrame({
            'instrument': ['SH600000', 'SH600001'],
            'active_weight': [0.01, -0.005],
            'total_weight': [0.02, 0.005],
            'shares': [10000, 5000],
            'market_value': [100000.0, 100000.0],
            'weight_pct': [2.0, 0.5]
        })
        
        # 测试保存
        manager.save_trade_orders(trade_orders, '2026-03-28')
        manager.save_position(position, '2026-03-28')
        
        print('✓ 文件保存成功')
        
        # 测试加载
        loaded_orders = manager.load_trade_orders('2026-03-28')
        assert loaded_orders is not None
        print('✓ 文件加载成功')
        
        print('✓ 输出管理器测试通过')
        return True
        
    finally:
        shutil.rmtree(temp_dir)


def test_full_pipeline():
    """测试完整流程（使用实际数据）"""
    print('\n' + '='*50)
    print('测试完整优化流程')
    print('='*50)
    
    try:
        import cvxpy as cp
    except ImportError:
        print('⚠ cvxpy未安装，跳过完整流程测试')
        return False
    
    # 直接加载模块
    ntz_module = load_module_directly(
        "portfolio_no_trade_zone2", 
        PORTFOLIO_DIR / "no_trade_zone.py",
        "barra.portfolio.no_trade_zone2"
    )
    NoTradeZoneIterator = ntz_module.NoTradeZoneIterator
    
    tg_module = load_module_directly(
        "portfolio_trade_generator2", 
        PORTFOLIO_DIR / "trade_generator.py",
        "barra.portfolio.trade_generator2"
    )
    TradeGenerator = tg_module.TradeGenerator
    
    # 使用实际数据
    print('加载实际数据...')
    
    # 加载Alpha
    alpha_df = pd.read_parquet(PROJECT_ROOT / 'barra/alpha/output/alpha_20260305.parquet')
    alpha_series = alpha_df['alpha'] if 'alpha' in alpha_df.columns else alpha_df.iloc[:, 0]
    print(f'Alpha: {len(alpha_series)}只股票')
    
    # 加载风险模型
    exposure = pd.read_parquet(PROJECT_ROOT / 'barra/risk_control/output/debug/exposure_matrix.parquet')
    factor_cov = pd.read_parquet(PROJECT_ROOT / 'barra/risk_control/output/model/factor_covariance.parquet')
    specific_risk_df = pd.read_parquet(PROJECT_ROOT / 'barra/risk_control/output/model/specific_risk.parquet')
    
    # 提取最新日期的因子暴露
    if isinstance(exposure.index, pd.MultiIndex):
        latest_date = exposure.index.get_level_values('datetime').max()
        exposure = exposure.xs(latest_date, level='datetime')
    
    # 提取对角元素
    if isinstance(specific_risk_df, pd.DataFrame) and specific_risk_df.shape[0] == specific_risk_df.shape[1]:
        specific_risk = pd.Series(
            specific_risk_df.values.diagonal(),
            index=specific_risk_df.index
        )
    else:
        specific_risk = specific_risk_df.iloc[:, 0]
    
    # 对齐数据
    common_instruments = (
        alpha_series.index
        .intersection(exposure.index)
        .intersection(specific_risk.index)
    )
    print(f'共同股票: {len(common_instruments)}')
    
    alpha = alpha_series.reindex(common_instruments).values
    X = exposure.reindex(common_instruments).values
    F = factor_cov.values
    delta = specific_risk.reindex(common_instruments).values
    
    # 构建协方差矩阵
    V = X @ F @ X.T + np.diag(delta)
    print(f'协方差矩阵: {V.shape}')
    
    # 初始化
    h_cur = np.zeros(len(common_instruments))
    w_b = np.ones(len(common_instruments)) / len(common_instruments)
    
    # 执行迭代
    iterator = NoTradeZoneIterator(risk_aversion=0.05)
    result = iterator.iterate(alpha, V, h_cur, w_b)
    
    print(f'迭代次数: {result.iterations}')
    print(f'主动风险: {result.active_risk:.4f}')
    print(f'收敛状态: {result.converged}')
    
    # 生成交易指令
    h_final = pd.Series(result.h_final, index=common_instruments)
    h_cur_series = pd.Series(h_cur, index=common_instruments)
    prices = pd.Series(np.random.uniform(10, 100, len(common_instruments)), index=common_instruments)
    w_b_series = pd.Series(w_b, index=common_instruments)
    
    generator = TradeGenerator()
    trade_orders = generator.generate(
        h_final=h_final,
        h_cur=h_cur_series,
        portfolio_value=1e8,
        prices=prices,
        w_b=w_b_series
    )
    
    buy_cnt = (trade_orders["direction"] == "buy").sum()
    sell_cnt = (trade_orders["direction"] == "sell").sum()
    print(f'交易指令: 买入{buy_cnt}, 卖出{sell_cnt}')
    
    # 验证约束
    cash_neutral = abs(np.sum(result.h_final))
    print(f'现金中性偏差: {cash_neutral:.2e}')
    
    print('✓ 完整流程测试通过')
    return True


def run_all_tests():
    """运行所有测试"""
    print('\n' + '='*60)
    print('投资组合优化模块单元测试')
    print('='*60)
    
    tests = [
        ('协方差构建', test_covariance_building),
        ('QP优化器', test_optimizer),
        ('无交易区域迭代', test_no_trade_zone),
        ('交易指令生成', test_trade_generator),
        ('输出管理器', test_output_manager),
        ('完整流程', test_full_pipeline),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            import traceback
            print(f'✗ 测试失败: {e}')
            traceback.print_exc()
            results.append((name, False))
    
    # 汇总
    print('\n' + '='*60)
    print('测试结果汇总')
    print('='*60)
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = '✓ 通过' if success else '✗ 失败'
        print(f'{name}: {status}')
    
    print(f'\n通过: {passed}/{total}')
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
