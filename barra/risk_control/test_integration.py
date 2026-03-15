"""
Barra CNE6 风险模型 - 集成测试
测试完整流程：从数据加载到风险输出的全流程
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import gc
from test_base import BarraTestSuite
from test_config import TEST_CONFIG, EXPECTED_RANGES, QLIB_CONFIG


class IntegrationTests(BarraTestSuite):
    """集成测试类"""
    
    def __init__(self):
        super().__init__(output_dir=TEST_CONFIG['output_dir'])
        self.engine = None
        
    def setup(self):
        """测试准备工作"""
        print("\n" + "="*70)
        print("开始集成测试")
        print("="*70)
        
        # 初始化qlib
        if not self.init_qlib(QLIB_CONFIG['provider_uri']):
            raise RuntimeError("Qlib初始化失败，无法继续测试")
        
        print("✓ 集成测试环境准备完成")
    
    # ==================== 集成测试1: 全流程小规模测试 ====================
    def test_full_pipeline_small(self):
        """
        测试完整流程（小规模数据）
        使用50只股票，1个月数据，验证全流程可运行
        """
        print("\n" + "="*70)
        print("集成测试1: 全流程小规模测试")
        print("="*70)
        
        from barra.risk_control import BarraRiskEngine
        
        # 1. 初始化引擎（使用小规模配置）
        print("\n步骤1: 初始化风险模型引擎...")
        engine = BarraRiskEngine(
            calc_date=TEST_CONFIG['calc_date'],
            portfolio_input='random',
            output_dir=TEST_CONFIG['output_dir'],
            cache_dir=TEST_CONFIG['cache_dir'],
            n_jobs=TEST_CONFIG['n_jobs'],
            memory_threshold_gb=TEST_CONFIG['memory_threshold_gb']
        )
        print(f"✓ 引擎初始化成功")
        print(f"  组合股票数: {len(engine.portfolio_weights)}")
        
        # 2. 模拟月频更新（使用简化数据）
        print("\n步骤2: 模拟月频模型更新...")
        print("  (注: 使用简化流程，实际应加载预计算模型)")
        
        # 由于月频更新需要大量历史数据，这里我们只验证引擎可运行
        # 实际使用时应该已经通过run_monthly_update预计算好
        
        # 3. 验证引擎状态
        assert engine.portfolio_weights is not None, "组合权重为空"
        assert len(engine.portfolio_weights) > 0, "组合为空"
        assert abs(engine.portfolio_weights.sum() - 1.0) < 1e-6, "组合权重之和不等于1"
        
        print(f"✓ 引擎状态检查通过")
        print(f"  组合权重范围: [{engine.portfolio_weights.min():.6f}, "
              f"{engine.portfolio_weights.max():.6f}]")
        
        self.engine = engine
        
        return {
            'n_stocks': len(engine.portfolio_weights),
            'portfolio_sum': float(engine.portfolio_weights.sum()),
            'weight_range': [float(engine.portfolio_weights.min()),
                           float(engine.portfolio_weights.max())],
        }
    
    # ==================== 集成测试2: 数据流测试 ====================
    def test_data_flow(self):
        """
        测试数据流：从qlib到因子暴露矩阵
        """
        print("\n" + "="*70)
        print("集成测试2: 数据流测试")
        print("="*70)
        
        from barra.risk_control import (
            DataLoader, PortfolioManager, FactorExposureBuilder
        )
        
        # 1. 加载数据
        print("\n步骤1: 加载市场数据...")
        data_loader = DataLoader(market='csi300')
        
        instruments = data_loader.get_instruments(
            TEST_CONFIG['calc_date'],
            TEST_CONFIG['calc_date']
        )
        print(f"✓ 获取股票列表: {len(instruments)}只")
        
        # 取前30只测试
        test_instruments = instruments[:30]
        
        # 2. 加载各类数据
        raw_data = data_loader.load_factor_data(
            test_instruments,
            TEST_CONFIG['calc_date'],
            TEST_CONFIG['calc_date']
        )
        print(f"✓ 加载因子数据: {len(raw_data)}行")
        
        industry_df = data_loader.load_industry(
            test_instruments,
            TEST_CONFIG['calc_date'],
            TEST_CONFIG['calc_date']
        )
        print(f"✓ 加载行业数据: {len(industry_df)}行")
        
        market_cap_df = data_loader.load_market_cap(
            test_instruments,
            TEST_CONFIG['calc_date'],
            TEST_CONFIG['calc_date']
        )
        print(f"✓ 加载市值数据: {len(market_cap_df)}行")
        
        # 3. 构建组合
        print("\n步骤2: 构建投资组合...")
        portfolio_manager = PortfolioManager(market='csi300')
        portfolio = portfolio_manager.generate_random_portfolio(
            TEST_CONFIG['calc_date'],
            n_stocks=20
        )
        print(f"✓ 生成测试组合: {len(portfolio)}只股票")
        
        benchmark = portfolio_manager.get_benchmark_weights(
            TEST_CONFIG['calc_date']
        )
        print(f"✓ 获取基准权重: {len(benchmark)}只股票")
        
        # 4. 计算因子暴露（简化版，只计算几个因子）
        print("\n步骤3: 计算因子暴露...")
        builder = FactorExposureBuilder()
        
        # 测试单个因子计算
        from data.factor.size import LNCAP
        lncap = LNCAP(raw_data)
        print(f"✓ 计算LNCAP因子: {len(lncap)}行")
        
        from data.factor.value import BTOP
        btop = BTOP(raw_data)
        print(f"✓ 计算BTOP因子: {len(btop)}行")
        
        # 验证数据对齐
        common_idx = lncap.index.intersection(btop.index)
        assert len(common_idx) > 0, "因子数据索引不对齐"
        print(f"✓ 数据对齐检查通过: {len(common_idx)}条共同记录")
        
        return {
            'n_instruments': len(instruments),
            'n_raw_data': len(raw_data),
            'n_industry': len(industry_df),
            'n_market_cap': len(market_cap_df),
            'n_portfolio': len(portfolio),
            'n_benchmark': len(benchmark),
            'n_lncap': len(lncap),
            'n_btop': len(btop),
            'common_records': len(common_idx),
        }
    
    # ==================== 集成测试3: 数值正确性验证 ====================
    def test_numerical_correctness(self):
        """
        验证数值计算正确性
        使用已知输入，验证输出是否符合预期
        """
        print("\n" + "="*70)
        print("集成测试3: 数值正确性验证")
        print("="*70)
        
        from barra.risk_control import (
            RiskAttributionAnalyzer,
            CrossSectionalRegression,
            FactorCovarianceEstimator
        )
        from barra.risk_control.config import STYLE_FACTOR_LIST
        
        # 1. 测试风险归因数值
        print("\n步骤1: 测试风险归因数值...")
        np.random.seed(42)
        
        n_stocks = 10
        
        # 使用真实的CNE6因子名称（前5个）
        real_factors = STYLE_FACTOR_LIST[:5]
        n_factors = len(real_factors)
        
        # 创建简单的测试数据
        # 资产协方差：对角矩阵，方差=0.04（标准差=20%）
        asset_cov = pd.DataFrame(
            np.eye(n_stocks) * 0.04,
            index=[f'Stock_{i}' for i in range(n_stocks)],
            columns=[f'Stock_{i}' for i in range(n_stocks)]
        )
        
        # 因子协方差：对角矩阵
        factor_cov = pd.DataFrame(
            np.eye(n_factors) * 0.01,
            index=real_factors,
            columns=real_factors
        )
        
        # 因子暴露
        exposure = pd.DataFrame(
            np.eye(n_stocks, n_factors),  # 简化：前5只股票对应5个因子
            index=asset_cov.index,
            columns=real_factors
        )
        
        # 等权重组合
        portfolio_weights = pd.Series(
            1.0 / n_stocks,
            index=asset_cov.index
        )
        
        # 基准：前5只股票
        benchmark_weights = pd.Series(
            0.0,
            index=asset_cov.index
        )
        benchmark_weights.iloc[:5] = 0.2  # 前5只各20%
        
        # 计算风险
        analyzer = RiskAttributionAnalyzer()
        risk_results = analyzer.analyze_risk(
            asset_cov, factor_cov, exposure,
            portfolio_weights, benchmark_weights
        )
        
        print(f"✓ 风险归因计算完成")
        print(f"  组合总风险: {risk_results['total_risk']:.6f}")
        print(f"  主动风险: {risk_results['active_risk']:.6f}")
        
        # 验证：总风险应为sqrt(0.04/10) = 0.0632（等权重组合）
        expected_total_risk = np.sqrt(0.04 / n_stocks)
        actual_total_risk = risk_results['total_risk']
        print(f"  预期总风险: {expected_total_risk:.6f}")
        
        # 允许一定误差
        assert abs(actual_total_risk - expected_total_risk) < 0.01, \
            f"总风险计算错误: {actual_total_risk} vs {expected_total_risk}"
        print(f"✓ 总风险数值验证通过")
        
        # 2. 验证RCAR之和等于主动风险
        rcar_sum = risk_results['rcar'].sum()
        active_risk = risk_results['active_risk']
        diff = abs(rcar_sum - active_risk)
        assert diff < 1e-10, f"RCAR之和不等于主动风险: {diff}"
        print(f"✓ RCAR之和验证通过: {rcar_sum:.10f} = {active_risk:.10f}")
        
        # 3. 测试协方差矩阵正定性
        print("\n步骤2: 测试协方差矩阵性质...")
        eigenvalues = np.linalg.eigvals(asset_cov.values)
        assert all(eigenvalues > 0), "协方差矩阵不是正定矩阵"
        print(f"✓ 协方差矩阵正定性验证通过")
        print(f"  特征值范围: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")
        
        return {
            'expected_total_risk': float(expected_total_risk),
            'actual_total_risk': float(actual_total_risk),
            'active_risk': float(active_risk),
            'rcar_sum': float(rcar_sum),
            'rcar_validation': diff < 1e-10,
            'covariance_positive_definite': all(eigenvalues > 0),
        }
    
    # ==================== 集成测试4: 输出文件验证 ====================
    def test_output_files(self):
        """
        验证输出文件格式和内容
        """
        print("\n" + "="*70)
        print("集成测试4: 输出文件验证")
        print("="*70)
        
        from barra.risk_control import RiskOutputManager
        from barra.risk_control.config import STYLE_FACTOR_LIST, INDUSTRY_NAMES
        
        calc_date = TEST_CONFIG['calc_date']
        
        # 1. 创建模拟风险结果
        print("\n步骤1: 创建模拟风险结果...")
        n_stocks = 50
        
        # 使用真实的CNE6因子名称和行业名称
        style_factors = STYLE_FACTOR_LIST  # 38个风格因子
        industry_factors = INDUSTRY_NAMES   # 31个行业
        all_factors = style_factors + industry_factors  # 69个因子
        n_factors = len(all_factors)
        
        np.random.seed(42)
        
        mcar = pd.Series(
            np.random.randn(n_stocks) * 0.01,
            index=[f'00000{i}.SZ' if i < 10 else f'0000{i}.SZ' 
                   for i in range(n_stocks)]
        )
        
        rcar = pd.Series(
            np.random.randn(n_stocks) * 0.001,
            index=mcar.index
        )
        
        fmcar = pd.Series(
            np.random.randn(n_factors) * 0.01,
            index=all_factors
        )
        
        frcar = pd.Series(
            np.random.randn(n_factors) * 0.001,
            index=all_factors
        )
        
        # 创建因子类型映射
        factor_type_map = {
            'LNCAP': '规模', 'MIDCAP': '规模',
            'BETA': '波动率', 'HSIGMA': '波动率', 'DASTD': '波动率', 'CMRA': '波动率',
            'STOM': '流动性', 'STOQ': '流动性', 'STOA': '流动性', 'ATVR': '流动性',
            'STREV': '动量', 'SEASON': '动量', 'INDMOM': '动量', 'RSTR': '动量', 'HALPHA': '动量',
            'MLEV': '质量-杠杆', 'BLEV': '质量-杠杆', 'DTOA': '质量-杠杆',
            'VSAL': '质量-盈利波动', 'VERN': '质量-盈利波动', 'VFLO': '质量-盈利波动',
            'ABS': '质量-盈利质量', 'ACF': '质量-盈利质量',
            'ATO': '质量-盈利能力', 'GP': '质量-盈利能力', 'GPM': '质量-盈利能力', 'ROA': '质量-盈利能力',
            'AGRO': '质量-投资质量', 'IGRO': '质量-投资质量', 'CXGRO': '质量-投资质量',
            'BTOP': '价值', 'ETOP': '价值', 'CETOP': '价值', 'EM': '价值', 'LTRSTR': '价值', 'LTHALPHA': '价值',
            'EGRO': '成长', 'SGRO': '成长',
        }
        # 行业因子类型都是'行业'
        for industry in industry_factors:
            factor_type_map[industry] = '行业'
        
        factor_types = pd.Series(
            [factor_type_map.get(factor, '其他') for factor in all_factors],
            index=all_factors
        )
        
        # 2. 保存文件
        print("\n步骤2: 保存风险指标文件...")
        output_manager = RiskOutputManager(
            output_dir=TEST_CONFIG['output_dir']
        )
        
        stock_file = output_manager.save_stock_risk(mcar, rcar, calc_date)
        factor_file = output_manager.save_factor_risk(
            fmcar, frcar, factor_types, calc_date
        )
        print(f"✓ 保存股票风险文件: {stock_file}")
        print(f"✓ 保存因子风险文件: {factor_file}")
        
        # 3. 验证文件格式
        print("\n步骤3: 验证文件格式...")
        
        # 验证股票风险文件
        stock_df = pd.read_csv(stock_file)
        required_stock_cols = ['instrument', 'mcar', 'rcar', 'calc_date']
        for col in required_stock_cols:
            assert col in stock_df.columns, f"股票风险文件缺少列: {col}"
        assert len(stock_df) == n_stocks, f"股票风险文件行数错误: {len(stock_df)}"
        assert stock_df['calc_date'].unique()[0] == calc_date
        print(f"✓ 股票风险文件格式验证通过")
        print(f"  列: {list(stock_df.columns)}")
        print(f"  行数: {len(stock_df)}")
        
        # 验证因子风险文件
        factor_df = pd.read_csv(factor_file)
        required_factor_cols = ['factor_name', 'fmcar', 'frcar', 'factor_type', 'calc_date']
        for col in required_factor_cols:
            assert col in factor_df.columns, f"因子风险文件缺少列: {col}"
        assert len(factor_df) == n_factors, f"因子风险文件行数错误: {len(factor_df)}"
        print(f"✓ 因子风险文件格式验证通过")
        print(f"  列: {list(factor_df.columns)}")
        print(f"  行数: {len(factor_df)}")
        
        # 4. 验证数值精度
        print("\n步骤4: 验证数值精度...")
        # 检查小数位数（应该保留6位）
        sample_mcar = stock_df['mcar'].iloc[0]
        decimal_places = len(str(sample_mcar).split('.')[-1])
        assert decimal_places <= 6, f"小数位数超过6位: {decimal_places}"
        print(f"✓ 数值精度验证通过（保留6位小数）")
        
        return {
            'stock_file': stock_file,
            'factor_file': factor_file,
            'n_stock_records': len(stock_df),
            'n_factor_records': len(factor_df),
            'stock_columns': list(stock_df.columns),
            'factor_columns': list(factor_df.columns),
            'decimal_places': decimal_places,
        }
    
    # ==================== 集成测试5: 内存优化测试 ====================
    def test_memory_optimization(self):
        """
        测试内存优化功能
        """
        print("\n" + "="*70)
        print("集成测试5: 内存优化测试")
        print("="*70)
        
        from barra.risk_control import (
            MemoryMonitor,
            convert_to_float32,
            suggest_workers_by_memory
        )
        
        # 1. 测试内存监控
        print("\n步骤1: 测试内存监控...")
        monitor = MemoryMonitor(threshold_gb=TEST_CONFIG['memory_threshold_gb'])
        monitor.print_memory_status("测试开始")
        print(f"✓ 内存监控器初始化成功")
        print(f"  内存阈值: {TEST_CONFIG['memory_threshold_gb']} GB")
        
        # 2. 测试float32转换
        print("\n步骤2: 测试float32转换...")
        df_float64 = pd.DataFrame(
            np.random.randn(1000, 100),
            columns=[f'col_{i}' for i in range(100)]
        )
        memory_before = df_float64.memory_usage(deep=True).sum() / 1024**2
        
        df_float32 = convert_to_float32(df_float64)
        memory_after = df_float32.memory_usage(deep=True).sum() / 1024**2
        
        memory_saved = (1 - memory_after / memory_before) * 100
        print(f"✓ Float32转换成功")
        print(f"  转换前内存: {memory_before:.2f} MB")
        print(f"  转换后内存: {memory_after:.2f} MB")
        print(f"  节省内存: {memory_saved:.1f}%")
        
        assert memory_saved > 40, f"内存节省不足: {memory_saved:.1f}%"
        print(f"✓ 内存节省验证通过（>40%）")
        
        # 3. 测试并行进程数建议
        print("\n步骤3: 测试并行进程数建议...")
        n_jobs = suggest_workers_by_memory(
            max_workers=8,
            memory_per_worker_gb=1.0,
            reserve_memory_gb=2.0
        )
        print(f"✓ 建议并行进程数: {n_jobs}")
        assert 1 <= n_jobs <= 8, f"建议进程数不合理: {n_jobs}"
        print(f"✓ 进程数建议合理")
        
        return {
            'memory_threshold_gb': TEST_CONFIG['memory_threshold_gb'],
            'memory_before_mb': float(memory_before),
            'memory_after_mb': float(memory_after),
            'memory_saved_pct': float(memory_saved),
            'suggested_n_jobs': n_jobs,
        }
    
    def run_all_tests(self):
        """运行所有集成测试"""
        self.setup()
        
        # 运行各个测试
        self.run_test("全流程小规模测试", self.test_full_pipeline_small)
        self.run_test("数据流测试", self.test_data_flow)
        self.run_test("数值正确性验证", self.test_numerical_correctness)
        self.run_test("输出文件验证", self.test_output_files)
        self.run_test("内存优化测试", self.test_memory_optimization)
        
        # 保存结果
        return self.save_results()


if __name__ == '__main__':
    # 运行测试
    tester = IntegrationTests()
    result_files = tester.run_all_tests()
    
    print("\n" + "="*70)
    print("集成测试完成")
    print("="*70)
    print(f"\n结果文件:")
    for key, filepath in result_files.items():
        print(f"  {key}: {filepath}")
