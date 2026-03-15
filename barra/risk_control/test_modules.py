"""
Barra CNE6 风险模型 - 模块化单元测试
测试各个独立模块的功能
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from test_base import BarraTestSuite
from test_config import TEST_CONFIG, EXPECTED_RANGES, QLIB_CONFIG


class ModuleTests(BarraTestSuite):
    """模块化单元测试类"""
    
    def __init__(self):
        super().__init__(output_dir=TEST_CONFIG['output_dir'])
        self.data_loader = None
        self.portfolio_manager = None
        
    def setup(self):
        """测试准备工作"""
        print("\n" + "="*70)
        print("开始模块化单元测试")
        print("="*70)
        
        # 初始化qlib
        if not self.init_qlib(QLIB_CONFIG['provider_uri']):
            raise RuntimeError("Qlib初始化失败，无法继续测试")
        
        # 导入模块
        from barra.risk_control import DataLoader, PortfolioManager
        self.data_loader = DataLoader(market='csi300')
        self.portfolio_manager = PortfolioManager(market='csi300')
        
        print("✓ 测试环境准备完成")
    
    # ==================== 测试1: 数据加载模块 ====================
    def test_data_loader(self):
        """测试数据加载器"""
        print("\n测试数据加载模块...")
        
        # 1.1 测试获取股票列表
        instruments = self.data_loader.get_instruments(
            TEST_CONFIG['test_start_date'],
            TEST_CONFIG['test_end_date']
        )
        self.assert_not_none(instruments, "股票列表为None")
        assert len(instruments) > 0, "股票列表为空"
        print(f"✓ 获取股票列表: {len(instruments)}只股票")
        
        # 1.2 测试加载收益率数据
        returns = self.data_loader.load_returns(
            instruments[:TEST_CONFIG['test_n_stocks']],
            TEST_CONFIG['calc_date'],
            TEST_CONFIG['calc_date']
        )
        self.assert_valid_dataframe(returns, min_rows=1, 
                                   required_columns=['return'])
        print(f"✓ 加载收益率数据: {len(returns)}条记录")
        
        # 1.3 测试加载市值数据
        market_cap = self.data_loader.load_market_cap(
            instruments[:TEST_CONFIG['test_n_stocks']],
            TEST_CONFIG['calc_date'],
            TEST_CONFIG['calc_date']
        )
        self.assert_valid_dataframe(market_cap, min_rows=1,
                                   required_columns=['circ_mv', 'total_mv'])
        print(f"✓ 加载市值数据: {len(market_cap)}条记录")
        
        # 1.4 测试加载行业数据
        industry = self.data_loader.load_industry(
            instruments[:TEST_CONFIG['test_n_stocks']],
            TEST_CONFIG['calc_date'],
            TEST_CONFIG['calc_date']
        )
        self.assert_valid_dataframe(industry, min_rows=1,
                                   required_columns=['industry_code'])
        print(f"✓ 加载行业数据: {len(industry)}条记录")
        
        return {
            'n_instruments': len(instruments),
            'n_returns': len(returns),
            'n_market_cap': len(market_cap),
            'n_industry': len(industry),
        }
    
    # ==================== 测试2: 组合管理模块 ====================
    def test_portfolio_manager(self):
        """测试组合管理器"""
        print("\n测试组合管理模块...")
        
        # 2.1 测试获取基准权重
        benchmark = self.portfolio_manager.get_benchmark_weights(
            TEST_CONFIG['calc_date']
        )
        self.assert_not_none(benchmark, "基准权重为None")
        assert len(benchmark) > 0, "基准权重为空"
        assert abs(benchmark.sum() - 1.0) < 1e-6, "基准权重之和不等于1"
        print(f"✓ 获取基准权重: {len(benchmark)}只股票")
        print(f"  权重范围: [{benchmark.min():.6f}, {benchmark.max():.6f}]")
        
        # 2.2 测试生成随机组合
        random_portfolio = self.portfolio_manager.generate_random_portfolio(
            TEST_CONFIG['calc_date'],
            n_stocks=TEST_CONFIG['test_n_stocks']
        )
        self.assert_not_none(random_portfolio, "随机组合为None")
        assert len(random_portfolio) == TEST_CONFIG['test_n_stocks'], \
            f"随机组合股票数不正确: {len(random_portfolio)}"
        assert abs(random_portfolio.sum() - 1.0) < 1e-6, "组合权重之和不等于1"
        print(f"✓ 生成随机组合: {len(random_portfolio)}只股票")
        print(f"  等权重: {random_portfolio.iloc[0]:.6f}")
        
        # 2.3 测试计算主动权重
        active_weights = self.portfolio_manager.calculate_active_weights(
            random_portfolio, benchmark
        )
        self.assert_not_none(active_weights, "主动权重为None")
        # 主动权重之和应接近0
        active_sum = active_weights.sum()
        assert abs(active_sum) < 1e-6, f"主动权重之和不为0: {active_sum}"
        print(f"✓ 计算主动权重: {len(active_weights)}只股票")
        print(f"  主动权重之和: {active_sum:.10f}")
        
        return {
            'n_benchmark': len(benchmark),
            'n_portfolio': len(random_portfolio),
            'n_active': len(active_weights),
            'active_weights_sum': active_sum,
        }
    
    # ==================== 测试3: 因子计算模块 ====================
    def test_factor_exposure(self):
        """测试因子暴露矩阵构建"""
        print("\n测试因子暴露模块...")
        
        from barra.risk_control import FactorExposureBuilder
        from barra.risk_control import CNE6_STYLE_FACTORS
        
        # 3.1 测试因子暴露构建器初始化
        builder = FactorExposureBuilder()
        self.assert_not_none(builder, "因子暴露构建器为None")
        print("✓ 因子暴露构建器初始化成功")
        
        # 3.2 加载测试数据（小规模）
        instruments = self.data_loader.get_instruments(
            TEST_CONFIG['calc_date'],
            TEST_CONFIG['calc_date']
        )[:20]  # 只用20只股票测试
        
        raw_data = self.data_loader.load_factor_data(
            instruments,
            TEST_CONFIG['calc_date'],
            TEST_CONFIG['calc_date']
        )
        self.assert_valid_dataframe(raw_data, min_rows=10)
        print(f"✓ 加载原始数据: {len(raw_data)}行, {len(raw_data.columns)}列")
        
        # 3.3 测试单个因子计算
        from data.factor.size import LNCAP
        lncap_result = LNCAP(raw_data)
        self.assert_valid_dataframe(lncap_result, min_rows=10, 
                                   required_columns=['LNCAP'])
        print(f"✓ 计算LNCAP因子: {len(lncap_result)}行")
        
        # 3.4 验证因子值范围
        self.assert_in_range(lncap_result['LNCAP'], -10, 50, 
                           "LNCAP因子值超出合理范围")
        print(f"  LNCAP范围: [{lncap_result['LNCAP'].min():.2f}, "
              f"{lncap_result['LNCAP'].max():.2f}]")
        
        # 3.5 测试去极值函数
        # 创建带MultiIndex的测试数据（模拟真实的instrument-datetime索引）
        test_index = pd.MultiIndex.from_tuples(
            [(f'Stock_{i}', '2024-03-01') for i in range(10)],
            names=['instrument', 'datetime']
        )
        test_data = pd.DataFrame({
            'factor': [1, 2, 3, 4, 5, 100, -100, 3, 4, 5]
        }, index=test_index)
        winsorized = builder.winsorize_factors(test_data, method='median')
        self.assert_valid_dataframe(winsorized, min_rows=10)
        # 极值应该被截断
        assert winsorized['factor'].max() < 50, "去极值未生效"
        assert winsorized['factor'].min() > -50, "去极值未生效"
        print(f"✓ 去极值测试通过")
        print(f"  原始范围: [{test_data['factor'].min()}, {test_data['factor'].max()}]")
        print(f"  处理后范围: [{winsorized['factor'].min():.2f}, "
              f"{winsorized['factor'].max():.2f}]")
        
        return {
            'n_raw_data': len(raw_data),
            'n_factors': len(CNE6_STYLE_FACTORS),
            'lncap_mean': float(lncap_result['LNCAP'].mean()),
            'lncap_std': float(lncap_result['LNCAP'].std()),
        }
    
    # ==================== 测试4: 协方差矩阵估计 ====================
    def test_covariance_estimation(self):
        """测试协方差矩阵估计"""
        print("\n测试协方差矩阵估计模块...")
        
        from barra.risk_control import FactorCovarianceEstimator
        from barra.risk_control.config import STYLE_FACTOR_LIST
        
        # 4.1 创建模拟因子收益率数据
        np.random.seed(42)
        n_periods = 60  # 5年数据
        
        # 使用真实的CNE6因子名称（前10个）
        factor_names = STYLE_FACTOR_LIST[:10]
        n_factors = len(factor_names)
        
        dates = pd.date_range('2019-01-01', periods=n_periods, freq='M')
        
        # 生成相关因子收益率
        factor_returns = pd.DataFrame(
            np.random.randn(n_periods, n_factors) * 0.05,
            index=dates,
            columns=factor_names
        )
        
        print(f"✓ 创建模拟数据: {n_periods}期, {n_factors}个因子")
        
        # 4.2 测试样本协方差估计
        estimator = FactorCovarianceEstimator(history_window=36)
        cov_matrix = estimator.estimate_sample_covariance(factor_returns)
        
        self.assert_not_none(cov_matrix, "协方差矩阵为None")
        assert cov_matrix.shape == (n_factors, n_factors), \
            f"协方差矩阵维度错误: {cov_matrix.shape}"
        
        # 检查对角线（方差）为正
        variances = np.diag(cov_matrix.values)
        assert all(v > 0 for v in variances), "方差为负值"
        
        print(f"✓ 协方差矩阵估计成功: {cov_matrix.shape}")
        print(f"  平均方差: {variances.mean():.6f}")
        print(f"  方差范围: [{variances.min():.6f}, {variances.max():.6f}]")
        
        # 4.3 测试相关系数矩阵
        corr_matrix = estimator.get_correlation_matrix()
        self.assert_not_none(corr_matrix, "相关系数矩阵为None")
        
        # 对角线应为1
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr = corr_matrix.abs().max().max()
        assert max_corr <= 1.0, f"相关系数大于1: {max_corr}"
        
        print(f"✓ 相关系数矩阵计算成功")
        print(f"  最大绝对相关系数: {max_corr:.4f}")
        
        return {
            'n_periods': n_periods,
            'n_factors': n_factors,
            'cov_shape': cov_matrix.shape,
            'mean_variance': float(variances.mean()),
            'max_correlation': float(max_corr),
        }
    
    # ==================== 测试5: 风险归因计算 ====================
    def test_risk_attribution(self):
        """测试风险归因计算"""
        print("\n测试风险归因模块...")
        
        from barra.risk_control import RiskAttributionAnalyzer
        
        # 5.1 创建模拟数据
        from barra.risk_control.config import STYLE_FACTOR_LIST
        
        n_stocks = 50
        
        # 使用真实的CNE6因子名称（前10个）
        real_factors = STYLE_FACTOR_LIST[:10]
        n_factors = len(real_factors)
        
        np.random.seed(42)
        
        # 资产协方差矩阵（对角矩阵简化）
        asset_cov = pd.DataFrame(
            np.diag(np.random.uniform(0.01, 0.05, n_stocks)),
            index=[f'Stock_{i}' for i in range(n_stocks)],
            columns=[f'Stock_{i}' for i in range(n_stocks)]
        )
        
        # 因子协方差矩阵
        factor_cov = pd.DataFrame(
            np.eye(n_factors) * 0.01,
            index=real_factors,
            columns=real_factors
        )
        
        # 因子暴露
        exposure = pd.DataFrame(
            np.random.randn(n_stocks, n_factors) * 0.1,
            index=asset_cov.index,
            columns=real_factors
        )
        
        # 组合权重
        portfolio_weights = pd.Series(
            np.random.uniform(0.5, 1.5, n_stocks),
            index=asset_cov.index
        )
        portfolio_weights = portfolio_weights / portfolio_weights.sum()
        
        # 基准权重
        benchmark_weights = pd.Series(
            1.0 / n_stocks,
            index=asset_cov.index
        )
        
        print(f"✓ 创建模拟数据: {n_stocks}只股票, {n_factors}个因子")
        
        # 5.2 测试风险归因
        analyzer = RiskAttributionAnalyzer()
        risk_results = analyzer.analyze_risk(
            asset_cov, factor_cov, exposure,
            portfolio_weights, benchmark_weights
        )
        
        self.assert_not_none(risk_results, "风险结果为空")
        assert 'total_risk' in risk_results, "缺少总风险指标"
        assert 'active_risk' in risk_results, "缺少主动风险指标"
        assert 'mcar' in risk_results, "缺少MCAR"
        assert 'rcar' in risk_results, "缺少RCAR"
        
        print(f"✓ 风险归因计算成功")
        print(f"  组合总风险: {risk_results['total_risk']:.6f}")
        print(f"  主动风险: {risk_results['active_risk']:.6f}")
        print(f"  MCAR数量: {len(risk_results['mcar'])}")
        print(f"  RCAR数量: {len(risk_results['rcar'])}")
        
        # 5.3 验证RCAR之和等于主动风险
        rcar_sum = risk_results['rcar'].sum()
        active_risk = risk_results['active_risk']
        diff = abs(rcar_sum - active_risk)
        assert diff < 1e-6, f"RCAR之和不等于主动风险: {rcar_sum} vs {active_risk}"
        print(f"✓ RCAR之和验证通过: {rcar_sum:.6f} ≈ {active_risk:.6f}")
        
        return {
            'n_stocks': n_stocks,
            'n_factors': n_factors,
            'total_risk': float(risk_results['total_risk']),
            'active_risk': float(risk_results['active_risk']),
            'rcar_sum': float(rcar_sum),
            'rcar_validation': diff < 1e-6,
        }
    
    # ==================== 测试6: 输出模块 ====================
    def test_output_module(self):
        """测试输出模块"""
        print("\n测试输出模块...")
        
        from barra.risk_control import RiskOutputManager
        
        # 6.1 创建模拟风险指标
        calc_date = TEST_CONFIG['calc_date']
        
        # 使用真实的CNE6因子名称（前20个因子作为示例）
        from barra.risk_control.config import STYLE_FACTOR_LIST, INDUSTRY_NAMES
        
        mcar = pd.Series(
            np.random.randn(50) * 0.01,
            index=[f'Stock_{i}' for i in range(50)]
        )
        
        rcar = pd.Series(
            np.random.randn(50) * 0.001,
            index=mcar.index
        )
        
        # 使用真实的CNE6因子：前10个风格因子 + 前10个行业
        test_factors = STYLE_FACTOR_LIST[:10] + INDUSTRY_NAMES[:10]
        
        fmcar = pd.Series(
            np.random.randn(20) * 0.01,
            index=test_factors
        )
        
        frcar = pd.Series(
            np.random.randn(20) * 0.001,
            index=test_factors
        )
        
        # 创建真实的因子类型映射
        factor_type_map = {
            'LNCAP': '规模', 'MIDCAP': '规模',
            'BETA': '波动率', 'HSIGMA': '波动率', 'DASTD': '波动率', 'CMRA': '波动率',
            'STOM': '流动性', 'STOQ': '流动性',
            'STREV': '动量', 'SEASON': '动量',
        }
        # 行业因子
        for industry in INDUSTRY_NAMES[:10]:
            factor_type_map[industry] = '行业'
        
        factor_types = pd.Series(
            [factor_type_map.get(factor, '其他') for factor in test_factors],
            index=test_factors
        )
        
        # 6.2 测试保存功能
        output_manager = RiskOutputManager(
            output_dir=TEST_CONFIG['output_dir']
        )
        
        stock_file = output_manager.save_stock_risk(mcar, rcar, calc_date)
        self.assert_not_none(stock_file, "股票风险文件路径为空")
        assert Path(stock_file).exists(), "股票风险文件未创建"
        print(f"✓ 保存股票风险指标: {stock_file}")
        
        factor_file = output_manager.save_factor_risk(
            fmcar, frcar, factor_types, calc_date
        )
        self.assert_not_none(factor_file, "因子风险文件路径为空")
        assert Path(factor_file).exists(), "因子风险文件未创建"
        print(f"✓ 保存因子风险指标: {factor_file}")
        
        # 6.3 测试读取功能
        stock_loaded = output_manager.load_stock_risk(calc_date)
        self.assert_valid_dataframe(stock_loaded, min_rows=1,
                                   required_columns=['instrument', 'mcar', 'rcar'])
        print(f"✓ 读取股票风险指标: {len(stock_loaded)}行")
        
        factor_loaded = output_manager.load_factor_risk(calc_date)
        self.assert_valid_dataframe(factor_loaded, min_rows=1,
                                   required_columns=['factor_name', 'fmcar', 'frcar'])
        print(f"✓ 读取因子风险指标: {len(factor_loaded)}行")
        
        return {
            'stock_file': stock_file,
            'factor_file': factor_file,
            'n_stock_records': len(stock_loaded),
            'n_factor_records': len(factor_loaded),
        }
    
    def run_all_tests(self):
        """运行所有模块测试"""
        self.setup()
        
        # 运行各个测试
        self.run_test("数据加载模块", self.test_data_loader)
        self.run_test("组合管理模块", self.test_portfolio_manager)
        self.run_test("因子暴露模块", self.test_factor_exposure)
        self.run_test("协方差估计模块", self.test_covariance_estimation)
        self.run_test("风险归因模块", self.test_risk_attribution)
        self.run_test("输出模块", self.test_output_module)
        
        # 保存结果
        return self.save_results()


if __name__ == '__main__':
    # 运行测试
    tester = ModuleTests()
    result_files = tester.run_all_tests()
    
    print("\n" + "="*70)
    print("模块化测试完成")
    print("="*70)
    print(f"\n结果文件:")
    for key, filepath in result_files.items():
        print(f"  {key}: {filepath}")
