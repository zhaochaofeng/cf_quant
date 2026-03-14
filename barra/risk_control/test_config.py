"""
Barra CNE6 风险模型 - 测试配置文件
配置qlib数据和测试参数
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Qlib配置
QLIB_CONFIG = {
    'provider_uri': '~/.qlib/qlib_data/custom_data_hfq',
    'region': 'cn',
}

# 测试数据配置
TEST_CONFIG = {
    # 测试日期范围（使用最近的数据）
    'test_start_date': '2024-01-01',
    'test_end_date': '2024-03-01',
    'calc_date': '2024-03-01',
    
    # 测试股票数量（小规模测试）
    'test_n_stocks': 50,
    
    # 历史数据月数（用于协方差估计）
    'history_months': 12,  # 测试时使用1年，生产用10年
    
    # 输出目录
    'output_dir': 'barra/risk_control/test_output',
    'cache_dir': 'barra/risk_control/test_cache',
    
    # 内存配置
    'memory_threshold_gb': 6.0,
    'stock_batch_size': 50,  # 测试时使用小批次
    'date_batch_size': 5,
    
    # 并行配置
    'n_jobs': 2,  # 测试时使用2个进程
    
    # 数值精度（用于结果验证）
    'tolerance': 1e-6,
}

# 预期结果范围（用于验证）
EXPECTED_RANGES = {
    # 因子收益率应该在合理范围内
    'factor_return_mean': (-0.1, 0.1),
    'factor_return_std': (0.0, 0.5),
    
    # 风险指标
    'active_risk': (0.0, 1.0),  # 跟踪误差
    'total_risk': (0.0, 1.0),   # 组合总风险
    
    # MCAR/RCAR
    'mcar_range': (-1.0, 1.0),
    'rcar_range': (-0.1, 0.1),
    
    # FMCAR/FRCAR
    'fmcar_range': (-1.0, 1.0),
    'frcar_range': (-0.1, 0.1),
}

# 测试报告配置
REPORT_CONFIG = {
    'report_file': 'test_report.txt',
    'summary_file': 'test_summary.csv',
    'detail_file': 'test_details.json',
}
