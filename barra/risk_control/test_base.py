"""
Barra CNE6 风险模型 - 测试基类和工具
"""
import sys
from pathlib import Path
from datetime import datetime
import time
import traceback

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import qlib
from typing import Dict, List, Callable, Any


class TestResult:
    """测试结果类"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = False
        self.execution_time = 0.0
        self.error_message = None
        self.details = {}
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'execution_time': self.execution_time,
            'error_message': self.error_message,
            'details': self.details,
            'timestamp': self.timestamp,
        }


class BarraTestSuite:
    """Barra测试套件基类"""
    
    def __init__(self, output_dir: str = 'barra/risk_control/test_output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.start_time = None
        
    def init_qlib(self, provider_uri: str = '~/.qlib/qlib_data/custom_data_hfq'):
        """初始化qlib，注册PTTM自定义操作符"""
        try:
            # 导入PTTM操作符
            from utils.qlib_ops import PTTM
            # 初始化qlib并注册自定义操作符
            qlib.init(
                provider_uri=provider_uri,
                custom_ops=[PTTM]
            )
            print(f"✓ Qlib初始化成功（已注册PTTM操作符）")
            return True
        except Exception as e:
            print(f"✗ Qlib初始化失败: {str(e)}")
            return False
    
    def run_test(self, test_name: str, test_func: Callable, 
                 *args, **kwargs) -> TestResult:
        """
        运行单个测试
        
        Args:
            test_name: 测试名称
            test_func: 测试函数
            *args, **kwargs: 测试函数参数
            
        Returns:
            TestResult对象
        """
        result = TestResult(test_name)
        print(f"\n{'='*60}")
        print(f"运行测试: {test_name}")
        print(f"{'='*60}")
        
        start = time.time()
        try:
            # 执行测试
            test_result = test_func(*args, **kwargs)
            
            # 如果测试函数返回字典，保存详细信息
            if isinstance(test_result, dict):
                result.details = test_result
            
            result.passed = True
            print(f"✓ 测试通过")
            
        except Exception as e:
            result.passed = False
            result.error_message = str(e)
            result.details['traceback'] = traceback.format_exc()
            print(f"✗ 测试失败: {str(e)}")
            print(traceback.format_exc())
        
        result.execution_time = time.time() - start
        print(f"执行时间: {result.execution_time:.2f}秒")
        
        self.results.append(result)
        return result
    
    def assert_almost_equal(self, actual, expected, tolerance=1e-6, 
                           message=""):
        """断言两个值近似相等"""
        if isinstance(actual, pd.Series):
            actual = actual.values
        if isinstance(expected, pd.Series):
            expected = expected.values
            
        diff = np.abs(np.array(actual) - np.array(expected))
        max_diff = np.max(diff)
        
        if max_diff > tolerance:
            raise AssertionError(
                f"{message}\n实际值与期望值差异过大: {max_diff:.2e} > {tolerance}\n"
                f"实际值: {actual}\n期望值: {expected}"
            )
    
    def assert_in_range(self, value, min_val, max_val, message=""):
        """断言值在范围内"""
        if isinstance(value, pd.Series):
            actual_min = value.min()
            actual_max = value.max()
        else:
            actual_min = actual_max = value
        
        if actual_min < min_val or actual_max > max_val:
            raise AssertionError(
                f"{message}\n值超出范围 [{min_val}, {max_val}]\n"
                f"实际范围: [{actual_min}, {actual_max}]"
            )
    
    def assert_not_none(self, value, message=""):
        """断言值不为None"""
        if value is None:
            raise AssertionError(f"{message}\n值为None")
    
    def assert_valid_dataframe(self, df: pd.DataFrame, 
                              min_rows: int = 1,
                              required_columns: List[str] = None,
                              message=""):
        """断言DataFrame有效"""
        if df is None or df.empty:
            raise AssertionError(f"{message}\nDataFrame为空")
        
        if len(df) < min_rows:
            raise AssertionError(
                f"{message}\n行数不足: {len(df)} < {min_rows}"
            )
        
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise AssertionError(
                    f"{message}\n缺少列: {missing}"
                )
    
    def generate_report(self) -> str:
        """生成测试报告"""
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("Barra CNE6 风险模型测试报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*70)
        
        # 统计信息
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        total_time = sum(r.execution_time for r in self.results)
        
        report_lines.append(f"\n总测试数: {total}")
        report_lines.append(f"通过: {passed} ({passed/total*100:.1f}%)")
        report_lines.append(f"失败: {failed} ({failed/total*100:.1f}%)")
        report_lines.append(f"总执行时间: {total_time:.2f}秒")
        
        # 详细结果
        report_lines.append("\n" + "="*70)
        report_lines.append("详细测试结果")
        report_lines.append("="*70)
        
        for i, result in enumerate(self.results, 1):
            status = "✓ 通过" if result.passed else "✗ 失败"
            report_lines.append(f"\n{i}. {result.test_name}")
            report_lines.append(f"   状态: {status}")
            report_lines.append(f"   时间: {result.execution_time:.2f}秒")
            report_lines.append(f"   时间戳: {result.timestamp}")
            
            if result.error_message:
                report_lines.append(f"   错误: {result.error_message}")
            
            if result.details:
                report_lines.append(f"   详细信息:")
                for key, value in result.details.items():
                    if key != 'traceback':
                        report_lines.append(f"     {key}: {value}")
        
        report_lines.append("\n" + "="*70)
        report_lines.append("测试结束")
        report_lines.append("="*70)
        
        return "\n".join(report_lines)
    
    def save_results(self):
        """保存测试结果到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 保存文本报告
        report_file = self.output_dir / f"test_report_{timestamp}.txt"
        report = self.generate_report()
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n测试报告已保存: {report_file}")
        
        # 2. 保存CSV摘要
        summary_data = []
        for result in self.results:
            summary_data.append({
                'test_name': result.test_name,
                'passed': result.passed,
                'execution_time': result.execution_time,
                'error_message': result.error_message if result.error_message else '',
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / f"test_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        print(f"测试摘要已保存: {summary_file}")
        
        # 3. 保存详细JSON（可选）
        import json
        details = {
            'timestamp': timestamp,
            'total_tests': len(self.results),
            'passed': sum(1 for r in self.results if r.passed),
            'failed': sum(1 for r in self.results if not r.passed),
            'results': [r.to_dict() for r in self.results],
        }
        
        detail_file = self.output_dir / f"test_details_{timestamp}.json"
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(details, f, indent=2, ensure_ascii=False, default=str)
        print(f"详细结果已保存: {detail_file}")
        
        return {
            'report_file': str(report_file),
            'summary_file': str(summary_file),
            'detail_file': str(detail_file),
        }
