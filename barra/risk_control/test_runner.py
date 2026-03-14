"""
Barra CNE6 风险模型 - 主测试运行脚本
运行所有测试并生成综合报告
"""
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from test_base import BarraTestSuite


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("Barra CNE6 风险模型完整测试套件")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    # ==================== 运行模块化单元测试 ====================
    print("\n" + "="*70)
    print("第一部分: 模块化单元测试")
    print("="*70)
    
    try:
        from test_modules import ModuleTests
        module_tester = ModuleTests()
        module_results = module_tester.run_all_tests()
        all_results.extend(module_tester.results)
        print(f"\n✓ 模块化测试完成，生成 {len(module_tester.results)} 个测试结果")
    except Exception as e:
        print(f"\n✗ 模块化测试失败: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    # ==================== 运行集成测试 ====================
    print("\n" + "="*70)
    print("第二部分: 集成测试")
    print("="*70)
    
    try:
        from test_integration import IntegrationTests
        integration_tester = IntegrationTests()
        integration_results = integration_tester.run_all_tests()
        all_results.extend(integration_tester.results)
        print(f"\n✓ 集成测试完成，生成 {len(integration_tester.results)} 个测试结果")
    except Exception as e:
        print(f"\n✗ 集成测试失败: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    # ==================== 生成综合报告 ====================
    print("\n" + "="*70)
    print("第三部分: 综合报告")
    print("="*70)
    
    # 创建综合测试套件（仅用于报告生成）
    suite = BarraTestSuite(output_dir='barra/risk_control/test_output')
    suite.results = all_results
    
    # 生成报告
    report = suite.generate_report()
    print(report)
    
    # 保存综合报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('barra/risk_control/test_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 文本报告
    report_file = output_dir / f"complete_test_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✓ 综合报告已保存: {report_file}")
    
    # 2. CSV摘要
    import pandas as pd
    summary_data = []
    for result in all_results:
        summary_data.append({
            'test_name': result.test_name,
            'passed': result.passed,
            'execution_time': result.execution_time,
            'error_message': result.error_message if result.error_message else '',
            'timestamp': result.timestamp,
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / f"complete_test_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')
    print(f"✓ 测试摘要已保存: {summary_file}")
    
    # 3. JSON详细报告
    import json
    details = {
        'timestamp': timestamp,
        'total_tests': len(all_results),
        'passed': sum(1 for r in all_results if r.passed),
        'failed': sum(1 for r in all_results if not r.passed),
        'total_time': sum(r.execution_time for r in all_results),
        'results': [r.to_dict() for r in all_results],
    }
    
    detail_file = output_dir / f"complete_test_details_{timestamp}.json"
    with open(detail_file, 'w', encoding='utf-8') as f:
        json.dump(details, f, indent=2, ensure_ascii=False, default=str)
    print(f"✓ 详细结果已保存: {detail_file}")
    
    # ==================== 最终统计 ====================
    print("\n" + "="*70)
    print("测试完成统计")
    print("="*70)
    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    failed = total - passed
    total_time = sum(r.execution_time for r in all_results)
    
    print(f"总测试数: {total}")
    print(f"通过: {passed} ({passed/total*100:.1f}%)")
    print(f"失败: {failed} ({failed/total*100:.1f}%)")
    print(f"总执行时间: {total_time:.2f}秒")
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'success_rate': passed/total*100 if total > 0 else 0,
        'report_file': str(report_file),
        'summary_file': str(summary_file),
        'detail_file': str(detail_file),
    }


def print_usage():
    """打印使用说明"""
    print("""
Barra CNE6 风险模型测试套件

使用方法:
    # 运行所有测试
    python barra/risk_control/test_runner.py
    
    # 单独运行模块化测试
    python barra/risk_control/test_modules.py
    
    # 单独运行集成测试
    python barra/risk_control/test_integration.py

环境要求:
    - conda activate python311-tf210
    - qlib数据已准备好 (~/.qlib/qlib_data/custom_data_hfq)
    - 内存: 8GB+

测试内容:
    1. 数据加载模块测试
    2. 组合管理模块测试
    3. 因子暴露模块测试
    4. 协方差估计模块测试
    5. 风险归因模块测试
    6. 输出模块测试
    7. 数据流集成测试
    8. 数值正确性验证
    9. 输出文件验证
    10. 内存优化测试

输出文件:
    - test_report_*.txt: 文本格式测试报告
    - test_summary_*.csv: CSV格式测试摘要
    - test_details_*.json: JSON格式详细结果
""")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Barra CNE6 风险模型测试套件')
    parser.add_argument('--help-usage', action='store_true',
                       help='显示详细使用说明')
    
    args = parser.parse_args()
    
    if args.help_usage:
        print_usage()
    else:
        # 运行所有测试
        results = run_all_tests()
        
        # 退出码：如果有失败的测试则返回1
        sys.exit(0 if results['failed'] == 0 else 1)
