"""
投资组合优化每日运行入口
"""
import argparse
import sys
from pathlib import Path

from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='主动投资组合优化'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='计算日期 (YYYY-MM-DD)，默认为最新交易日'
    )
    
    parser.add_argument(
        '--position',
        type=str,
        default='zero',
        help='当前持仓输入: zero/dict/CSV路径'
    )
    
    parser.add_argument(
        '--value',
        type=float,
        default=1e8,
        help='组合净值（元），默认1亿'
    )
    
    parser.add_argument(
        '--risk_aversion',
        type=float,
        default=0.05,
        help='风险厌恶系数，默认0.05'
    )
    
    parser.add_argument(
        '--max_turnover',
        type=float,
        default=0.10,
        help='换手率上限，默认0.10'
    )
    
    parser.add_argument(
        '--use_qp',
        action='store_true',
        help='是否使用QP优化作为初始解'
    )
    
    parser.add_argument(
        '--save_mysql',
        action='store_true',
        help='是否保存到MySQL'
    )
    
    parser.add_argument(
        '--portfolio',
        type=str,
        default='default',
        help='组合名称'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录'
    )
    
    return parser.parse_args()


def get_latest_trade_date() -> str:
    """获取最新交易日"""
    from qlib.data import D
    import pandas as pd
    
    # 获取交易日历
    cal = D.calendar(freq='day')
    latest = pd.Timestamp(cal[-1])
    return latest.strftime('%Y-%m-%d')


def init_qlib():
    """初始化Qlib"""
    import qlib
    from utils import get_config
    
    config = get_config()
    provider_uri = config.get('qlib', {}).get('provider_uri', '~/.qlib/qlib_data/custom_data_hfq')
    
    try:
        qlib.init(provider_uri=provider_uri)
        logger.info(f'Qlib初始化成功: provider_uri={provider_uri}')
    except Exception as e:
        logger.error(f'Qlib初始化失败: {e}')
        raise


def main():
    """主函数"""
    args = parse_args()
    
    # 确定计算日期
    if args.date:
        calc_date = args.date
    else:
        init_qlib()
        calc_date = get_latest_trade_date()
    
    logger.info(f'计算日期: {calc_date}')
    logger.info(f'组合净值: {args.value:,.0f}元')
    logger.info(f'风险厌恶系数: {args.risk_aversion}')
    logger.info(f'换手率上限: {args.max_turnover}')
    
    try:
        # 初始化Qlib
        init_qlib()
        
        # 导入并创建引擎
        from portfolio_engine import PortfolioEngine
        
        engine = PortfolioEngine(
            calc_date=calc_date,
            output_dir=args.output_dir,
            risk_aversion=args.risk_aversion,
            max_turnover=args.max_turnover
        )
        
        # 执行优化
        result = engine.run(
            position_input=args.position,
            portfolio_value=args.value,
            use_qp_init=args.use_qp,
            save_to_mysql=args.save_mysql,
            portfolio_name=args.portfolio
        )
        
        logger.info('投资组合优化完成')
        return 0
        
    except FileNotFoundError as e:
        logger.error(f'文件不存在: {e}')
        return 1
    except ValueError as e:
        logger.error(f'参数错误: {e}')
        return 1
    except Exception as e:
        logger.error(f'运行失败: {e}', exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
