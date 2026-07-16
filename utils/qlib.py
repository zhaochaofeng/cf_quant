import pandas as pd

import qlib
from qlib.data import D
from config import PROVIDER_URI
from .logger import LoggerFactory
from .qlib_ops import PTTM
from .utils import sql_engine
from datetime import datetime

logger = LoggerFactory.get_logger(__name__)


def init_qlib(provider_uri=PROVIDER_URI, custom_ops=None):
    """初始化qlib """
    if custom_ops is None:
        custom_ops = [PTTM]
    qlib.init(
        provider_uri=provider_uri,
        custom_ops=custom_ops,
    )

'''
def get_instruments(market='all', start_time=None, end_time=None, filter_pipe=None) -> list:
    """ 获取股票列表 """

    if start_time is None or end_time is None:
        start_time = end_time = datetime.now().strftime('%Y-%m-%d')
        logger.warning(f'start_time 和 end_time 为空，使用默认值: {start_time}')
    config = D.instruments(market=market, filter_pipe=filter_pipe)
    instruments = D.list_instruments(
        config, start_time=start_time, end_time=end_time, as_list=True
    )

def get_universe(date: str, exclude_suspended: bool = True, exclude_ipo_lt_1y: bool = True) -> list:
    """获取 A 股股票集合，可选排除停牌和次新股

    Args:
        date: str, 交易日 'YYYY-MM-DD'
        exclude_suspended: bool, 是否排除停牌股票，默认 True
        exclude_ipo_lt_1y: bool, 是否排除上市不足一年的次新股，默认 True

    Returns:
        list: 股票代码列表
    """
    from qlib.data import D  # 延迟导入，确保 qlib 已初始化

    # ========== 输入验证 ==========
    if not isinstance(date, str) or not date:
        raise ValueError(f'date 必须为非空字符串，当前值: {date}')
    try:
        pd.Timestamp(date)
    except Exception:
        raise ValueError(f'date 格式错误，需为 YYYY-MM-DD 格式，当前值: {date}')

    # ========== Step 1: 获取当日交易股票 ==========
    instruments = D.instruments(market='all')
    stocks = D.list_instruments(
        instruments, start_time=date, end_time=date, as_list=True
    )

    if not stocks:
        logger.warning(f'get_universe: {date} 无交易股票')
        return []

    logger.info(f'get_universe {date}: Step1 当日交易股票 {len(stocks)} 只')

    # ========== Step 2: 排除次新 ==========
    if exclude_ipo_lt_1y:
        try:
            engine = sql_engine()
            codes_str = ','.join([f"'{c}'" for c in stocks])
            sql = f"""
                SELECT DISTINCT qlib_code, list_date
                FROM stock_info_ts
                WHERE qlib_code IN ({codes_str})
            """
            info_df = pd.read_sql(sql, engine)
            engine.dispose()

            if info_df.empty:
                logger.warning(f'get_universe: stock_info_ts 查询为空，date={date}')
                return stocks

            # 按 qlib_code 分组取第一条（去重）
            info_df = info_df.drop_duplicates(subset='qlib_code')
            list_date_map = info_df.set_index('qlib_code')['list_date'].dropna().to_dict()

            cutoff_date = pd.Timestamp(date) - pd.DateOffset(years=1)
            stocks = [s for s in stocks
                      if s not in list_date_map or pd.Timestamp(list_date_map[s]) <= cutoff_date]

            logger.info(f'get_universe {date}: Step2 排除次新后 {len(stocks)} 只')
        except Exception as e:
            logger.error(f'get_universe: 排除次新失败: {e}')
            raise RuntimeError(f'排除次新失败: {e}')

    # ========== Step 3: 排除停牌 ==========
    if exclude_suspended:
        try:
            # 停牌股票当日无交易，也无收益率数据
            ret_df = D.features(
                stocks, ['$change'],
                start_time=date, end_time=date,
            )
            ret_df = ret_df.reset_index()

            # $change 非 NaN = 正常交易
            trading_stocks = ret_df[
                ret_df['$change'].notna()
            ]['instrument'].unique().tolist()

            stocks = trading_stocks
            logger.info(f'get_universe {date}: Step3 排除停牌后 {len(stocks)} 只')
        except Exception as e:
            logger.error(f'get_universe: 排除停牌失败: {e}')
            raise RuntimeError(f'排除停牌失败: {e}')

    return stocks
'''