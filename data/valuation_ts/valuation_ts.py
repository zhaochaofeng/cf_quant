'''
    tushare 市值数据入库 MySQL
'''

import time
import fire
import traceback
from utils import TSDataProcesssor
from utils import send_email

feas = {
    'ts_code': 'ts_code',
    'qlib_code': 'ts_code',
    'day': 'trade_date',
    'close': 'close',
    'turnover_rate': 'turnover_rate',
    'turnover_rate_f': 'turnover_rate_f',
    'volume_ratio': 'volume_ratio',
    'pe': 'pe',
    'pe_ttm': 'pe_ttm',
    'pb': 'pb',
    'ps': 'ps',
    'ps_ttm': 'ps_ttm',
    'dv_ratio': 'dv_ratio',
    'dv_ttm': 'dv_ttm',
    'total_share': 'total_share',
    'float_share': 'float_share',
    'free_share': 'free_share',
    'total_mv': 'total_mv',
    'circ_mv': 'circ_mv'
}


def main(start_date: str, end_date: str, use_trade_day: bool = True):
    try:
        t = time.time()
        process = TSDataProcesssor(start_date, end_date,
                                   api_func='daily_basic',
                                   feas=feas,
                                   table_name='valuation_ts',
                                   use_trade_day=use_trade_day,
                                   log_file='logs/{}.log'.format(end_date),
                                   now_date=None)
        # 获取股票集合
        stocks = process.get_stocks()
        # 处理数据
        data = process.process_data(stocks)
        print(data[0:2])
        # 写入数据库
        process.write_to_mysql(data)
        print('耗时： {}s'.format(round(time.time()-t, 4)))
    except Exception as e:
        error_info = traceback.format_exc()
        send_email('Data: valuation_ts', error_info)

if __name__ == '__main__':
    fire.Fire(main)
