'''
    Tushare 日级交易数据（非复权）+ 复权因子 入库MySQL
'''

import time
import fire
import traceback
from utils import send_email
from data.process_data import TSTradeDailyData

feas = {
     'ts_code': 'ts_code',
     'qlib_code': 'ts_code',
     'day': 'trade_date',
     'open': 'open',
     'high': 'high',
     'low': 'low',
     'close': 'close',
     'pre_close': 'pre_close',
     'change': 'change',
     'pct_chg': 'pct_chg',
     'vol': 'vol',
     'amount': 'amount',
     'adj_factor': 'adj_factor'
}

def main(
        start_date: str,
        end_date: str,
        now_date: str = None,
        use_trade_day: bool = True,
        is_alive: bool = True
        ):
    try:
        t = time.time()
        processor = TSTradeDailyData(
            start_date,
            end_date,
            now_date,
            use_trade_day=use_trade_day,
            feas=feas,
            table_name='trade_daily_ts'
        )
        if not processor.is_trade_day:
            return

        stocks = processor.get_stocks(is_alive=is_alive)
        df = processor.fetch_data_from_api(stocks, api_fun='daily', batch_size=1000, req_per_min=600)
        data = processor.process(df)
        processor.write_to_mysql(data)
        processor.logger.info('耗时：{}s'.format(round(time.time() - t, 4)))
    except:
        err_msg = traceback.format_exc()
        send_email('Data: trade_daily_ts', err_msg)

if __name__ == '__main__':
    fire.Fire(main)