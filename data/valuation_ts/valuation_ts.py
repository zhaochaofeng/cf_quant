'''
    Tushare 市值数据入库MySQL
'''
import time
import fire
import traceback
from utils import send_email
from data.process_data import TSCommonData

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

def main(
        start_date: str,
        end_date: str,
        now_date: str = None,
        use_trade_day: bool = True
        ):
    try:
        t = time.time()
        processor = TSCommonData(
            start_date,
            end_date,
            now_date,
            use_trade_day=use_trade_day,
            feas=feas,
            table_name='valuation_ts'
        )
        if not processor.is_trade_day:
            return

        stocks = processor.get_stocks(is_alive=True)
        df = processor.fetch_data_from_api(stocks, api_fun='daily_basic', batch_size=1, req_per_min=700)
        data = processor.process(df)
        processor.write_to_mysql(data)
        processor.logger.info('耗时：{}s'.format(round(time.time() - t, 4)))
    except:
        err_msg = traceback.format_exc()
        send_email('Data: valuation_ts', err_msg)

if __name__ == '__main__':
    fire.Fire(main)

