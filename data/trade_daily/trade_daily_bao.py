'''
    功能：BaoStock表导入mysql
    描述：日级交易数据（非复权）+ 复权因子
    刷数：设置[start_date, end_date] (单code请求)
'''

import time
import fire
import traceback
from utils import send_email
from data.process_data import BaoTradeDailyData

feas = {
    'code': 'code',
    'qlib_code': 'code',
    'day': 'date',
    'open': 'open',
    'close': 'close',
    'high': 'high',
    'low': 'low',
    'pre_close': 'preclose',
    'pct_chg': 'pctChg',
    'vol': 'volume',
    'amount': 'amount',
    'is_st': 'isST',
    'adj_factor': 'adj_factor'
}

round_dic = {
    'open': 2,
    'close': 2,
    'high': 2,
    'low': 2,
    'pre_close': 2,
    'pct_chg': 2,
    'vol': 2,
    'amount': 3,
    'is_st': 0,
    'adj_factor': 4
}

def main(
        start_date: str,
        end_date: str,
        now_date: str = None,
        ):
    try:
        t = time.time()
        processor = BaoTradeDailyData(
            start_date,
            end_date,
            now_date,
            feas=feas,
            table_name='trade_daily_bao'
        )
        if not processor.is_trade_day:
            return

        stocks = processor.get_stocks(table_name='stock_info_bao', code_name='code')
        df = processor.fetch_data_from_api(stocks[0:10], api_fun='query_history_k_data_plus', round_dic=round_dic)
        data = processor.process(df)
        processor.write_to_mysql(data)
        processor.logger.info('耗时：{}s'.format(round(time.time() - t, 4)))
    except:
        err_msg = traceback.format_exc()
        send_email('Data: trade_daily_bao', err_msg)

if __name__ == '__main__':
    fire.Fire(main)