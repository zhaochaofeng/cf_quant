import time
import fire
import traceback
import pandas as pd
from datetime import datetime
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.database import get_database
from vnpy.trader.datafeed import get_datafeed
from vnpy.trader.object import HistoryRequest

from utils.utils import tushare_pro
from utils.utils import send_email
from utils.utils import is_trade_day

class UpdateStockData:
    def __init__(self, start_date, end_date):
        if not is_trade_day(end_date):
            print('{} 不是交易日！！！'.format(end_date))
            exit(0)
        self.start_date = start_date
        self.end_date = end_date
        self.pro = tushare_pro()
        self.database = get_database()

    def get_stocks(self):
        print('-' * 100)
        print('get_stocks ...')
        # 上市股票
        df_l = self.pro.stock_basic(list_status='L')
        # 退市股票
        df_d = self.pro.stock_basic(list_status='D')
        df = pd.concat([df_l, df_d], ignore_index=True)
        print('股票数：{}'.format(len(df)))
        return df['ts_code'].values.tolist()

    def get_tushare_data(self, code):
        datafeed = get_datafeed()
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        symbol, exchange = code.split('.')
        exchange = Exchange.SZSE if exchange == 'SZ' else Exchange.SSE

        # 代码逻辑是左右比区间[start,end]，实际请求为左开右闭(start,end]  ???
        req = HistoryRequest(
            symbol=symbol,
            exchange=exchange,
            start=start,
            end=end,
            interval=Interval.DAILY
        )

        # 获取k线历史数据
        data = datafeed.query_bar_history(req)
        return data

    def download_to_mysql(self, data):
        print('-' * 100)
        print('download_to_mysql ...')
        # 存入mysql
        for i, d in enumerate(data):
            if (i+1) % 100 == 0:
                print('{} / {}'.format(i+1, len(data)))
            res = self.database.save_bar_data(d)

    def delete_mysql_data(self, code):
        print('-' * 100)
        print('delete_mysql_data: '.format(code))
        # 删除数据库中k线数据
        symbol, exchange = code.split('.')
        exchange = Exchange.SZSE if exchange == 'SZ' else Exchange.SSE

        self.database.delete_bar_data(
            symbol=symbol,
            exchange=exchange,
            interval=Interval.DAILY
        )

    def main(self):
        # 股票集合
        codes = self.get_stocks()
        if len(codes) == 0:
            raise Exception('没有股票数据！！！')
        # 获取交易数据
        data = []
        # codes = codes[0:10]
        print('-' * 100)
        print('get_tushare_data ...')
        for i, code in enumerate(codes):
            time.sleep(0.075)
            if (i+1) % 100 == 0:
                print('{} / {}'.format(i+1, len(codes)))
            tmp = self.get_tushare_data(code)
            if not tmp:
                continue
            data.append(tmp)
        print('data len: {}'.format(len(data)))
        if len(data) == 0:
            raise Exception('没有数据！！！')
        # 存入mysql
        self.download_to_mysql(data)

if __name__ == '__main__':
    t = time.time()
    try:
        fire.Fire(UpdateStockData)
    except Exception as e:
        err = traceback.format_exc()
        send_email("Data:vnpy_online:process", err)
    print('耗时：{} s'.format(round(time.time() - t, 4)))
