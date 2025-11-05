'''
    股票基本上信息表：stock_info_bao
'''
import time
import traceback
from datetime import datetime

import fire
import pandas as pd
from data.process_data import Base
from utils import bao_stock_connect, send_email, is_trade_day, bao_api

feas = {
    'code': 'code',
    'qlib_code': 'code',
    'day': '',
    'name': 'code_name',
    'list_date': 'ipoDate',
    'exchange': 'code',
    'out_date': 'outDate',
    'status': 'status'
}

exchange_map = {'BJ': 'BSE', 'SH': 'SSE', 'SZ': 'SZSE'}
# sh.600849在2013年底退市，为了与tushare保持数据一致，做过滤处理
exclude_codes = {'sh.600849', 'sz.000022', 'sz.000043', 'sz.300114'}


class BaoStockInfoProcessor(Base):
    ''' 股票信息数据 '''

    def __init__(self,
                 feas: dict = feas,
                 table_name: str = 'stock_info_bao',
                 now_date: str = None,
                 **kwargs
                 ):
        super().__init__(feas=feas, table_name=table_name, **kwargs)
        self.now_date = now_date if now_date else datetime.now().strftime('%Y-%m-%d')
        if not is_trade_day(self.now_date):
            self.logger.info('非交易日，不处理')
            exit(0)

    def fetch_data_from_api(self):
        ''' 股票基本信息 + 行业数据 '''
        self.logger.info('\n{}\n{}'.format('=' * 100, 'fetch_data_from_api ...'))
        try:
            bs = bao_stock_connect()
            df = bao_api(bs, 'query_stock_basic')
            if df.empty:
                err_msg = 'df is empty !'
                self.logger.error(err_msg)
                raise Exception(err_msg)
            df = df[df['type'] == '1'][['code', 'code_name', 'ipoDate', 'outDate', 'status']]
            # 排除噪声数据
            df = df[~df['code'].isin(exclude_codes)]
            self.logger.info('df shape: {}'.format(df.shape))
            bs.logout()
            return df
        except Exception as e:
            error_msg = 'error in fetch_data_from_api: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def parse_line(self, row) -> dict:
        ''' 解析单条数据 '''
        try:
            tmp = {}
            for f in self.feas.keys():
                if f == 'day' and self.feas[f] == '':
                    v = self.now_date
                else:
                    v = row[self.feas[f]]
                    if pd.isna(v) or v == '':
                        v = None
                    elif f == 'qlib_code':
                        suffix, code = v.split('.')
                        v = '{}{}'.format(suffix.upper(), code)
                    elif f == 'exchange':
                        suffix, code = v.split('.')
                        v = exchange_map[suffix.upper()]
                tmp[f] = v
            return tmp
        except Exception as e:
            error_msg = 'parse_line error: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def main(self) -> None:
        t = time.time()
        try:
            df = self.fetch_data_from_api()
            data = self.process(df)
            self.write_to_mysql(data)
            print('耗时：{}s'.format(round(time.time() - t, 4)))
        except:
            error_msg = traceback.format_exc()
            self.logger.error(error_msg)
            send_email('Data: stock_info_bao', error_msg)
            raise Exception(error_msg)


if __name__ == '__main__':
    fire.Fire(BaoStockInfoProcessor)
    '''
        python stock_info_bao.py.bak --now_date 2025-11-02 main
    '''
