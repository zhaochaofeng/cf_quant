'''
    检查 MySQL 数据准确性类模版
'''

import time
import pandas as pd
from typing import Optional
from .utils import sql_engine, tushare_pro
from .utils import get_trade_cal_inter, is_trade_day
from .logger import LoggerFactory


class CheckMySQLData:
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 table_name: str,
                 feas: list,
                 ts_api_func: Optional[str] = None,
                 log_file: Optional[str] = None,
                 level: str = "INFO",
                 ):
        """
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'
            table_name: 数据库表名
            feas: 数据库表字段。必须包含股票代码和日期字段（如ts_code,day），且放在前两个位置
            ts_api_func: Tushare API函数名。如'daily_basic'
            log_file: 日志文件名
            level: 日志级别
        """
        self.start_date = start_date
        self.end_date = end_date
        self.table_name = table_name
        self.feas = feas
        self.ts_api_func = ts_api_func
        self.logger = LoggerFactory.get_logger(__name__, log_file=log_file, level=level)
        self.pro = tushare_pro()
        if not is_trade_day(self.end_date):
            msg = '{} is not a trade day !!!'.format(self.end_date)
            self.logger.warning(msg)
            exit(0)

    def fetch_data_from_mysql(self):
        """
        从 MySQL 中获取数据
        """
        self.logger.info('\n{}\n{}'.format('=' * 100, 'fetch_data_from_mysql ...'))
        try:
            engine = sql_engine()
            sql = f"""
                SELECT {','.join(self.feas)} FROM {self.table_name} WHERE day>='{self.start_date}' AND day<='{self.end_date}'
            """
            self.logger.info('\n{}\n{}\n{}'.format('-' * 50, sql, '-' * 50))
            df = pd.read_sql(sql, engine)
            self.logger.info('df shape: {}'.format(df.shape))
            if df.empty:
                error_msg = 'df is empty !!!'
                self.logger.error(error_msg)
                raise Exception(error_msg)
            df.set_index(keys=['day', 'ts_code'], inplace=True)
            return df
        except Exception as e:
            error_msg = 'fetch_data_from_mysql error: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def fetch_data_from_ts(self, stocks):
        '''
        从 Tushare 中获取数据
        '''
        self.logger.info('\n{}\n{}'.format('=' * 100, 'fetch_data_from_ts ...'))
        try:
            if self.ts_api_func is None:
                error_msg = 'ts_api_func is None !!!'
                self.logger.error(error_msg)
                raise Exception(error_msg)
            start_date = self.start_date.replace('-', '')
            end_date = self.end_date.replace('-', '')

            n_days = len(get_trade_cal_inter(self.start_date, self.end_date))
            if n_days == 0:
                error_msg = 'no trade_date between {} and {}'.format(start_date, end_date)
                self.logger.error(error_msg)
                raise Exception(error_msg)
            batch_size = min(1000, 6000 // n_days)  # 最多一次请求1000只股票，6000条数据
            self.logger.info('batch_size: {}, loop_n: {}'.format(
                batch_size,
                len(stocks) // batch_size + (1 if len(stocks) % batch_size > 0 else 0)))

            df = pd.DataFrame()
            for k in range(0, len(stocks), batch_size):
                time.sleep(60 / 700)  # 一分钟最多访问700次
                api_fun = getattr(self.pro, self.ts_api_func)
                tmp = api_fun(ts_code=','.join(stocks[k: k + batch_size]),
                              start_date=start_date,
                              end_date=end_date)
                if tmp.empty or len(tmp) == 0:
                    self.logger.info('no data: {}'.format(','.join(stocks[k: k + batch_size])))
                    continue
                df = pd.concat([df, tmp], axis=0, join='outer')

            if df.empty:
                error_msg = 'df is empty !!!'
                self.logger.error(error_msg)
                raise Exception(error_msg)
            df['day'] = pd.to_datetime(df['trade_date']).dt.date
            df = df[self.feas]
            self.logger.info('df shape: {}'.format(df.shape))
            df.set_index(keys=['day', 'ts_code'], inplace=True)
            return df
        except Exception as e:
            error_msg = 'fetch_data_from_ts error: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def check(self, df_mysql, df_api):
        '''
        检查 MySQL 与 API 数据是否相同
        '''
        self.logger.info('\n{}\n{}'.format('=' * 100, 'check ...'))
        try:
            self.logger.info('df_mysql shape: {}, df_api shape: {}'.format(df_mysql.shape, df_api.shape))
            diff = (df_mysql.eq(df_api)) | ((df_mysql.isna()) & (df_api.isna()))    # 值相同 ｜ 都为NaN
            mask_ne = (diff != True).any(axis=1)
            index_ne = diff.index[mask_ne]  # 包含不相等值的行索引

            res = []  # 存放错误信息
            for index in index_ne:
                mysql_f = []
                api_f = []
                try:
                    mysql_row = df_mysql.loc[index]
                    for f in self.feas[2:]:
                        mysql_f.append('{}:{}'.format(f, mysql_row[f]))
                except:
                    mysql_f = ['NaN']

                try:
                    api_row = df_api.loc[index]
                    for f in self.feas[2:]:
                        api_f.append('{}:{}'.format(f, api_row[f]))
                except:
                    api_f = ['NaN']

                res.append('{}: [mysql: {};  api: {}]'.format(index, ', '.join(mysql_f), ', '.join(api_f)))
            if len(res) == 0:
                self.logger.info('检查没有异常 ！！！')
            return res
        except Exception as e:
            error_msg = 'check error: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)


if __name__ == '__main__':
    from utils import CheckMySQLData

    check = CheckMySQLData(
        table_name='valuation_ts',
        feas=['ts_code', 'day', 'close', 'turnover_rate'],
        start_date='2025-10-23',
        end_date='2025-10-23',
        ts_api_func='daily_basic',
        log_file='log/test.log',
    )

    df_mysql = check.fetch_data_from_mysql()
    stocks = df_mysql.index.get_level_values('ts_code').unique().tolist()

    df_ts = check.fetch_data_from_ts(stocks)
    res = check.check(df_mysql, df_ts)
    print(res)
