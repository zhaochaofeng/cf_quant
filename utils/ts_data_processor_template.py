'''
    从 tushare 导入数据到 MySQL 模版类
'''

import pandas as pd
from typing import Optional
from datetime import datetime
from .utils import tushare_pro, sql_engine, get_trade_cal_inter, is_trade_day
from .logger import LoggerFactory
from .db_mysql import MySQLDB

class TSDataProcesssor:

    def __init__(self,
                 start_date: str,
                 end_date: str,
                 api_func: str,
                 feas: dict,
                 table_name: str,
                 now_date: Optional[str] = None,
                 use_trade_day: bool = False,
                 log_file: Optional[str] = None,
                 level: str = "INFO",
                 ) -> None:
        """
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'
            api_func: Tushare API函数名。如'daily_basic'
            feas: 字段映射字典，key为目标字段名，value为源字段名
            table_name: 目标数据库表名。如valuation_ts
            now_date: 当前日期，用于获取股票集合
            use_trade_day: 是否为指定trade_date以获取所有股票数据
            log_file: 日志文件路径，默认为None
            level: 日志等级，默认为"INFO"
        """
        self.start_date = start_date
        self.end_date = end_date
        self.api_func = api_func
        self.feas = feas
        self.table_name = table_name
        self.now_date = now_date
        self.use_trade_day = use_trade_day
        self.pro = tushare_pro()

        if self.now_date is None:
            self.now_date = datetime.now().strftime('%Y-%m-%d')
        self.logger = LoggerFactory.get_logger(__name__, log_file=log_file, level=level)
        if not is_trade_day(self.end_date):
            msg = '{} is not a trade date, exit !!!'.format(self.end_date)
            self.logger.warning(msg)
            exit(0)

    def get_stocks(self) -> list:
        ''' 获取股票集合 '''

        self.logger.info('{}\n{}'.format('-' * 100, 'get_stocks...'))
        engine = sql_engine()
        sql = '''
            select ts_code from stock_info_ts where day='{}';
        '''.format(self.now_date)
        self.logger.info(sql)
        df = pd.read_sql(sql, engine)
        if df.empty or len(df) == 0:
            error_msg = 'table stock_info_ts has no data: {}'.format(self.now_date)
            self.logger.error(error_msg)
            raise Exception(error_msg)
        codes = df['ts_code'].values.tolist()
        self.logger.info('stocks len: {}'.format(len(codes)))
        return codes

    def fetch_data_from_ts(self, **kwargs) -> pd.DataFrame:
        """
        通过Tushare API函数名和参数获取数据

        Args:
            api_func: tushare API函数名
            **kwargs: 传递给API函数的参数

        Returns:
            pd.DataFrame: 获取的数据

        Raises:
            AttributeError: 当指定的API函数不存在时
            Exception: API调用过程中的其他异常
        """
        try:
            # 获取API函数
            api_function = getattr(self.pro, self.api_func)
            # 调用API函数并传入参数
            df = api_function(**kwargs)
            return df
        except AttributeError:
            raise AttributeError(f"Tushare API中不存在函数: {self.api_func}")
        except Exception as e:
            raise Exception(f"调用API {self.api_func} 时发生错误: {str(e)}")

    def parse_line(self, row) -> dict:
        ''' 解析单条数据 '''
        tmp = {}
        for f in self.feas.keys():
            try:
                v = row[self.feas[f]]
                if f == 'qlib_code':
                    code, suffix = v.split('.')
                    v = '{}{}'.format(suffix.upper(), code)
                elif f == 'day':
                    # 日期格式转换
                    v = datetime.strptime(v, '%Y%m%d').strftime('%Y-%m-%d')
                if pd.isna(v):
                    v = None
                tmp[f] = v
            except Exception as e:
                error_msg = 'parse_line error: {}'.format(e)
                self.logger.error(error_msg)
                raise Exception(error_msg)
        return tmp

    def process_data(self, stocks) -> list:
        """
        从Tushare中获取输出，处理成固定格式
        """
        self.logger.info('{}\n{}'.format('-' * 100, 'process_data...'))
        try:
            if self.use_trade_day and self.start_date == self.end_date:
                trade_date = self.end_date.replace('-', '')
                df = self.fetch_data_from_ts(trade_date=trade_date)
                df = df[df['ts_code'].isin(stocks)]
            else:
                start_date = self.start_date.replace('-', '')
                end_date = self.end_date.replace('-', '')

                df = pd.DataFrame()
                # 请求数据天数
                n_days = len(get_trade_cal_inter(self.start_date, self.end_date))
                if n_days == 0:
                    error_msg = 'no trade_date between {} and {}'.format(start_date, end_date)
                    self.logger.error(error_msg)
                    raise Exception(error_msg)
                batch_size = min(1000, 6000//n_days)    # 最多一次请求1000只股票，6000条数据

                for k in range(0, len(stocks), batch_size):
                    tmp = self.fetch_data_from_ts(
                        ts_code=','.join(stocks[k: k+batch_size]),
                        start_date=start_date,
                        end_date=end_date)
                    if tmp.empty or len(tmp) == 0:
                        self.logger.info('no data: {}'.format(','.join(stocks[k: k+batch_size])))
                        continue
                    df = pd.concat([df, tmp], axis=0, join='outer')

            self.logger.info('df shape: {}'.format(df.shape))
            if df.empty:
                error_msg = 'df is empty !!!'
                self.logger.error(error_msg)
                raise Exception(error_msg)

            data = []
            for index, row in df.iterrows():
                line = self.parse_line(row)
                data.append(line)
            self.logger.info('data len: {}'.format(len(data)))

            if len(data) == 0:
                error_msg = 'no data !!!'
                self.logger.error(error_msg)
                raise Exception(error_msg)
        except Exception as e:
            error_msg = 'process_data error: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)
        return data

    def write_to_mysql(self, data):
        """ 数据写入 MySQL """
        self.logger.info('{}\n{}'.format('-' * 100, 'write_to_mysql...'))
        try:
            feas = list(self.feas.keys())
            feas_format = ['%({})s'.format(f) for f in feas]
            sql = """
                INSERT INTO {} ({}) VALUES({})
            """.format(self.table_name, ','.join(feas), ','.join(feas_format))
            with MySQLDB() as db:
                db.executemany(sql, data)
        except Exception as e:
            error_msg = 'write_to_mysql error: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)

if __name__ == '__main__':
    process = TSDataProcesssor('2025-10-23', '2025-10-24', 'daily_basic', feas={
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
    },
                               table_name='valuation_tushare',
                               use_trade_day=False,
                               log_file='log/test.log', now_date='2025-10-24')
    stocks = process.get_stocks()
    data = process.process_data(stocks[0:1])
    print(data)
    process.write_to_mysql(data)





