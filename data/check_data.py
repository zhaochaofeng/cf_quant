'''
    交叉检验MySQL数据质量。
    ApiToSql: 从API中获取数据 与 MySQL 中数据对比。检查数据是否更新、入库是否出错
    SqlToSql: 比较不同平台数据。如 Tushare,BaoStock
'''

import time
import pandas as pd
from typing import Optional
from utils import sql_engine, tushare_pro
from utils import get_trade_cal_inter, is_trade_day
from utils import LoggerFactory
from utils import MySQLDB


def ts_api(pro, api_func, **kwargs) -> pd.DataFrame:
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
        api_function = getattr(pro, api_func)
        # 调用API函数并传入参数
        df = api_function(**kwargs)
        return df
    except AttributeError:
        raise AttributeError(f"Tushare API中不存在函数: {api_func}")
    except Exception as e:
        raise Exception(f"调用API {api_func} 时发生错误: {str(e)}")


class CheckMySQLData:
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 table_name: str,
                 feas: list,
                 use_trade_day: bool = False,
                 log_file: Optional[str] = None,
                 level: str = "INFO",
                 ):
        """
        Args:
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'
            table_name: 数据库表名
            feas: 数据库表字段。必须包含股票代码和日期字段（ts_code,day），且放在前两个位置
            ts_api_func: Tushare API函数名。如'daily_basic'
            use_trade_day: 是否指定api中trade_date参数，若指定，则trade_date=end_date
            log_file: 日志文件名
            level: 日志级别
        """
        self.start_date = start_date
        self.end_date = end_date
        self.table_name = table_name
        self.feas = feas
        self.use_trade_day = use_trade_day
        self.logger = LoggerFactory.get_logger(__name__, log_file=log_file, level=level)
        self.is_trade_day = True
        if not is_trade_day(self.end_date):
            msg = '{} is not a trade day !!!'.format(self.end_date)
            self.logger.warning(msg)
            self.is_trade_day = False

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

    def fetch_data_from_ts(self, stocks: list, api_fun: str, batch_size: int = 1, req_per_min: int = 600):
        ''' 从Tushare获取通用数据
            Args:
                batch_size: 1次请求ts_code的个数(有些API可以请求多个ts_code)
                req_per_min: 1分钟请求的次数上界
        '''
        import warnings
        warnings.filterwarnings("ignore")
        self.logger.info('\n{}\n{}'.format('=' * 100, 'fetch_data_from_ts...'))
        try:
            pro = tushare_pro()
            if self.use_trade_day and self.start_date == self.end_date:
                trade_date = self.end_date.replace('-', '')
                df = ts_api(pro, api_fun, trade_date=trade_date)
                df = df[df['ts_code'].isin(stocks)]  # 过滤股票
            else:
                start_date = self.start_date.replace('-', '')
                end_date = self.end_date.replace('-', '')
                # 请求数据天数
                n_days = len(get_trade_cal_inter(self.start_date, self.end_date))
                if n_days == 0:
                    error_msg = 'no trade_date between {} and {}'.format(start_date, end_date)
                    self.logger.error(error_msg)
                    raise Exception(error_msg)
                batch_size = min(1000, 6000 // n_days, batch_size)  # 最多一次请求1000只股票，6000条数据
                self.logger.info('batch_size: {}, loop_n: {}'.format(
                    batch_size,
                    len(stocks) // batch_size + (1 if len(stocks) % batch_size > 0 else 0)))

                df_list = []
                for k in range(0, len(stocks), batch_size):
                    if (k + 1) % 100 == 0:
                        self.logger.info('processed : {} / {}'.format(k + batch_size, len(stocks)))
                    tmp = ts_api(pro, api_fun,
                                 ts_code=','.join(stocks[k:k + batch_size]),
                                 start_date=start_date, end_date=end_date)
                    if tmp.empty:
                        # self.logger.info('no data: {}'.format(','.join(stocks[k: k + batch_size])))
                        continue
                    df_list.append(tmp)
                    time.sleep(60 / req_per_min)
                df = pd.concat(df_list, axis=0, join='outer')

            if df.empty:
                err_msg = 'df is empty !'
                self.logger.error(err_msg)
                raise Exception(err_msg)

            df['day'] = pd.to_datetime(df['trade_date']).dt.date
            df = df[self.feas]
            self.logger.info('df shape: {}'.format(df.shape))
            df.set_index(keys=['day', 'ts_code'], inplace=True)
            return df
        except Exception as e:
            error_msg = 'error in fetch_data_from_api: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def check(self, df_mysql, df_api, is_repair=True):
        '''
        检查 MySQL 与 API 数据是否相同
        Args:
            df_mysql: mysql中的数据
            df_api: api中请求的数据
            is_repair: 当df_mysql与df_api不一致时，使用api数据修复mysql数据
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

                if is_repair:
                    # 修复mysql中的数据
                    with MySQLDB() as db:
                        api_row = df_api.loc[index]
                        params = {
                            'day': index[0],
                            'ts_code': index[1],
                        }
                        for f in self.feas[2:]:
                            v = api_row[f]
                            if pd.isna(v):
                                v = None
                            params[f] = v
                        sql = """
                            UPDATE {} SET {} WHERE day=%(day)s AND ts_code=%(ts_code)s
                        """.format(self.table_name, ','.join(['{}=%({})s'.format(f, f) for f in self.feas[2:]]))
                        db.execute(sql, params)

                res.append('{}: [mysql: {};  api: {}]'.format(index, ', '.join(mysql_f), ', '.join(api_f)))
            if len(res) == 0:
                self.logger.info('检查没有异常 ！！！')
            return res
        except Exception as e:
            error_msg = 'check error: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)


if __name__ == '__main__':
    check = CheckMySQLData(
        start_date='2025-10-23',
        end_date='2025-10-23',
        table_name='valuation_ts',
        feas=['ts_code', 'day', 'close', 'turnover_rate'],
        use_trade_day=True,
    )

    df_mysql = check.fetch_data_from_mysql()
    stocks = df_mysql.index.get_level_values('ts_code').unique().tolist()
    df_ts = check.fetch_data_from_ts(stocks, api_fun='daily_basic', batch_size=1, req_per_min=700)
    res = check.check(df_mysql, df_ts)
    print(res)
