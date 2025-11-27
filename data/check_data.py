'''
    交叉检验MySQL数据质量。
    ApiToSql: 从API中获取数据 与 MySQL 中数据对比。检查数据是否更新、入库是否出错
    SqlToSql: 比较不同平台数据。如 Tushare,BaoStock
'''

import time
import pandas as pd
from typing import Optional
import qlib
from qlib.data import D
from utils import (
    sql_engine, tushare_pro, tushare_ts,
    get_trade_cal_inter, is_trade_day, LoggerFactory,
    MySQLDB, ts_api
)


class CheckMySQLData:
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 table_name: str = None,
                 feas: list = None,
                 use_trade_day: bool = False,
                 log_file: Optional[str] = None,
                 level: str = "INFO",
                 ):
        """
        Args:
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'
            table_name: 数据库表名
            feas: 数据库表字段。必须包含股票日期和代码字段（如day, ts_code），且按照顺序放在前两个位置
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

    def fetch_data_from_mysql(self, table_name: str = None, conditions_dict: dict = None,
                              sql_str: str = None) -> pd.DataFrame:
        """
        从 MySQL 中获取数据
        Args:
            conditions_dict: 在基础sql上加条件
            sql_str: 外部直接输入SQL
        """
        if table_name is None:
            table_name = self.table_name
        self.logger.info('\n{}\n{}'.format('=' * 100, 'fetch_data_from_mysql ...'))
        try:
            engine = sql_engine()
            if sql_str:
                sql = sql_str
                self.logger.info('\n{}\n{}\n{}'.format('-' * 50, sql, '-' * 50))
                df = pd.read_sql(sql, engine)
            else:
                sql = f"""
                    SELECT {','.join(self.feas)} FROM {table_name} WHERE day>='{self.start_date}' AND day<='{self.end_date}'
                """

                if conditions_dict:
                    conditions = []
                    params = {}
                    for key, value in conditions_dict.items():
                        if ' ' in key:
                            # 处理带操作符的条件，属性和操作之间必须带空格，如 'list_date <='
                            # 构造成 'list_date <= %(list_date)s'
                            field, operator = key.split(' ', 1)
                            conditions.append(f"{field} {operator} %({field})s")
                            params[field] = value
                        else:
                            # 默认等值条件
                            conditions.append(f"{key} = %({key})s")
                            params[key] = value

                    sql = sql + " AND " + " AND ".join(conditions)
                    self.logger.info('\n{}\n{}\n{}'.format('-' * 50, sql, '-' * 50))
                    df = pd.read_sql(sql, engine, params=params)
                else:
                    self.logger.info('\n{}\n{}\n{}'.format('-' * 50, sql, '-' * 50))
                    df = pd.read_sql(sql, engine)

            self.logger.info('df shape: {}'.format(df.shape))
            if df.empty:
                error_msg = 'df is empty !!!'
                self.logger.error(error_msg)
                raise Exception(error_msg)
            df.set_index(keys=self.feas[0:2], inplace=True)
            return df
        except Exception as e:
            error_msg = 'fetch_data_from_mysql error: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def fetch_data_from_ts(self,
                           stocks: list,
                           api_fun: str,
                           batch_size: int = 1,
                           req_per_min: int = 600,
                           ts_type: str = None,
                           code_type: str = None,
                           feas: list = None,
                           **kwargs
                           ):
        ''' 从Tushare获取通用数据
            Args:
                batch_size: 1次请求ts_code的个数(有些API可以请求多个ts_code)
                req_per_min: 1分钟请求的次数上界
                ts_type: tushare 接口类型。pro(默认): tushare_pro; ts: tushare_ts;
                code_type: 股票code类型。ts(默认): 000001.SZ; qlib: 如SZ000001; bao: 如：sh.000001
                feas: 取数字段
        '''
        import warnings
        warnings.filterwarnings("ignore")
        self.logger.info('\n{}\n{}'.format('=' * 100, 'fetch_data_from_ts...'))
        try:
            if ts_type == 'ts':
                pro = tushare_ts()
            else:
                pro = tushare_pro()
            if feas is None:
                feas = self.feas
            if self.use_trade_day:
                date_inter = get_trade_cal_inter(self.start_date, self.end_date)
                df_list = []
                for date in date_inter:
                    date = date.replace('-', '')
                    tmp = ts_api(pro, api_fun, trade_date=date)
                    df_list.append(tmp)
                df = pd.concat(df_list, axis=0, join='outer')
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
                                 start_date=start_date, end_date=end_date, **kwargs)
                    if tmp is None or tmp.empty:
                        # self.logger.info('no data_new: {}'.format(','.join(stocks[k: k + batch_size])))
                        continue
                    df_list.append(tmp)
                    time.sleep(60 / req_per_min)
                df = pd.concat(df_list, axis=0, join='outer')

            if df.empty:
                err_msg = 'df is empty !'
                self.logger.error(err_msg)
                raise Exception(err_msg)

            df['day'] = pd.to_datetime(df['trade_date']).dt.date
            if code_type == 'qlib':
                df['qlib_code'] = df['ts_code'].apply(lambda x: '{}{}'.format(x[7:9], x[0:6]))
            elif code_type == 'bao':
                df['code'] = df['ts_code'].apply(lambda x: '{}.{}'.format(x[7:9].lower(), x[0:6]))
            df = df[feas]
            self.logger.info('df shape: {}'.format(df.shape))
            df.set_index(keys=feas[0:2], inplace=True)
            return df
        except Exception as e:
            error_msg = 'error in fetch_data_from_api: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def fetch_data_from_qlib(self, provider_uri='~/.qlib/qlib_data/custom_data_hfq', index_list=None):
        """ 从qlib文件系统读取数据 """
        self.logger.info('\n{}\n{}'.format('=' * 100, 'fetch_data_from_qlib...'))
        if index_list is None:
            index_list = []
        qlib.init(provider_uri=provider_uri)
        fields = ['${}'.format(f) for f in self.feas[2:]]
        # instruments = ['SZ300760']
        config = D.instruments(market='all')
        instruments = D.list_instruments(instruments=config, start_time=self.start_date, end_time=self.end_date,
                                         as_list=True)
        # instruments = instruments[0:20]
        instruments = list(set(instruments) - set(index_list))
        df = D.features(instruments, fields, start_time=self.start_date, end_time=self.end_date)
        na = df.isna().all(axis=1)
        df = df[~na]  # 停牌等原因没有交易数据，字段全为NaN
        if df.empty:
            error_msg = 'df is empty !'
            self.logger.error(error_msg)
            raise Exception(error_msg)
        df.columns = self.feas[2:]
        df.reset_index(inplace=True)
        df['day'] = pd.to_datetime(df['datetime']).dt.date
        df.rename(columns={'instrument': 'qlib_code'}, inplace=True)
        df = df[self.feas]
        self.logger.info('df shape: {}'.format(df.shape))
        self.logger.info('instruments len: {}'.format(len(instruments)))
        df.set_index(keys=self.feas[0:2], inplace=True)
        return df, instruments

    def check(self, df_target, df_test, is_repair=True, compare_type: str = 'eq', epsilon: float = 0.0001):
        '''
        检查 MySQL 与 API 数据是否相同
        Args:
            df_target: 待检测数据(mysql)
            df_test: 测试数据（api或mysql）
            is_repair: 当df_target与df_test不一致时，可以使用test数据修复mysql数据
            compare_type: 比较类型。eq(默认): 值相等；round: 设置误差范围
            epsilon: 误差大小
        '''
        self.logger.info('\n{}\n{}'.format('=' * 100, 'check ...'))
        try:
            self.logger.info('df_target shape: {}, df_test shape: {}'.format(df_target.shape, df_test.shape))
            if compare_type == 'eq':
                diff = (df_target.eq(df_test)) | ((df_target.isna()) & (df_test.isna()))  # 值相同 ｜ 都为NaN
            elif compare_type == 'round':
                diff = (abs((df_target - df_test) / df_test) < epsilon) | ((df_target.isna()) & (df_test.isna()))
            else:
                raise Exception('compare_type error')

            mask_ne = (diff.ne(True)).any(axis=1)
            index_ne = diff.index[mask_ne]  # 包含不相等值的行索引

            res = []  # 存放错误信息
            for index in index_ne:
                target_f = []
                test_f = []
                try:
                    target_row = df_target.loc[index]
                    for f in self.feas[2:]:
                        target_f.append('{}:{}'.format(f, target_row[f]))
                except:
                    target_f = ['NaN']

                try:
                    test_row = df_test.loc[index]
                    for f in self.feas[2:]:
                        test_f.append('{}:{}'.format(f, test_row[f]))
                except:
                    test_f = ['NaN']

                if is_repair:
                    # 修复数据
                    with MySQLDB() as db:
                        test_row = df_test.loc[index]
                        params = {
                            'day': index[0],
                            'ts_code': index[1],
                        }
                        for f in self.feas[2:]:
                            v = test_row[f]
                            if pd.isna(v):
                                v = None
                            params[f] = v
                        sql = """
                            UPDATE {} SET {} WHERE day=%(day)s AND ts_code=%(ts_code)s
                        """.format(self.table_name, ','.join(['{}=%({})s'.format(f, f) for f in self.feas[2:]]))
                        db.execute(sql, params)

                res.append('{}: [target: {};  test: {}]'.format(index, ', '.join(target_f), ', '.join(test_f)))
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
