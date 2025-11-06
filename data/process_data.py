'''
    从TuShare/BaoStock等平台将数据导入MySQL
'''

import time
import pandas as pd
from datetime import datetime
from abc import ABC, abstractmethod
from utils import LoggerFactory
from utils import MySQLDB
from utils import sql_engine, tushare_pro, bao_stock_connect
from utils import get_trade_cal_inter, is_trade_day
from utils import ts_api, bao_api


class Base(ABC):
    def __init__(self,
                 feas: dict,
                 table_name: str,
                 log_file=None,
                 ):
        '''
        Args:
            feas: mysql表字段与平台表字段映射关系。如{'day': 'trade_date'}
            table_name: 输入插入的mysql表名。如stock_info_ts
            log_file: 日志文件路径。如log/{}.log.format(day)
        '''
        self.feas = feas
        self.table_name = table_name
        self.logger = LoggerFactory.get_logger(__name__, log_file=log_file)

    @abstractmethod
    def fetch_data_from_api(self) -> pd.DataFrame:
        ''' 从平台获取数据 '''
        pass

    @abstractmethod
    def parse_line(self, row) -> dict:
        ''' 解析一行数据 '''
        pass

    def process(self, df) -> list:
        """ 数据处理 """
        self.logger.info('\n{}\n{}'.format('=' * 100, 'process ...'))
        try:
            data = []
            for index, row in df.iterrows():
                tmp = self.parse_line(row)
                data.append(tmp)
            self.logger.info('data len: {}'.format(len(data)))
            if len(data) == 0:
                err_msg = 'data is empty !'
                self.logger.error(err_msg)
                raise Exception(err_msg)
            return data
        except Exception as e:
            error_msg = 'error in process: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def write_to_mysql(self, data) -> None:
        """ 数据写入 MySQL """
        self.logger.info('\n{}\n{}'.format('=' * 100, 'write_to_mysql...'))
        try:
            feas = list(self.feas.keys())
            feas_format = ['%({})s'.format(f) for f in feas]
            sql = """
                INSERT INTO {} ({}) VALUES({})
            """.format(self.table_name, ','.join(feas).replace(',change', ',`change`'), ','.join(feas_format))

            with MySQLDB() as db:
                db.executemany(sql, data)
        except Exception as e:
            error_msg = 'write_to_mysql error: {}'.format(e)
            self.logger.error(error_msg)


class ProcessData(Base):
    def __init__(self, now_date: str = None, **kwargs):
        '''
        Args:
            now_date: 指定获取股票集合的日期，默认为当天
        '''
        super().__init__(**kwargs)
        self.now_date = now_date if now_date else datetime.now().strftime('%Y-%m-%d')

    def get_stocks(self, table_name: str = 'stock_info_ts', is_alive: bool = False):
        """ 获取股票列表
            Args:
                 table_name: 股票信息表名
                 is_alive: 是否仅获取当前上市的股票（排除退市）
        """
        self.logger.info('\n{}\n{}'.format('=' * 100, 'get_stocks...'))
        engine = sql_engine()
        sql = '''
                select ts_code from {} where day='{}'
            '''.format(table_name, self.now_date)
        if is_alive:
            sql += ' and status=1'

        self.logger.info('\n{}\n{}\n{}'.format('-'*50, sql, '-'*50))
        df = pd.read_sql(sql, engine)
        if df.empty:
            err_msg = 'table stock_info_ts has no data: {}'.format(self.now_date)
            self.logger.info(err_msg)
            raise Exception(err_msg)
        codes = df['ts_code'].values.tolist()
        self.logger.info('stocks len: {}'.format(len(codes)))
        return codes


class TSFinacialData(ProcessData):
    ''' Tushare 财务数据处理类 '''
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 now_date: str = None,
                 **kwargs):
        super().__init__(now_date=now_date, **kwargs)
        self.start_date = start_date
        self.end_date = end_date
        self.has_financial_data = True
        # [1月1日-4月30, 7月1日-8月31日，10月1日-10月31]
        # 仅在当前的上述日期范围内才进行财务数据获取
        inter_valid = get_trade_cal_inter(self.get_date(1, 1), self.get_date(4, 30)) + \
                    get_trade_cal_inter(self.get_date(7, 1), self.get_date(8, 31)) + \
                    get_trade_cal_inter(self.get_date(10, 1), self.get_date(10, 31))
        inter_get = get_trade_cal_inter(self.start_date, self.end_date)
        if len(set(inter_valid) & set(inter_get)) == 0:
            self.logger.info('No finacial report released !!!')
            self.has_financial_data = False

    def get_date(self, month, day):
        curr_year = datetime.now().year
        return '{}-{:02d}-{:02d}'.format(curr_year, month, day)

    def fetch_data_from_api(self, stocks, api_fun):
        """ 从Tushare获取财务数据 """
        import warnings
        warnings.filterwarnings("ignore")
        self.logger.info('\n{}\n{}'.format('=' * 100, 'fetch_data_from_api...'))
        try:
            pro = tushare_pro()
            df_list = []
            start_date = self.start_date.replace('-', '')
            end_date = self.end_date.replace('-', '')
            for i, stock in enumerate(stocks):
                if (i + 1) % 100 == 0:
                    self.logger.info('processed num: {}'.format(i + 1))
                tmp = ts_api(pro, api_fun, ts_code=stock, start_date=start_date, end_date=end_date)
                if tmp.empty:
                    continue
                df_list.append(tmp)
                time.sleep(60/500)  # 1min最多请求500次
            df = pd.concat(df_list, axis=0, join='outer')
            if df.empty:
                msg = 'df is empty: {}'.format(self.now_date)
                self.logger.error(msg)

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
                v = row[self.feas[f]]
                if pd.isna(v):
                    v = None
                elif f == 'qlib_code':
                    code, suffix = v.split('.')
                    v = '{}{}'.format(suffix.upper(), code)
                elif f in ['ann_date', 'f_ann_date', 'end_date']:
                    # 日期格式转换
                    v = datetime.strptime(v, '%Y%m%d').strftime('%Y-%m-%d')
                tmp[f] = v
            return tmp
        except Exception as e:
            error_msg = 'parse_line error: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)


class TSCommonData(ProcessData):
    ''' Tushare 通用格式数据 '''
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 now_date: str = None,
                 use_trade_day: bool = False,
                 **kwargs
                 ):
        '''
        Args:
            use_trade_day: 是否指定api中trade_date参数，若指定，则trade_date=end_date
        '''

        super().__init__(now_date=now_date, **kwargs)
        self.start_date = start_date
        self.end_date = end_date
        self.use_trade_day = use_trade_day
        self.is_trade_day = True   # 是否为交易日
        if not is_trade_day(self.end_date):
            msg = '{} is not a trade date, exit !!!'.format(self.end_date)
            self.logger.warning(msg)
            self.is_trade_day = False

    def fetch_data_from_api(self, stocks: list, api_fun: str, batch_size: int = 1, req_per_min: int = 600):
        ''' 从Tushare获取通用数据
        Args:
            batch_size: 1次请求ts_code的个数(有些API可以请求多个ts_code)
            req_per_min: 1分钟请求的次数上界
        '''
        import warnings
        warnings.filterwarnings("ignore")
        self.logger.info('\n{}\n{}'.format('=' * 100, 'fetch_data_from_api...'))
        try:
            pro = tushare_pro()
            if self.use_trade_day and self.start_date == self.end_date:
                trade_date = self.end_date.replace('-', '')
                df = ts_api(pro, api_fun, trade_date=trade_date)
                df = df[df['ts_code'].isin(stocks)]   # 过滤股票
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
                    if pd.isna(v):
                        v = None
                    elif f == 'qlib_code':
                        code, suffix = v.split('.')
                        v = '{}{}'.format(suffix.upper(), code)
                    elif f == 'day':
                        # 日期格式转换
                        v = datetime.strptime(v, '%Y%m%d').strftime('%Y-%m-%d')
                tmp[f] = v
            return tmp
        except Exception as e:
            error_msg = 'parse_line error: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)


class TSTradeDailyData(TSCommonData):
    ''' Tushare 日级交易数据 '''

    def __init__(self,
                 start_date: str,
                 end_date: str,
                 now_date: str = None,
                 use_trade_day: bool = True,
                 **kwargs
                 ):
        super().__init__(start_date, end_date, now_date, use_trade_day, **kwargs)

    def fetch_data_from_api(self, stocks: list, api_fun: str, batch_size: int = 1000, req_per_min: int = 600):
        ''' 从Tushare获取交易数据 + 赋权因子
        Args:
            batch_size: 1次请求ts_code的个数(有些API可以请求多个ts_code)
            req_per_min: 1分钟请求的次数上界
        '''
        # import warnings
        # warnings.filterwarnings("ignore")
        self.logger.info('\n{}\n{}'.format('=' * 100, 'fetch_data_from_api...'))
        try:
            pro = tushare_pro()
            if self.use_trade_day and self.start_date == self.end_date:
                trade_date = self.end_date.replace('-', '')
                df = ts_api(pro, api_fun, trade_date=trade_date)
                df = df[df['ts_code'].isin(stocks)]   # 过滤股票
                self.logger.info('df shape: {}'.format(df.shape))
                df.set_index(keys=['ts_code', 'trade_date'], inplace=True)
                # 复权因子
                factor = pro.adj_factor(trade_date=trade_date)
                self.logger.info('factor shape: {}'.format(factor.shape))
                factor.set_index(keys=['ts_code', 'trade_date'], inplace=True)
                if df.empty or factor.empty:
                    err_msg = 'df({}) or factor({}) is empty !'.format(df.shape, factor.shape)
                    self.logger.error(err_msg)
                    raise Exception(err_msg)
                # 合并 交易数据 和 复权因子
                factor = factor.reindex(df.index)
                merged = pd.concat([df, factor], axis=1, join='outer')
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
                factor_list = []
                for k in range(0, len(stocks), batch_size):
                    if (k + 1) % 100 == 0:
                        self.logger.info('processed : {} / {}'.format(k + batch_size, len(stocks)))
                    tmp = ts_api(pro, api_fun,
                                 ts_code=','.join(stocks[k:k + batch_size]),
                                 start_date=start_date, end_date=end_date)
                    tmp_factor = ts_api(pro, 'adj_factor',
                                 ts_code=','.join(stocks[k:k + batch_size]),
                                 start_date=start_date, end_date=end_date)
                    if not tmp.empty:
                        df_list.append(tmp)
                    if not tmp_factor.empty:
                        factor_list.append(tmp_factor)
                    time.sleep(60 / req_per_min)
                df = pd.concat(df_list, axis=0, join='outer')
                factor = pd.concat(factor_list, axis=0, join='outer')
                if df.empty or factor.empty:
                    err_msg = 'df({}) or factor({}) is empty !'.format(df.shape, factor.shape)
                    self.logger.error(err_msg)
                    raise Exception(err_msg)

                self.logger.info('df shape: {}, factor shape: {}'.format(df.shape, factor.shape))
                df.set_index(keys=['ts_code', 'trade_date'], inplace=True)
                factor.set_index(keys=['ts_code', 'trade_date'], inplace=True)

                # 合并 交易数据 和 复权因子
                factor = factor.reindex(df.index)
                merged = pd.concat([df, factor], axis=1, join='outer')

            merged.reset_index(inplace=True)
            self.logger.info('merged shape: {}'.format(merged.shape))
            return merged
        except Exception as e:
            error_msg = 'error in fetch_data_from_api: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)

class BaoCommonData(ProcessData):
    ''' BaoStack 通用格式数据 '''
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 now_date: str = None,
                 **kwargs
                 ):

        super().__init__(now_date=now_date, **kwargs)
        self.start_date = start_date
        self.end_date = end_date
        self.is_trade_day = True   # 是否为交易日
        if not is_trade_day(self.end_date):
            msg = '{} is not a trade date, exit !!!'.format(self.end_date)
            self.logger.warning(msg)
            self.is_trade_day = False

    def fetch_data_from_api(self, stocks: list, api_fun: str) -> pd.DataFrame:
        ''' 从BaoStock获取通用数据
        Args:
            batch_size: 1次请求ts_code的个数(有些API可以请求多个ts_code)
            req_per_min: 1分钟请求的次数上界
        '''
        pass

    def parse_line(self, row) -> dict:
        ''' 解析单条数据 '''
        try:
            tmp = {}
            for f in self.feas.keys():
                if f == 'day' and self.feas[f] == '':
                    v = self.now_date
                else:
                    v = row[self.feas[f]]
                    if pd.isna(v):
                        v = None
                    elif f == 'qlib_code':
                        suffix, code = v.split('.')
                        v = '{}{}'.format(suffix.upper(), code)
                    elif f == 'pct_chg':
                        if abs(v) > 9999.99:
                            self.logger.warning(f"警告: pct_chg值 {v} 超出范围，已截断为9999.99或-9999.99")
                            self.logger.warning(row)
                            v = 9999.99 if v > 0 else -9999.99
                    elif f == 'vol':
                        v = v / 100  # 股转化为手
                    elif f == 'amount':
                        v = v / 1000  # 元转为千元
                tmp[f] = v
            return tmp
        except Exception as e:
            error_msg = 'parse_line error: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)

class BaoTradeDailyData(BaoCommonData):
    ''' BaoStock 日级交易数据 '''
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 now_date: str = None,
                 **kwargs
                 ):
        super().__init__(start_date, end_date, now_date, **kwargs)
        self.start_date = start_date
        self.end_date = end_date
        self.now_date = now_date
        self.bs = bao_stock_connect()

    def fetch_data_from_api(self, stocks: list, api_fun: str, round_dic: dict) -> pd.DataFrame:
        self.logger.info('\n{}\n{}'.format('=' * 100, 'fetch_data_from_api...'))
        try:
            fea_bao = list(self.feas.values())
            [fea_bao.remove(f) for f in ['qlib_code', 'adj_factor'] if f in fea_bao]

            df_list = []
            factor_list = []
            for i, stock in enumerate(stocks):
                if (i + 1) % 100 == 0:
                    self.logger.info('processed : {} / {}'.format(i + 1, len(stocks)))
                rs = bao_api(self.bs, api_fun,
                              code=stock, fields="{}".format(','.join(fea_bao)),
                              start_date=self.start_date, end_date=self.end_date,
                              frequency="d", adjustflag="3"
                             )
                df = rs.get_data()
                if df.empty:
                    continue
                df = df[~(df['amount'] == '')]  # BaoStock在停牌日也能请求到数据，volume/amount为'', 需要排除
                if df.empty:
                    continue
                factor = self.get_factor(stock, self.start_date, self.end_date)
                if factor.empty:
                    continue

                df_list.append(df)
                factor_list.append(factor)

            df = pd.concat(df_list, axis=0, join='outer')
            factor = pd.concat(factor_list, axis=0, join='outer')
            if df.empty or factor.empty:
                err_msg = 'df({}) or factor({}) is empty !'.format(df.shape, factor.shape)
                self.logger.error(err_msg)
                raise Exception(err_msg)
            self.logger.info('df shape: {}'.format(df.shape))
            self.logger.info('factor shape: {}'.format(factor.shape))
            df.set_index(keys=['code', 'date'], inplace=True)
            factor.set_index(keys=['code', 'date'], inplace=True)
            factor = factor.reindex(df.index)
            merged = pd.concat([df, factor], axis=1, join='outer')
            self.logger.info('merged shape: {}'.format(merged.shape))

            print(merged.head())

            for f in merged.columns:
                merged[f] = pd.to_numeric(merged[f], errors='coerce')
            merged.reset_index(inplace=True)
            merged = merged.round(round_dic)
            print(merged.head())
            return merged
        except Exception as e:
            error_msg = 'fetch_data_from_api error: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def get_factor(self, code, start_date, end_date):
        """
        获取指定股票在指定日期范围内每天的后复权因子

        参数:
        code (str): 股票代码，格式如"sh.600000"
        start_date (str): 起始日期，格式为"YYYY-MM-DD"
        end_date (str): 终止日期，格式为"YYYY-MM-DD"

        返回:
        pd.DataFrame: 包含code, date, adj_factor字段的DataFrame，包含日期范围内每一天的数据
        """
        self.logger.info('\n{}\n{}'.format('=' * 100, 'get_factor...'))
        try:
            # 查询指定股票的除权除息数据（包含后复权因子）
            rs = self.bs.query_adjust_factor(
                code=code,
                start_date="1990-01-01",  # 从较早日期开始查询
                end_date=end_date
            )

            if rs.error_code != '0':
                raise Exception(f"获取{code}复权因子失败: {rs.error_msg}")

            # 生成指定日期范围内的所有日期
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            full_dates = pd.DataFrame({'date': date_range})

            # 获取除权除息日数据
            df = rs.get_data()
            if df.empty:
                self.logger.warning(f"未获取到{code}的除权除息数据")
                result_df = full_dates.copy()
                result_df['code'] = code
                result_df['adj_factor'] = 1.0
                return result_df[['code', 'date', 'adj_factor']]

            # print(df[['code', 'dividOperateDate', 'backAdjustFactor']])

            # 筛选出需要的后复权因子列并重命名
            df = df[['code', 'dividOperateDate', 'backAdjustFactor']].rename(
                columns={'dividOperateDate': 'date', 'backAdjustFactor': 'adj_factor'}
            )

            # 转换日期为datetime类型，因子为数值类型
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            df['adj_factor'] = pd.to_numeric(df['adj_factor'], errors='coerce')

            # 关键修复：找到起始日期前最近的除权除息日因子值
            start_dt = pd.to_datetime(start_date)
            # start_date这一天不是除权除息日
            if df[df['date'] == start_dt].empty:
                # 筛选出在起始日期之前的factor
                before_start = df[df['date'] < start_dt]
                if not before_start.empty:
                    # 取起始日期前最近的一个因子值
                    latest_before_start = before_start.sort_values('date', ascending=False).iloc[0]
                    # 将该因子值添加到合并数据中
                    pre_start_row = pd.DataFrame({
                        'date': [start_dt],
                        'adj_factor': [latest_before_start['adj_factor']]
                    })
                else:
                    pre_start_row = pd.DataFrame({
                        'date': [start_dt],
                        'adj_factor': [1.0]
                    })
                    # 合并到原始除权除息数据中
                df = pd.concat([df, pre_start_row], ignore_index=True)

            # 将除权除息日数据与完整日期序列合并
            merged_df = pd.merge(full_dates, df[['date', 'adj_factor']], on='date', how='left')

            # 调试：检查合并后的数据
            # print(f"合并后的数据样本:")
            # print(merged_df[merged_df['adj_factor'].notna()])

            # 向前填充因子值（关键修复点：确保填充逻辑正确）
            merged_df['adj_factor'] = merged_df['adj_factor'].ffill()

            # 添加股票代码列
            merged_df['code'] = code

            # 调整列顺序并转换日期为字符串格式
            result_df = merged_df[['code', 'date', 'adj_factor']]
            result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')

            return result_df

        except Exception as e:
            error_msg = f"获取{code}的复权因子失败: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(f"发生异常: {str(e)}")
