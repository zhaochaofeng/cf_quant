'''
    从TuShare/BaoStock等平台将数据导入MySQL
'''

import time
import pandas as pd
from datetime import datetime
from abc import ABC, abstractmethod
from utils import LoggerFactory
from utils import MySQLDB
from utils import sql_engine, tushare_pro
from utils import get_trade_cal_inter


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


class Base(ABC):
    def __init__(self,
                 feas: dict,
                 table_name: str,
                 log_file=None,
                 ):
        self.feas = feas
        self.table_name = table_name
        self.logger = LoggerFactory.get_logger(__name__, log_file=log_file)

    @abstractmethod
    def fetch_data_from_api(self):
        ''' 从平台获取数据 '''
        pass

    @abstractmethod
    def parse_line(self, row):
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

    def write_to_mysql(self, data):
        """ 数据写入 MySQL """
        self.logger.info('\n{}\n{}'.format('=' * 100, 'write_to_mysql...'))
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


class TSProcessData(Base):
    def __init__(self, now_date: str = None, **kwargs):
        super().__init__(**kwargs)
        self.now_date = now_date if now_date else datetime.now().strftime('%Y-%m-%d')

    def get_stocks(self):
        """ 获取股票列表 """
        self.logger.info('\n{}\n{}'.format('=' * 100, 'get_stocks...'))
        engine = sql_engine()
        sql = '''
                select ts_code from stock_info_ts where day='{}';
            '''.format(self.now_date)
        self.logger.info('\n{}\n{}\n{}'.format('-'*50, sql, '-'*50))
        df = pd.read_sql(sql, engine)
        if df.empty:
            err_msg = 'table stock_info_ts has no data: {}'.format(self.now_date)
            self.logger.info(err_msg)
            raise Exception(err_msg)
        codes = df['ts_code'].values.tolist()
        self.logger.info('stocks len: {}'.format(len(codes)))
        return codes

class TSFinacialData(TSProcessData):
    ''' 财务数据处理类 '''
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
                time.sleep(60/700)  # 1min最多请求700次
            df = pd.concat(df_list, axis=0, join='outer')
            if df.empty:
                msg = 'df is empty: {}'.format(self.now_date)
                self.logger.error(msg)

            return df
        except Exception as e:
            error_msg = 'error in fetch_data_from_api: {}'.format(e)
            self.logger.error(error_msg)
            raise Exception(error_msg)


    def parse_line(self, row):
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






