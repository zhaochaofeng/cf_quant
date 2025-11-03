'''
    从TuShare/BaoStock等平台将数据导入MySQL
'''

import pandas as pd
from abc import ABC, abstractmethod
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

    def process(self, df) -> list:
        self.logger.info('\n{}\n{}'.format('=' * 100, 'process ...'))
        try:
            data = []
            for index, row in df.iterrows():
                tmp = self.parse_line(row)    # parse_line需要在子类中实现
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






