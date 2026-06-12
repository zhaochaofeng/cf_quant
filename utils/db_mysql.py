'''
    MySQL 操作模版
'''
import pandas as pd
import pymysql
from typing import Union
from .utils import get_config
from .logger import LoggerFactory
logger = LoggerFactory.get_logger(__name__)


class MySQLDB:
    def __init__(self):
        config = get_config()
        self.conn = pymysql.connect(
            host=config['mysql']['host'],
            user=config['mysql']['user'],
            password=config['mysql']['password'],
            db=config['mysql']['db'],
            port=config['mysql']['port'],
            charset='utf8',
            cursorclass=pymysql.cursors.DictCursor
        )
        self.cursor = self.conn.cursor()
        # logger = LoggerFactory.get_logger(__name__)

    def __enter__(self):
        ''' 进入上下文管理器 '''
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器，自动提交或回滚"""
        commit_error = None
        try:
            if exc_type:
                self.conn.rollback()
                logger.error(f"事务回滚：{exc_val}")
            else:
                self.conn.commit()
        except Exception as e:
            logger.error(f"退出上下文时发生错误：{e}")
            if self.conn:
                self.conn.rollback()
            # 记录commit异常，稍后抛出
            if not exc_type:
                commit_error = e
        finally:
            self.close()
        # 如果commit失败且没有原始异常，主动抛出commit异常
        if commit_error:
            raise Exception(f'事务提交失败并已回滚：{commit_error}')
        # 返回 False 让原始异常继续向上抛出
        return False

    def query(self, sql, params=None):
        ''' 查询 '''
        try:
            s = self.cursor.mogrify(sql, params or ())
            logger.info('\n{}\n{}\n{}'.format('-' * 50, s, '-' * 50))
            self.cursor.execute(sql, params or ())
            return self.cursor.fetchall()
        except pymysql.MySQLError as e:
            raise Exception('error in query: {}'.format(e))

    def execute(self, sql, params=None):
        ''' 增删改（单条） '''
        try:
            s = self.cursor.mogrify(sql, params or ())
            logger.info('\n{}\n{}\n{}'.format('-' * 50, s, '-' * 50))
            self.cursor.execute(sql, params or ())
        except pymysql.MySQLError as e:
            raise Exception('error in execute: {}'.format(e))

    def executemany(self, sql, params_list, batch_size=10000, auto_commit=False):
        ''' 增删改（批量） '''
        if len(params_list) == 0:
            return
        try:
            s = self.cursor.mogrify(sql, params_list[0])
            logger.info('\n{}\n{}\n{}'.format('-' * 50, s, '-' * 50))
            for k in range(0, len(params_list), batch_size):
                self.cursor.executemany(sql, params_list[k: k+batch_size])
                # 当数据量大时，按批提交
                if auto_commit:
                    self.conn.commit()
        except pymysql.MySQLError as e:
            raise Exception('error in executemany: {}'.format(e))

    def close(self):
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            # print('关闭数据库连接!!!')
        except Exception as e:
            err_msg = '关闭数据库连接时发生错误：{}'.format(e)
            logger.error(err_msg)
            raise Exception(err_msg)


def write_to_mysql(table_name: str,
                   rows:Union[list[dict], pd.DataFrame],
                   fields: list,
                   unique_key: list = None,
                   overwrite: bool=False):
    """ 将数据写入MySQL

    Args:
        table_name (str): 表名
        rows (Union[list[dict], pd.DataFrame]): 数据
        fields (list): 字段
        unique_key (list, optional): 唯一键. Defaults to None.
        overwrite (bool, optional): 是否覆盖. Defaults to False.
    """
    columns = ', '.join(fields)
    values = ', '.join([f"%({f})s" for f in fields])
    sql = f"""
        INSERT INTO {table_name} ({columns}) VALUES ({values}) 
    """
    if overwrite:
        if not unique_key:
            raise ValueError("Unique key is required for overwrite")
        update_clause = ", ".join([f"{f}=VALUES({f})" for f in fields if f not in unique_key])
        sql += f"ON DUPLICATE KEY UPDATE {update_clause}"
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict(orient='records')
    with MySQLDB() as db:
        db.executemany(sql, rows)
    logger.info("Saved %d rows to %s ", len(rows), table_name)


def unit_test():
    """ 测试用例 """
    with MySQLDB() as db:
        # query
        # sql = """ select * from test where day>=%s """
        # print(db.query(sql, '2025-09-03'))
        '''
        [{'id': 3, 'day': datetime.date(2025, 9, 3)}, {'id': 4, 'day': datetime.date(2025, 9, 4)}]
        '''

        # execute
        # data_new = {'day': '2025-09-10'}
        # sql = """ insert into test (day) values (%(day)s)"""
        # db.execute(sql, data_new)

        # executemany
        data_list = [{'day': '2025-10-11'}, {'day': '2025-10-12'}]
        sql = ''' insert into test (day) values (%(day)s)'''
        db.executemany(sql, data_list)
