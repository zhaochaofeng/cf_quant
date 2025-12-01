'''
    MySQL 操作模版
'''

import pymysql
from .utils import get_config

class MySQLDB:
    def __init__(self):
        config = get_config()
        self.conn = pymysql.connect(
            host=config['mysql']['host'],
            user=config['mysql']['user'],
            password=config['mysql']['password'],
            db=config['mysql']['db'],
            charset='utf8',
            cursorclass=pymysql.cursors.DictCursor
        )
        self.cursor = self.conn.cursor()

    def __enter__(self):
        ''' 进入上下文管理器 '''
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器，自动提交或回滚"""
        try:
            if exc_type:
                self.conn.rollback()
                print(f"事务回滚：{exc_val}")
            else:
                self.conn.commit()
        except Exception as e:
            # 确保即使在提交/回滚时出错也能关闭连接
            print(f"退出上下文时发生错误：{e}")
            if self.conn:
                self.conn.rollback()
        finally:
            self.close()
        # 返回 False 让异常继续向上抛出
        return False

    def query(self, sql, params=None):
        ''' 查询 '''
        try:
            s = self.cursor.mogrify(sql, params or ())
            print('{}\n{}\n{}'.format('-' * 100, s, '-' * 100))
            self.cursor.execute(sql, params or ())
            return self.cursor.fetchall()
        except pymysql.MySQLError as e:
            raise Exception('error in query: {}'.format(e))

    def execute(self, sql, params=None):
        ''' 增删改（单条） '''
        try:
            s = self.cursor.mogrify(sql, params or ())
            print('{}\n{}\n{}'.format('-' * 100, s, '-' * 100))
            self.cursor.execute(sql, params or ())
        except pymysql.MySQLError as e:
            raise Exception('error in execute: {}'.format(e))

    def executemany(self, sql, params_list, batch_size=10000, auto_commit=False):
        ''' 增删改（批量） '''
        if len(params_list) == 0:
            return
        try:
            s = self.cursor.mogrify(sql, params_list[0])
            print('{}\n{}\n{}'.format('-' * 100, s, '-' * 100))
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
            print('关闭数据库连接时发生错误：{}'.format(e))

if __name__ == '__main__':
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
        data_list = [{'day': '2025-09-07'}, {'day': '2025-09-08'}]
        sql = ''' insert into test (day) values (%(day)s)'''
        db.executemany(sql, data_list)





