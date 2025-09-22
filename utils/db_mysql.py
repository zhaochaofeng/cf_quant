'''
    MySQL 操作模版
'''

import pymysql
from utils import get_config

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

    def query(self, sql, params=None):
        ''' 查询 '''
        try:
            s = self.cursor.mogrify(sql, params or ())
            print('{}\n{}\n{}'.format('-' * 100, s, '-' * 100))
            self.cursor.execute(sql, params or ())
            return self.cursor.fetchall()
        except pymysql.MySQLError as e:
            raise Exception('error in query: {}'.format(e))

    def execute(self, sql,params=None):
        ''' 增删改 '''
        try:
            s = self.cursor.mogrify(sql, params or ())
            print('{}\n{}\n{}'.format('-' * 100, s, '-' * 100))
            self.cursor.execute(sql, params or ())
            self.conn.commit()
        except pymysql.MySQLError as e:
            self.conn.rollback()
            raise Exception('error in execute: {}'.format(e))

    def executemany(self, sql, params_list):
        ''' 增删改（批量） '''
        try:
            s = self.cursor.mogrify(sql, params_list[0])
            print('{}\n{}\n{}'.format('-' * 100, s, '-' * 100))
            self.cursor.executemany(sql, params_list)
            self.conn.commit()
        except pymysql.MySQLError as e:
            self.conn.rollback()  # 回滚
            raise Exception('error in executemany: {}'.format(e))
        finally:
            self.conn.close()

    def close(self):
        self.cursor.close()
        self.conn.close()
        print('关闭数据库连接!!!')

if __name__ == '__main__':
    db = MySQLDB()
    # query
    # sql = """ select * from test where day>=%s """
    # print(db.query(sql, '2025-09-03'))
    '''
    [{'id': 3, 'day': datetime.date(2025, 9, 3)}, {'id': 4, 'day': datetime.date(2025, 9, 4)}]
    '''

    # execute
    data = {'day': '2025-09-05'}
    sql = """ insert into test (day) values (%(day)s)"""
    db.execute(sql, data)

    # executemany
    # data_list = [{'day': '2025-09-03'}, {'day': '2025-09-04'}]
    # sql = ''' insert into test (day) values (%(day)s)'''.format(data_list)
    # db.executemany(sql, data_list)





