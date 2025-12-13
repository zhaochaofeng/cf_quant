
import yaml
import tushare as ts
import numpy as np
import pandas as pd
import pymysql
import redis
from pathlib import Path
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from jqdatasdk import *

import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
import traceback

def get_config(config_path: str = None):
    ''' 读取配置文件 '''
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # print('配置文件加载完成：{}'.format(config_path))
        return config
    except Exception as e:
        print('配置文件加载失败：{}'.format(e))
    return None

def tushare_pro():
    config = get_config()
    pro = ts.pro_api(config['tushare']['token'])
    return pro

def tushare_ts():
    ''' 有些数据不能使用pro对象访问 '''
    config = get_config()
    ts.set_token(config['tushare']['token'])
    return ts

def bao_stock_connect():
    import baostock as bs
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)
    # bs.logout()  # 登出系统
    return bs

def jq_connect():
    config = get_config()
    auth(config['joinqaunt']['username'], config['joinqaunt']['password'])

def mysql_connect():
    ''' 创建mysql连接实例 '''
    config = get_config()
    conn = pymysql.connect(host=config['mysql']['host'],
                           user=config['mysql']['user'],
                           password=config['mysql']['password'],
                           db=config['mysql']['db'],
                           charset='utf8')
    return conn

def redis_connect():
    ''' 创建redis连接实例 '''
    config = get_config()
    conn = redis.Redis(host=config['redis']['host'],
                       password=config['redis']['password'])
    return conn

def sql_engine():
    ''' 创建sqlalchemy连接mysql的引擎 '''
    config = get_config()
    url = "mysql+pymysql://{}:{}@{}:3306/{}?charset=utf8mb4".format(
        config['mysql']['user'],
        config['mysql']['password'],
        config['mysql']['host'],
        config['mysql']['db']
    )
    engine = create_engine(url, echo=False)
    return engine

def get_feas(config_path: str = None):
    ''' 读取特征字段 '''
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'features.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # print('配置文件加载完成：{}'.format(config_path))
        return config
    except Exception as e:
        print('配置文件加载失败：{}'.format(e))
    return None

def is_trade_day(date):
    ''' 判断是否为交易日 '''
    pro = tushare_pro()
    date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
    return True if (pro.trade_cal(start_date=date, end_date=date)['is_open'][0] == 1) else False

def get_n_pretrade_day(date, n):
    ''' date 前n个交易日。格式 YYYY-MM-DD
        若date为非交易日，则会将date向前移动到最近的一个交易日。
        如 get_n_pretrade_day('2025-08-30', 1), 2025-08-30为非交易日，最近的一个交易日为2025-08-29，则函数返回2025-08-28
        n=0时，返回原始日期date，如果为非交易日，则返回历史离date最近的交易日
    '''
    pro = tushare_pro()
    trade_days = pro.trade_cal(
        exchange='',
        start_date=(datetime.strptime(date, '%Y-%m-%d') - timedelta(days=(n+1000))).strftime('%Y%m%d'),
        end_date=datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d'),
        is_open='1')
    trade_days = trade_days['cal_date'].tolist()
    trade_days.sort()
    return datetime.strptime(trade_days[-(n+1)], '%Y%m%d').strftime('%Y-%m-%d')

def get_n_nexttrade_day(date, n):
    ''' date 后n个交易日
        若date为非交易日，则会将date向后移动到最近的一个交易日
    '''
    pro = tushare_pro()
    trade_days = pro.trade_cal(
        exchange='',
        start_date=date.replace('-', ''),
        end_date=(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=(n+1000))).strftime('%Y%m%d'),
        is_open='1')
    trade_days = trade_days['cal_date'].tolist()
    trade_days.sort()
    return datetime.strptime(trade_days[n], '%Y%m%d').strftime('%Y-%m-%d')

def get_trade_cal_inter(start_date, end_date):
    '''
    返回[start_date, end_date]之间的交易日列表
    :param start_date: 起始日期，格式YYYY-MM-DD
    :param end_date: 终止日期，格式YYYY-MM-DD
    :return: list
    '''
    pro = tushare_pro()
    start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
    df = pro.trade_cal(start_date=start_date, end_date=end_date, is_open='1')
    cal_list = pd.to_datetime(df['cal_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d').values.tolist()
    cal_list.sort()   # 从旧到新排序
    return cal_list

def get_month_start_end(date_str):
    """
    根据输入日期字符串，返回该月的第一天和最后一天
    参数:
        date_str: 日期字符串，格式为 'YYYY-MM-DD'
    返回:
        (month_start, month_end): 元组，包含该月第一天和最后一天的日期字符串（格式为 'YYYY-MM-DD'）
    """
    # 将字符串转换为datetime对象
    date = datetime.strptime(date_str, '%Y-%m-%d')
    # 获取当月第一天
    month_start = date.replace(day=1)
    # 获取下个月的第一天，并减去一天得到本月的最后一天
    if month_start.month == 12:
        next_month = month_start.replace(year=month_start.year + 1, month=1)
    else:
        next_month = month_start.replace(month=month_start.month + 1)
    month_end = next_month - timedelta(days=1)
    # 转换为字符串格式输出
    return month_start.strftime('%Y-%m-%d'), month_end.strftime('%Y-%m-%d')

def send_email(subject, body):
    ''' 报警邮件
        subject: 邮件主题
        body: 邮件正文
    '''
    conf = get_config()['email']
    sender = conf['sender']
    password = conf['password']
    smtpserver = conf['smtpserver']
    port = conf['port']
    receiver = conf['receiver']

    # 构建邮件
    msg = MIMEText(body, 'plain', 'utf-8')
    msg['From'] = formataddr(("报警系统", sender))   # 发件人信息
    msg['To'] = formataddr(("接收人", receiver))     # 收件人信息
    msg['Subject'] = Header(subject, 'utf-8')       # 邮件主题

    try:
        smtp = smtplib.SMTP_SSL(smtpserver, port)       # 创建SMTP连接
        smtp.login(sender, password)                    # 登录SMTP服务器
        smtp.sendmail(sender, [receiver], msg.as_string())  # 发送邮件
        smtp.quit()                                         # 退出SMTP连接
        print("邮件发送成功 ！")
    except Exception as e:
        print("邮件发送失败:", e)

def bao_api(bs, api_func, **kwargs) -> pd.DataFrame:
    """
    通过BaoStock API函数名和参数获取数据

    Args:
        api_func: baostock API函数名
        **kwargs: 传递给API函数的参数

    Returns:
        pd.DataFrame: 获取的数据
    """
    try:
        # 获取API函数
        api_function = getattr(bs, api_func)
        # 调用API函数并传入参数
        df = api_function(**kwargs).get_data()
        return df
    except AttributeError:
        raise AttributeError(f"BaoStock API中不存在函数: {api_func}")
    except Exception as e:
        raise Exception(f"调用API {api_func} 时发生错误: {str(e)}")


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


if __name__ == '__main__':
    print(get_n_nexttrade_day('2025-09-20', 1))

