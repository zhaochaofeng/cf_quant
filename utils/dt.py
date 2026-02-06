"""
    时间与日期相关函数
"""

import time
import datetime
from functools import wraps
from utils.logger import LoggerFactory

# 模块级别的 logger，避免重复创建
logger = LoggerFactory.get_logger(name=__name__, level="INFO")


class DateTimeUtils:
    """时间日期工具类"""
    
    @staticmethod
    def time_decorator(func):
        """
        计算函数执行时间的装饰器
        Examples:

        time_decorator = DateTimeUtils.time_decorator
        @time_decorator
        def test():
            time.sleep(2)
            print('hello world')
        test()
        """
        @wraps(func)
        def timer(*args, **kwargs):
            start = datetime.datetime.now()
            result = func(*args, **kwargs)
            end = datetime.datetime.now()
            duration = end - start
            logger.info('\n{}\nfunc "{}" consume time: {}\n{}'.format('-'*30, func.__name__, duration, '-'*30))
            return result
        return timer


# 为了向后兼容，仍然提供函数接口
time_decorator = DateTimeUtils.time_decorator



