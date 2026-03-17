"""
    时间与日期相关函数
"""

import calendar
from datetime import datetime
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
            start = datetime.now()
            result = func(*args, **kwargs)
            end = datetime.now()
            duration = end - start
            logger.info('\n{}\nfunc "{}" consume time: {}\n{}'.format('-'*30, func.__name__, duration, '-'*30))
            return result
        return timer

    @staticmethod
    def subtract_months(end_dt: str, months: int) -> str:
        """
        从指定日期减去指定月份数

        准确处理月份计算，考虑不同月份的天数差异：
        - 3月31日 - 1个月 = 2月28/29日
        - 5月31日 - 1个月 = 4月30日

        Args:
            end_dt: 结束日期
            months: 要减去的月份数

        Returns:
            计算后的开始日期
        """
        end_dt = datetime.strptime(end_dt, '%Y-%m-%d')
        # 计算目标年份和月份
        # 减1的目的是处理当前月份与目标月份相同的情况。如 end_dt=2026-03-05, months=3
        total_months = end_dt.year * 12 + end_dt.month - 1  # 0-based month index
        target_months = total_months - months

        year = target_months // 12
        month = target_months % 12 + 1  # 转换回1-based

        # 处理日期：如果目标月份天数不足，取该月最后一天
        max_day = calendar.monthrange(year, month)[1]
        day = min(end_dt.day, max_day)

        start_dt = datetime(year, month, day)
        start_dt = datetime.strftime(start_dt, '%Y-%m-%d')
        return start_dt


# 为了向后兼容，仍然提供函数接口
time_decorator = DateTimeUtils.time_decorator



