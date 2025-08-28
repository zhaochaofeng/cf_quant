'''
    功能：用于shell脚本中交易日判断
'''

import sys
from utils import is_trade_day

if __name__ == '__main__':
    if is_trade_day(sys.argv[1]):
        # 交易日
        exit(0)
    else:
        # 非交易日
        exit(5)
