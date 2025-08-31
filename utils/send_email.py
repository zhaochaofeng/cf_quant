'''
    功能：用于shell脚本执行失败告警
'''

import sys
from utils import send_email

if __name__ == '__main__':
    subject = sys.argv[1]
    body = sys.argv[2]
    send_email(subject, body)
