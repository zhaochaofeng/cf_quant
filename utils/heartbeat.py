#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器心跳检查脚本
用于检查指定服务器是否存活，如果不存活则发送告警邮件

功能：
1. 通过TCP连接检查单个服务器的指定端口是否开放
2. 服务器宕机时自动发送告警邮件
3. 设计为通过crontab定时执行，实现定期检查

使用示例：
python heartbeat.py 47.93.20.118 20022

Crontab配置示例（每5分钟检查一次）：
*/5 * * * * cd /Users/chaofeng/code/cf_quant/utils && python heartbeat.py 47.93.20.118 20022
# 每30分钟检查一次服务器（更宽松的检查频率）
*/30 * * * * cd /Users/chaofeng/code/cf_quant/utils && python heartbeat.py 47.93.20.118 20022 --timeout 10 >> /tmp/heartbeat_20022_detailed.log 2>&1
"""

import socket
import sys
from datetime import datetime
import argparse

# 导入同目录下的工具模块
try:
    from utils import send_email
    from logger import LoggerFactory
except ImportError:
    # 如果直接运行，尝试相对导入
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import send_email
    from logger import LoggerFactory

# 使用LoggerFactory创建日志记录器
# 日志文件会轮转，避免单个文件过大
logger = LoggerFactory.get_logger(
    name="heartbeat",
    log_file="logs/heartbeat.log",  # 日志文件路径
    level="INFO",                   # 日志级别
    max_bytes=10 * 1024 * 1024,     # 单个日志文件最大10MB
    backup_count=5,                 # 保留5个备份文件
    fmt="%(asctime)s : %(name)s:%(lineno)d : %(levelname)s : %(message)s",
    console=True                    # 同时输出到控制台
)


def check_server_alive(host: str, port: int, timeout: int = 5) -> bool:
    """
    通过TCP连接检查服务器是否存活

    使用socket的connect_ex方法尝试连接指定服务器的端口。
    connect_ex方法是非阻塞的，返回错误码而不是抛出异常。
    错误码为0表示连接成功，其他值表示连接失败。

    Args:
        host: 服务器IP地址或主机名
        port: 服务器端口号，范围1-65535
        timeout: 连接超时时间（秒），默认5秒

    Returns:
        bool: True表示服务器存活（端口开放），False表示服务器不存活

    Raises:
        不会抛出异常，所有错误都会被捕获并返回False

    Example:
        >>> check_server_alive("127.0.0.1", 8080)
        True
        >>> check_server_alive("192.168.1.100", 9999)
        False
    """
    try:
        # 创建IPv4的TCP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)  # 设置连接超时时间

        # connect_ex是非阻塞连接方法，返回错误码而不是抛出异常
        # 错误码为0表示连接成功，其他值表示连接失败
        result = sock.connect_ex((host, port))
        sock.close()  # 关闭socket连接

        if result == 0:
            logger.info(f"服务器 {host}:{port} 连接成功")
            return True
        else:
            # 常见错误码：
            # 61 - Connection refused (连接被拒绝)
            # 111 - Connection refused (Linux)
            # 10061 - Connection refused (Windows)
            logger.warning(f"服务器 {host}:{port} 连接失败，错误码: {result}")
            return False

    except socket.timeout:
        # 连接超时，服务器可能没有响应或网络延迟过高
        logger.error(f"服务器 {host}:{port} 连接超时（{timeout}秒）")
        return False
    except socket.error as e:
        # socket相关错误，如无效的主机名或端口
        logger.error(f"服务器 {host}:{port} 连接错误: {e}")
        return False
    except Exception as e:
        # 其他未知错误
        logger.error(f"检查服务器 {host}:{port} 时发生未知错误: {e}")
        return False


def send_heartbeat_alert(host: str, port: int, error_msg: str = "") -> None:
    """
    发送服务器宕机告警邮件

    当服务器检查失败时，生成告警邮件内容并发送。
    邮件主题和内容包含服务器地址、检查时间和错误信息。

    Args:
        host: 服务器IP地址
        port: 服务器端口号
        error_msg: 错误信息

    Returns:
        None

    Raises:
        不会抛出异常，所有错误都会被日志记录

    Note:
        依赖utils.py中的send_email函数发送邮件
        邮件配置在config.yaml的email部分

    Example:
        >>> send_heartbeat_alert("192.168.1.100", 8080, "连接被拒绝")
    """
    # 获取当前时间，用于邮件时间戳
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 告警邮件
    subject = f"【服务器告警】{host}:{port} 服务器宕机"
    body = f"""
服务器心跳检查告警

服务器地址: {host}:{port}
检查时间: {current_time}
服务器状态: 宕机

错误信息: {error_msg if error_msg else "无法连接到服务器"}

请立即检查服务器状态！
"""

    try:
        # 调用utils.py中的send_email函数发送邮件
        send_email(subject, body)
        logger.info(f"已发送宕机告警邮件: {host}:{port}")
    except Exception as e:
        # 邮件发送失败，记录错误但不中断程序
        logger.error(f"发送告警邮件失败: {e}")


def main() -> int:
    """
    命令行入口函数

    解析命令行参数，执行单次服务器检查。
    返回退出码：0表示服务器存活，1表示服务器不存活。

    Returns:
        int: 退出码，0表示成功，1表示失败

    Command Line Usage:
        python heartbeat.py <host> <port> [--timeout <seconds>]

    Example:
        >>> python heartbeat.py 47.93.20.118 20022
        >>> python heartbeat.py 47.93.20.118 21022 --timeout 10
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='服务器心跳检查工具 - 单次检查模式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s 47.93.20.118 20022                    # 检查服务器，默认5秒超时
  %(prog)s 47.93.20.118 21022 --timeout 10       # 检查服务器，10秒超时

Crontab配置示例:
  # 每5分钟检查一次服务器
  */5 * * * * cd /path/to/script && python %(prog)s 47.93.20.118 20022

  # 每30分钟检查一次服务器
  */30 * * * * cd /path/to/script && python %(prog)s 47.93.20.118 21022
        """
    )

    # 必需参数：服务器地址和端口
    parser.add_argument('host', help='服务器IP地址或主机名')
    parser.add_argument('port', type=int, help='服务器端口号 (1-65535)')

    # 可选参数
    parser.add_argument('--timeout', type=int, default=5,
                       help='连接超时时间（秒），默认5秒')

    # 解析命令行参数
    args = parser.parse_args()

    # 单次检查模式
    logger.info(f"开始检查服务器: {args.host}:{args.port}, 超时:{args.timeout}秒")
    is_alive = check_server_alive(args.host, args.port, args.timeout)

    if is_alive:
        # 服务器存活
        logger.info(f"服务器 {args.host}:{args.port} 检查通过")
        return 0  # 成功退出码
    else:
        # 服务器不存活
        logger.error(f"服务器 {args.host}:{args.port} 检查失败")
        send_heartbeat_alert(args.host, args.port, "单次检查失败")
        return 1  # 失败退出码


if __name__ == "__main__":
    sys.exit(main())