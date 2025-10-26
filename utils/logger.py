import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

class LoggerFactory:
    """日志工厂类"""

    # 缓存已创建的日志记录器，避免重复创建
    # 不需要主动关闭Logger、Handler，程序结束后自动释放
    _loggers = {}

    @classmethod
    def get_logger(
            cls,       # 表示当前类对象，可以调用类属性
            name: str = "app",
            log_file: Optional[str] = None,
            level: str = "INFO",
            max_bytes: int = 10 * 1024 * 1024,  # 10MB
            backup_count: int = 5,
            fmt: Optional[str] = None,
            console: bool = True
    ) -> logging.Logger:
        """
        获取或创建一个日志记录器
        
        Args:
            name: 日志记录器名称
            log_file: 日志文件路径
            level: 日志级别。
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的备份日志文件数量
            fmt: 日志格式
            console: 是否输出到控制台
            
        Returns:
            logging.Logger: 配置好的日志记录器
        """

        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))

        # 避免重复添加handler
        if logger.handlers:
            cls._loggers[name] = logger
            return logger

        # 默认格式
        if fmt is None:
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

        formatter = logging.Formatter(fmt)

        # 控制台输出
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # 文件输出
        if log_file:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        cls._loggers[name] = logger
        return logger


if __name__ == "__main__":
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)

    # 创建日志记录器
    logger = LoggerFactory.get_logger(
        name="my_app",
        log_file="logs/my_app.log",
        level="DEBUG",
        max_bytes=2,   # 2B就轮转，测试
        backup_count=3,
        fmt="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    logger.info("Hello, World!")
    logger.info('If delay is true')