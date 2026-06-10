"""
读写工具
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


class PickleIO:
    """Pickle 文件读写工具类"""

    @staticmethod
    def write(obj: Any, path: PathLike) -> str:
        """将对象序列化到 pickle 文件

        Args:
            obj: 要序列化的 Python 对象
            path: 文件路径，自动创建父目录
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        return str(path)

    @staticmethod
    def read(path: PathLike) -> Any:
        """从 pickle 文件反序列化对象

        Args:
            path: 文件路径

        Returns:
            反序列化后的 Python 对象
        """
        path = Path(path)
        with open(path, 'rb') as f:
            return pickle.load(f)


class DataFrameIO:
    """通用 DataFrame 读写工具类，支持 csv 和 parquet 格式。
    """

    @staticmethod
    def write(df: pd.DataFrame, path: PathLike, type: str = 'parquet') -> str:
        """保存 DataFrame 到 csv 或 parquet 文件。

        Args:
            df: 要保存的 DataFrame。
            path: 文件路径，自动创建父目录。
            type: 文件类型，'csv' 或 'parquet'。

        Returns:
            保存的文件路径字符串。

        Raises:
            ValueError: 不支持的 type 参数。
        """
        if type not in ('csv', 'parquet'):
            raise ValueError(f"Unsupported type '{type}'. Supported types: 'csv', 'parquet'")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if type == 'csv':
            df.to_csv(path, encoding='utf-8')
        else:
            df.to_parquet(path)

        return str(path)

    @staticmethod
    def read(path: PathLike, type: str = 'parquet') -> 'pd.DataFrame | None':
        """从 csv 或 parquet 文件加载 DataFrame。

        Args:
            path: 文件路径。
            type: 文件类型，'csv' 或 'parquet'。

        Returns:
            加载的 DataFrame，文件不存在时返回 None。

        Raises:
            ValueError: 不支持的 type 参数。
        """
        if type not in ('csv', 'parquet'):
            raise ValueError(f"Unsupported type '{type}'. Supported types: 'csv', 'parquet'")

        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path not found: '{path}'")

        if type == 'csv':
            return pd.read_csv(path, encoding='utf-8')
        else:
            return pd.read_parquet(path)


