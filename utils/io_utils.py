"""
读写工具
"""
import os
import os.path
import pickle
import shutil
from pathlib import Path
from typing import Any, Union

import pandas as pd

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

    @staticmethod
    def update(obj: Any, path: PathLike) -> str:
        if not os.path.exists(str(path)):
            return PickleIO.write(obj, path)
        path_tmp = str(path) + '.tmp'
        PickleIO.write(obj, path_tmp)
        os.replace(path_tmp, path)
        return str(path)


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

    @staticmethod
    def update(df: pd.DataFrame, path: PathLike, type: str = 'parquet') -> str:
        if not os.path.exists(str(path)):
            return DataFrameIO.write(df, path, type)
        path_tmp = str(path) + '.tmp'
        DataFrameIO.write(df, path_tmp , type)
        os.replace(path_tmp, path)
        return str(path)


def overwrite(src: PathLike, dst: PathLike, keep_src: bool = False):
    """
    通用安全覆盖函数，支持覆盖文件 / 文件夹
    逻辑：目标存在则先彻底删除，再替换源到目标

    :param src: 源路径（文件/文件夹，必须存在）
    :param dst: 目标路径（存在会被完全覆盖）
    :param keep_src: 是否保留源数据。True 为复制，False 为移动（默认）
    """
    try:
        # 校验源必须存在
        if not os.path.exists(src):
            raise FileNotFoundError(f"源不存在: {src}")

        # 目标存在，区分文件/目录分别删除
        if os.path.exists(dst):
            if os.path.isdir(dst):
                # 目标是文件夹，递归删除全部内容
                shutil.rmtree(dst)
            else:
                # 目标是普通文件，直接删除
                os.unlink(dst)

        if keep_src:
            # 复制：保留源数据
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        else:
            # 移动：原子重命名/替换，文件、目录都兼容
            os.replace(src, dst)
        print(f"操作成功：{src} → {dst}")
        return dst
    except FileNotFoundError as e:
        print(f"错误：{e}")
    except PermissionError:
        print(f"错误：权限不足，无法操作 {dst}")
    except OSError as e:
        print(f"系统操作失败：{e}")
    except Exception as e:
        print(f"未知异常：{type(e).__name__}: {e}")

