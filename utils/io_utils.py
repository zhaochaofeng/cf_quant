"""
读写工具
"""

import pickle
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


class PickleIO:
    """Pickle 文件读写工具类"""

    @staticmethod
    def write(obj: Any, path: PathLike) -> None:
        """将对象序列化到 pickle 文件

        Args:
            obj: 要序列化的 Python 对象
            path: 文件路径，自动创建父目录
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

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
