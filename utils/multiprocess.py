"""
    并行计算
"""

from typing import Callable, Iterable
import multiprocessing as mp


def multiprocessing_wrapper(func_calls: list[tuple[Callable, tuple]], n: int = 1) -> list:
    """ 独立函数/参数并行计算

    Parameters
    ----------
    func_calls : List[Tuple[Callable, Tuple]]
        the list of functions and their parameters
        如：[(np.sum, (2,3,))]
    n : int
        the number of subprocesses

    Returns
    -------
    list

    """
    if n == 1 or max(1, min(n, len(func_calls))) == 1:
        return [f(*args) for f, args in func_calls]

    with mp.Pool(processes=max(1, min(n, len(func_calls)))) as pool:
        results = [
            pool.apply_async(f, args) for f, args in func_calls
        ]
        # AsyncResult.get() 是阻塞方法
        return [result.get() for result in results]


def multiprocessing_wrapper_same(func: Callable, args: Iterable, n: int = 1) -> list:
    """ 相同函数，不同参数并行计算
    
    Parameters
    ----------
    func : Callable
        统一的函数（必须在模块级别定义，不能是 lambda 或嵌套函数）
    args : Iterable
        参数集合，每个元素会传递给 func
        如：[1, 2, 3] 或 [(a1, b1), (a2, b2)]
    n : int
        进程数
    
    Returns
    -------
    list
        计算结果列表（注意：使用 imap_unordered，不保证顺序）
    
    Examples
    --------
    >>> def square(x): return x ** 2
    >>> multiprocessing_wrapper_same(square, [1, 2, 3], n=2)
    [1, 4, 9]
    
    >>> def add(args): return args[0] + args[1]
    >>> multiprocessing_wrapper_same(add, [(1, 2), (3, 4)], n=2)
    [3, 7]
    """
    if n == 1:
        return [func(arg) for arg in args]
    
    with mp.Pool(processes=n) as pool:
        results = []
        for res in pool.imap_unordered(func, args):
            results.append(res)
    
    return results

