"""
    并行计算
"""

from typing import Callable
import multiprocessing as mp


def multiprocessing_wrapper(func_calls: list[tuple[Callable, tuple]], n: int = 1) -> list:
    """It will use multiprocessing to call the functions in func_calls with the given parameters.
    The results equals to `return  [f(*args) for f, args in func_calls]`
    It will not call multiprocessing if `n=1`

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




