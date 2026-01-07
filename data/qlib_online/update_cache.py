"""cache_updater.py - Qlib 缓存定时更新工具（支持重试和并行）"""

import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import List, Tuple

import fire
import numpy as np
import qlib
from qlib.config import C
from qlib.constant import REG_CN
from qlib.data.cache import CacheUtils
from qlib.data.cache import DiskExpressionCache, DiskDatasetCache
from qlib.data.data import Cal, ExpressionD, Inst
from qlib.data.data import LocalExpressionProvider, LocalDatasetProvider

from utils import LoggerFactory
from utils import get_config
from utils import send_email
# from utils.qlib_ops import CMean, CStd  # 导入自定义算子


class DiskExpressionCache2(DiskExpressionCache):
    def __init__(self, provider, **kwargs):
        super().__init__(provider, **kwargs)

    def update(self, sid, cache_uri, freq: str = "day"):
        cp_cache_uri = self.get_cache_dir(freq).joinpath(sid).joinpath(cache_uri)
        meta_path = cp_cache_uri.with_suffix(".meta")
        if not self.check_cache_exists(cp_cache_uri, suffix_list=[".meta"]):
            self.logger.info(f"The cache {cp_cache_uri} has corrupted. It will be removed")
            self.clear_cache(cp_cache_uri)
            return 2

        with CacheUtils.writer_lock(self.r, f"{str(C.dpm.get_data_uri())}:expression-{cache_uri}"):
            with meta_path.open("rb") as f:
                d = pickle.load(f)
            instrument = d["info"]["instrument"]
            field = d["info"]["field"]
            freq = d["info"]["freq"]
            last_update_time = d["info"]["last_update"]

            # get newest calendar
            # from .data import Cal, ExpressionD  # pylint: disable=C0415

            whole_calendar = Cal.calendar(start_time=None, end_time=None, freq=freq)
            # calendar since last updated.
            new_calendar = Cal.calendar(start_time=last_update_time, end_time=None, freq=freq)

            # get append data
            if len(new_calendar) <= 1:
                # Including last updated calendar, we only get 1 item.
                # No future updating is needed.
                return 1
            else:
                # get the data needed after the historical data are removed.
                # The start index of new data
                current_index = len(whole_calendar) - len(new_calendar) + 1

                # The existing data length
                size_bytes = os.path.getsize(cp_cache_uri)
                ele_size = np.dtype("<f").itemsize
                assert size_bytes % ele_size == 0
                ele_n = size_bytes // ele_size - 1

                expr = ExpressionD.get_expression_instance(field)
                lft_etd, rght_etd = expr.get_extended_window_size()
                # The expression used the future data after rght_etd days.
                # So the last rght_etd data should be removed.
                # There are most `ele_n` period of data can be remove
                remove_n = min(rght_etd, ele_n)
                assert new_calendar[1] == whole_calendar[current_index]

                # ---- Fix -----
                # 二进制文件的第一个元素
                with open(cp_cache_uri, "rb") as f:
                    ref_start_index = int(np.frombuffer(f.read(4), dtype="<f")[0])
                expected_start_idx = ref_start_index + ele_n - remove_n
                query_left_shift = remove_n + (current_index - expected_start_idx)
                query_left_shift = min(query_left_shift, ele_n)

                data = self.provider.expression(
                    instrument, field, whole_calendar[current_index - query_left_shift], new_calendar[-1], freq
                )

                data = np.array(data).astype("<f")
                # 删除尾部为nan的元素
                if len(data) > 0 and np.isnan(data[-1]):
                    non_nan_mask = ~np.isnan(data)
                    if np.any(non_nan_mask):
                        last_valid_idx = np.where(non_nan_mask)[0][-1]
                        data = data[:last_valid_idx + 1]
                    else:
                        data = np.array([], dtype="<f")

                with open(cp_cache_uri, "ab") as f:
                    # data = np.array(data).astype("<f")
                    # Remove the last bits
                    f.truncate(size_bytes - ele_size * remove_n)
                    f.write(data)
                # ---------------
                # update meta file
                d["info"]["last_update"] = str(new_calendar[-1])
                with meta_path.open("wb") as f:
                    pickle.dump(d, f, protocol=C.dump_protocol_version)
        return 0

class DiskDatasetCache2(DiskDatasetCache):
    def __init__(self, provider, **kwargs):
        super().__init__(provider, **kwargs)

    def update(self, cache_uri, freq: str = "day"):
        """
        Update dataset cache to latest calendar.

        最小改动版：
        - 不再做复杂的增量追加，而是在需要更新时直接整体重建该 cache_uri 对应的数据集缓存。
        - 好处：避免由于历史尾部停牌等原因造成“旧数据段”和“新增数据段”之间日期/NaN 对齐不一致的问题。
        """
        cp_cache_uri = self.get_cache_dir(freq).joinpath(cache_uri)
        meta_path = cp_cache_uri.with_suffix(".meta")
        if not self.check_cache_exists(cp_cache_uri):
            self.logger.info(f"The cache {cp_cache_uri} has corrupted. It will be removed")
            self.clear_cache(cp_cache_uri)
            return 2

        im = DiskDatasetCache.IndexManager(cp_cache_uri)
        with CacheUtils.writer_lock(self.r, f"{str(C.dpm.get_data_uri())}:dataset-{cache_uri}"):
            # 读取原 meta，拿到 instruments / fields / freq / inst_processors
            with meta_path.open("rb") as f:
                d = pickle.load(f)
            instruments = d["info"]["instruments"]
            fields = d["info"]["fields"]
            freq = d["info"]["freq"]
            inst_processors = d["info"].get("inst_processors", [])

            self.logger.debug("Rebuilding dataset cache: {}".format(d))

            # 与原逻辑保持一致：字典型 instruments 暂不支持更新

            if Inst.get_inst_type(instruments) == Inst.DICT:
                self.logger.info(f"The file {cache_uri} has dict cache. Skip updating")
                return 1

            # 直接整体重建：调用已有的 gen_dataset_cache
            # - gen_dataset_cache 内部会：
            #   * 使用完整交易日历 Cal.calendar(freq=freq)
            #   * 调用 provider.dataset(...) 生成数据
            #   * 写入 .data / .meta / .index
            features = self.gen_dataset_cache(
                cache_path=cp_cache_uri,
                instruments=instruments,
                fields=fields,
                freq=freq,
                inst_processors=inst_processors,
            )

            # 如果没有数据可用，认为无增量可更新
            if features.empty:
                return 1

            # gen_dataset_cache 已经更新了 meta 中的 last_update 等信息，这里只返回成功
            return 0


@dataclass
class UpdateResult:
    """更新结果记录"""
    cache_type: str  # "expression" or "dataset"
    cache_id: str  # cache identifier
    status: int  # 0=success, 1=no_need, 2=failed
    error_msg: str = ""  # 错误信息
    retry_count: int = 0  # 重试次数


@dataclass
class UpdateSummary:
    """更新汇总"""
    success: List[UpdateResult] = field(default_factory=list)
    no_need: List[UpdateResult] = field(default_factory=list)
    failed: List[UpdateResult] = field(default_factory=list)

    def add(self, result: UpdateResult):
        if result.status == 0:
            self.success.append(result)
        elif result.status == 1:
            self.no_need.append(result)
        else:
            self.failed.append(result)

    def print_summary(self):
        print("\n" + "=" * 60)
        print("                   更新汇总报告")
        print("=" * 60)
        print(f"✅ 更新成功: {len(self.success)}")
        print(f"⏭️ 无需更新: {len(self.no_need)}")
        print(f"❌ 更新失败: {len(self.failed)}")

        if self.failed:

            print("\n" + "-" * 60)
            print("失败详情:")
            print("-" * 60)
            for r in self.failed:
                print(f"  [{r.cache_type}] {r.cache_id}")
                print(f"    重试次数: {r.retry_count}")
                print(f"    错误信息: {r.error_msg}")
                print()


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器 工厂函数"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    result, error = func(*args, **kwargs)
                    if result != 2:  # 非失败状态直接返回
                        return result, error, attempt
                    last_error = error
                except Exception as e:
                    last_error = str(e)

                if attempt < max_retries:
                    time.sleep(delay * (attempt + 1))  # 递增延迟

            return 2, last_error, max_retries

        return wrapper

    return decorator


class CacheUpdater:
    """缓存更新器"""

    def __init__(self,
                 provider_uri: str = "~/.qlib/qlib_data/custom_data_hfq",
                 max_retries: int = 3,
                 max_workers: int = 4,
                 early_stop_failures: int = 5):
        self.provider_uri = provider_uri
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.early_stop_failures = early_stop_failures  # 达到这个失败数就提前停止
        self.summary = UpdateSummary()
        self.logger = LoggerFactory.get_logger(__name__)

        # 初始化 qlib
        config = get_config()
        qlib.init(
            default_conf="server",
            region=REG_CN,
            redis_host=config['redis']['host'],
            redis_port=config['redis']['port'],
            redis_task_db=3,
            redis_password=config['redis']['password'],
            provider_uri=provider_uri,
            # custom_ops=[CMean, CStd]  # 注册自定义算子
        )

        # 创建缓存实例
        self.expr_provider = LocalExpressionProvider()
        self.expr_cache = DiskExpressionCache2(provider=self.expr_provider)

        self.dataset_provider = LocalDatasetProvider()
        self.dataset_cache = DiskDatasetCache2(provider=self.dataset_provider)

    @retry_on_failure(max_retries=3, delay=1.0)
    def _update_expression_cache(self, sid: str, cache_uri: str, freq: str) -> Tuple[int, str]:
        """更新单个表达式缓存"""
        try:
            result = self.expr_cache.update(sid, cache_uri, freq)
            return result, ""
        except Exception as e:
            return 2, str(e)

    @retry_on_failure(max_retries=3, delay=1.0)
    def _update_dataset_cache(self, cache_uri: str, freq: str) -> Tuple[int, str]:
        """更新单个数据集缓存"""
        try:
            result = self.dataset_cache.update(cache_uri, freq)
            return result, ""
        except Exception as e:
            return 2, str(e)

    def _get_expression_cache_tasks(self, freq: str) -> List[Tuple[str, str]]:
        """获取所有表达式缓存任务"""
        # 汇总 <sid, cache_file> 集合
        tasks = []
        feature_cache_dir = Path(C.dpm.get_data_uri(freq)) / C.features_cache_dir_name

        if feature_cache_dir.exists():
            for instrument_dir in feature_cache_dir.iterdir():
                if instrument_dir.is_dir():
                    # sz000001
                    sid = instrument_dir.name
                    # field对应的hash值，如 fc8ffd431c5a48ebdcabac43a92d3edb
                    for cache_file in instrument_dir.iterdir():
                        if cache_file.suffix == "" and cache_file.is_file():
                            tasks.append((sid, cache_file.name))
        return tasks

    def _get_dataset_cache_tasks(self, freq: str) -> List[str]:
        """获取所有数据集缓存任务"""
        tasks = []
        dataset_cache_dir = Path(C.dpm.get_data_uri(freq)) / C.dataset_cache_dir_name

        if dataset_cache_dir.exists():
            for cache_file in dataset_cache_dir.iterdir():
                if cache_file.suffix == "" and cache_file.is_file():
                    tasks.append(cache_file.name)
        return tasks

    def update_expression_caches_parallel(self, freq: str = "day"):
        """并行更新所有表达式缓存"""
        tasks = self._get_expression_cache_tasks(freq)
        self.logger.info(f"Found {len(tasks)} expression caches to update")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._update_expression_cache, sid, cache_uri, freq): (sid, cache_uri)
                for sid, cache_uri in tasks
            }

            for future in as_completed(futures):
                # 检查是否已经达到失败阈值
                if len(self.summary.failed) >= self.early_stop_failures:
                    print(f"\n⚠️  检测到 {len(self.summary.failed)} 个失败，提前停止...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                sid, cache_uri = futures[future]
                cache_id = f"{sid}/{cache_uri}"

                try:
                    status, error, retry_count = future.result()
                    result = UpdateResult(
                        cache_type="expression",
                        cache_id=cache_id,
                        status=status,
                        error_msg=error,
                        retry_count=retry_count
                    )
                    self.summary.add(result)

                    status_icon = {0: "✅", 1: "⏭️", 2: "❌"}.get(status, "?")
                    if status == 2:
                        self.logger.error(f"  {status_icon} Expression: {cache_id} - {error}")
                    else:
                        self.logger.info(f"  {status_icon} Expression: {cache_id}")

                except Exception as e:
                    result = UpdateResult(
                        cache_type="expression",
                        cache_id=cache_id,
                        status=2,
                        error_msg=str(e),
                        retry_count=self.max_retries
                    )
                    self.summary.add(result)
                    self.logger.error(f"  ❌ Expression: {cache_id} - {e}")

    def update_dataset_caches_parallel(self, freq: str = "day"):
        """并行更新所有数据集缓存"""
        tasks = self._get_dataset_cache_tasks(freq)
        self.logger.info(f"Found {len(tasks)} dataset caches to update")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._update_dataset_cache, cache_uri, freq): cache_uri
                for cache_uri in tasks
            }

            for future in as_completed(futures):
                # 检查是否已经达到失败阈值
                if len(self.summary.failed) >= self.early_stop_failures:
                    self.logger.error(f"\n⚠️  检测到 {len(self.summary.failed)} 个失败，提前停止...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                cache_uri = futures[future]

                try:
                    status, error, retry_count = future.result()
                    result = UpdateResult(
                        cache_type="dataset",
                        cache_id=cache_uri,
                        status=status,
                        error_msg=error,
                        retry_count=retry_count
                    )
                    self.summary.add(result)

                    status_icon = {0: "✅", 1: "⏭️", 2: "❌"}.get(status, "?")
                    if status == 2:
                        self.logger.error(f"  {status_icon} Dataset: {cache_uri} - {error}")
                    else:
                        self.logger.info(f"  {status_icon} Dataset: {cache_uri}")

                except Exception as e:
                    result = UpdateResult(
                        cache_type="dataset",
                        cache_id=cache_uri,
                        status=2,
                        error_msg=str(e),
                        retry_count=self.max_retries
                    )
                    self.summary.add(result)
                    self.logger.error(f"  ❌ Dataset: {cache_uri} - {e}")

    def update_all(self, freq: str = "day"):
        """更新所有缓存"""
        t = time.time()
        self.logger.info(f"Starting cache update (freq={freq}, workers={self.max_workers})")
        self.logger.info("-" * 60)

        self.logger.info("\n[1/2] Updating Expression Caches...")
        self.update_expression_caches_parallel(freq)

        if not self.summary.failed:
            self.logger.info("\n[2/2] Updating Dataset Caches...")
            self.update_dataset_caches_parallel(freq)

        # 打印汇总
        self.summary.print_summary()

        if self.summary.failed:
            send_email('Data:qlib_online:update_cache', 'Error, please review the log')
        self.logger.info('耗时: {}s'.format(round(time.time() - t, 4)))


if __name__ == "__main__":
    fire.Fire(CacheUpdater)
    '''
    python update_cache.py update_all --max_workers 6 --early_stop_failures 1
    '''