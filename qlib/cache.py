""" DiskDatasetCache """
@classmethod
def read_data_from_cache(cls, cache_path: Union[str, Path], start_time, end_time, fields):
    """read_cache_from

    This function can read data from the disk cache dataset

    :param cache_path:
    :param start_time:
    :param end_time:
    :param fields: The fields order of the dataset cache is sorted. So rearrange the columns to make it consistent.
    :return:
    """

    im = DiskDatasetCache.IndexManager(cache_path)
    index_data = im.get_index(start_time, end_time)
    if index_data.shape[0] > 0:
        start, stop = (
            index_data["start"].iloc[0].item(),
            index_data["end"].iloc[-1].item(),
        )
    else:
        start = stop = 0

    with pd.HDFStore(cache_path, mode="r") as store:
        if "/{}".format(im.KEY) in store.keys():
            df = store.select(key=im.KEY, start=start, stop=stop)
            df = df.swaplevel("datetime", "instrument").sort_index()
            # read cache and need to replace not-space fields to field
            df = cls.cache_to_origin_data(df, fields)
        else:
            df = pd.DataFrame(columns=fields)

    # ===================== 补丁开始：按日历补齐 NaN 日期 =====================
    # 目标：在 [start_time, end_time] 的交易日范围内，
    #       每个 (datetime, instrument) 组合都有一行；
    #       没有成交数据的日期保留为 NaN，而不是整行被省略。
    if not df.empty and start_time is not None and end_time is not None:
        meta_path = Path(cache_path).with_suffix(".meta")
        freq = None
        if meta_path.exists():
            try:
                with meta_path.open("rb") as f_meta:
                    meta = pickle.load(f_meta)
                    freq = meta.get("info", {}).get("freq", None)
            except Exception:
                freq = None

        if freq is not None:
            # 延迟导入，保持原文件风格
            from .data import Cal  # pylint: disable=C0415

            # 完整交易日日历
            full_calendar = Cal.calendar(start_time=start_time, end_time=end_time, freq=freq)
            if len(full_calendar) > 0:
                # 该缓存中涉及到的股票列表
                instruments = df.index.get_level_values("instrument").unique()
                # 构造完整 MultiIndex：每个 (交易日, 股票) 组合都保留
                full_index = pd.MultiIndex.from_product(
                    [full_calendar, instruments], names=["datetime", "instrument"]
                )
                # 以完整索引重建 DataFrame，对缺失组合自动填 NaN
                df = df.reindex(full_index)
    # ===================== 补丁结束 =====================

    return df

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
        from .data import Inst  # pylint: disable=C0415

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


""" DiskExpressionCache """
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
            expected_start_idx = ref_start_index + ele_n
            query_left_shift = remove_n + (current_index - expected_start_idx)
            query_left_shift = min(query_left_shift, ele_n)

            data = self.provider.expression(
                instrument, field, whole_calendar[current_index - query_left_shift], new_calendar[-1], freq
            )

            data = np.array(data).astype("<f")
            # 删除尾部为nan的元素. 未来数据指标（如label）的NaN保存
            if rght_etd <= 0:
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


