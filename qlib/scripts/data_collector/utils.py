def get_calendar_list(bench_code="CSI300") -> List[pd.Timestamp]:
    """get SH/SZ history calendar list

    Parameters
    ----------
    bench_code: str
        value from ["CSI300", "CSI500", "ALL", "US_ALL"]

    Returns
    -------
        history calendar list
    """

    logger.info(f"get calendar list: {bench_code}......")

    def _get_calendar_from_eastmoney(url, max_retry=2, retry_sleep=3):
        """从东方财富获取交易日历"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": "http://quote.eastmoney.com/",
        }
        for i in range(1, max_retry + 1):
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                _value_list = resp.json()["data"]["klines"]
                return sorted(map(lambda x: pd.Timestamp(x.split(",")[0]), _value_list))
            except Exception as e:
                logger.warning(f"_get_calendar_from_eastmoney attempt {i}/{max_retry} failed: {e}")
                if i == max_retry:
                    return None
                time.sleep(retry_sleep)
        return None

    def _get_calendar_from_baostock(start_date="2000-01-01", end_date=None):
        """从baostock获取交易日历（备用方案）"""
        import baostock as bs
        if end_date is None:
            end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        try:
            lg = bs.login()
            rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
            data_list = []
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                if row[1] == "1":  # is_trading_day == 1
                    data_list.append(pd.Timestamp(row[0]))
            bs.logout()
            return sorted(data_list)
        except Exception as e:
            logger.warning(f"_get_calendar_from_baostock failed: {e}")
            return None

    def _get_calendar(url):
        """获取交易日历，优先东方财富，失败则使用baostock备用"""
        # 先尝试东方财富
        calendar = _get_calendar_from_eastmoney(url)
        if calendar:
            return calendar
        # 备用：使用baostock
        logger.info("东方财富API不可用，切换到baostock获取交易日历...")
        calendar = _get_calendar_from_baostock()
        if calendar:
            return calendar
        raise ValueError("无法获取交易日历：东方财富和baostock均失败")

    calendar = _CALENDAR_MAP.get(bench_code, None)
    if calendar is None:
        if bench_code.startswith("US_") or bench_code.startswith("IN_") or bench_code.startswith("BR_"):
            print(Ticker(CALENDAR_BENCH_URL_MAP[bench_code]))
            print(Ticker(CALENDAR_BENCH_URL_MAP[bench_code]).history(interval="1d", period="max"))
            df = Ticker(CALENDAR_BENCH_URL_MAP[bench_code]).history(interval="1d", period="max")
            calendar = df.index.get_level_values(level="date").map(pd.Timestamp).unique().tolist()
        else:
            if bench_code.upper() == "ALL":

                @deco_retry
                def _get_calendar(month):
                    _cal = []
                    try:
                        resp = requests.get(
                            SZSE_CALENDAR_URL.format(month=month, random=random.random), timeout=None
                        ).json()
                        for _r in resp["data"]:
                            if int(_r["jybz"]):
                                _cal.append(pd.Timestamp(_r["jyrq"]))
                    except Exception as e:
                        raise ValueError(f"{month}-->{e}") from e
                    return _cal

                month_range = pd.date_range(start="2000-01", end=pd.Timestamp.now() + pd.Timedelta(days=31), freq="M")
                calendar = []
                for _m in month_range:
                    cal = _get_calendar(_m.strftime("%Y-%m"))
                    if cal:
                        calendar += cal
                calendar = list(filter(lambda x: x <= pd.Timestamp.now(), calendar))
            else:
                calendar = _get_calendar(CALENDAR_BENCH_URL_MAP[bench_code])
        _CALENDAR_MAP[bench_code] = calendar
    logger.info(f"end of get calendar list: {bench_code}.")
    return calendar



