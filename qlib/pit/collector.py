class PitNormalize(BaseNormalize):
    def __init__(self, interval: str = "quarterly", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interval = interval

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # dt的作用是当date为缺失值时，用period的调整值来作为估计
        dt = df["period"].apply(
            lambda x: (
                pd.to_datetime(x) + pd.DateOffset(days=(45 if self.interval == PitCollector.INTERVAL_QUARTERLY else 90))
            ).date()
        )
        df["date"] = df["date"].fillna(dt.astype(str))

        df["period"] = pd.to_datetime(df["period"])
        df["period"] = df["period"].apply(
            lambda x: x.year if self.interval == PitCollector.INTERVAL_ANNUAL else x.year * 100 + (x.month - 1) // 3 + 1
        )
        return df

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # return get_calendar_list()
        return []


