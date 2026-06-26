-- 股票停复牌数据
CREATE TABLE IF NOT EXISTS suspend_d(
    id INT NOT NULL AUTO_INCREMENT,
    ts_code VARCHAR(10) NOT NULL COMMENT 'TS股票代码，股票代码带BJ/SH/SZ后缀, 如 000001.SZ',
    qlib_code VARCHAR(10) NOT NULL COMMENT 'Qlib股票代码, 如 SZ000001',
    day DATE NOT NULL COMMENT '取数日期，格式YYYY-MM-DD',
    suspend_timing VARCHAR(50) COMMENT '日内停牌时间段. 如 9:32-9:42,9:44-9:54',
    suspend_type VARCHAR(1) NOT NULL COMMENT '停复牌类型：S-停牌，R-复牌',
    PRIMARY KEY (id),
    INDEX day_idx (day)
) ENGINE=InnoDB
    DEFAULT CHARSET=utf8mb4
    COLLATE=utf8mb4_unicode_ci
    COMMENT='Tushare 股票停牌数据'
    AUTO_INCREMENT=1;

