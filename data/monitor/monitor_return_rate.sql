
CREATE TABLE IF NOT EXISTS monitor_return_rate(
    id INT NOT NULL AUTO_INCREMENT,
    qlib_code VARCHAR(10) NOT NULL COMMENT 'Qlib股票代码, 如 SZ000001',
    day DATE NOT NULL COMMENT '取数日期，格式YYYY-MM-DD',
    score Decimal(10, 6) NOT NULL COMMENT '预测得分',
    PRIMARY KEY (id),
    UNIQUE KEY un_code_day (qlib_code, day)
) ENGINE=InnoDB
    DEFAULT CHARSET=utf8mb4
    COLLATE=utf8mb4_unicode_ci
    COMMENT='预测收益率'
    AUTO_INCREMENT=1;



