-- 持仓组合alpha预测值

CREATE TABLE IF NOT EXISTS alpha (
    id INT NOT NULL AUTO_INCREMENT,
    day DATE NOT NULL COMMENT '计算日期，格式YYYY-MM-DD',
    portfolio VARCHAR(50) NOT NULL COMMENT '持仓组合名称',
    alpha DECIMAL(10, 6) NOT NULL COMMENT 'alpha预测值',
    PRIMARY KEY (id),
    UNIQUE KEY un_day (day)
) ENGINE=InnoDB
    DEFAULT CHARSET=utf8mb4
    COLLATE=utf8mb4_unicode_ci
    COMMENT='持仓组合alpha预测值'
    AUTO_INCREMENT=1;
