-- 上海银行间同业拆放利率（Shanghai Interbank Offered Rate，简称Shibor）
-- 利率指标经过年化处理. 在计算日频超额收益时，需要先除以 252 进行日频化
CREATE TABLE IF NOT EXISTS shibor(
    id INT NOT NULL AUTO_INCREMENT,
    `date` DATE NOT NULL COMMENT '日期，格式YYYY-MM-DD',
    on_rate DECIMAL(10, 4) NOT NULL COMMENT '隔夜利率(%)',
    `1w` DECIMAL(10, 4) NOT NULL COMMENT '1周利率(%)',
    `2w` DECIMAL(10, 4) NOT NULL COMMENT '2周利率(%)',
    `1m` DECIMAL(10, 4) NOT NULL COMMENT '1个月利率(%)',
    `3m` DECIMAL(10, 4) NOT NULL COMMENT '3个月利率(%)',
    `6m` DECIMAL(10, 4) NOT NULL COMMENT '6个月利率(%)',
    `9m` DECIMAL(10, 4) NOT NULL COMMENT '9个月利率(%)',
    `1y` DECIMAL(10, 4) NOT NULL COMMENT '1年利率(%)',
    PRIMARY KEY (id),
    UNIQUE KEY un_day (date)
) ENGINE=InnoDB
    DEFAULT CHARSET=utf8mb4
    COLLATE=utf8mb4_unicode_ci
    COMMENT='上海银行间同业拆放利率（Shanghai Interbank Offered Rate，简称Shibor）'
    AUTO_INCREMENT=1;



