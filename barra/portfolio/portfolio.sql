
-- 持仓信息

CREATE TABLE IF NOT EXISTS portfolio (
    id INT NOT NULL AUTO_INCREMENT,
    day DATE NOT NULL COMMENT '计算日期，格式YYYY-MM-DD',
    portfolio VARCHAR(50) NOT NULL COMMENT '持仓组合名称',
    qlib_code VARCHAR(10) NOT NULL COMMENT 'Qlib股票代码, 如 SZ000001',
    active_weight DECIMAL(10, 6) NOT NULL COMMENT '主动权重',
    total_weight DECIMAL(10, 6) NOT NULL COMMENT '总持仓权重',
    shares BIGINT NOT NULL COMMENT '持仓股数',
    market_value DECIMAL(16, 2) NOT NULL COMMENT '持仓市值（元）',
    weight_pct DECIMAL(8, 4) NOT NULL COMMENT '总权重百分比',
    PRIMARY KEY (id),
    UNIQUE KEY un_day_portfolio_code (day, portfolio, qlib_code)
) ENGINE=InnoDB
    DEFAULT CHARSET=utf8mb4
    COLLATE=utf8mb4_unicode_ci
    COMMENT='持仓信息表'
    AUTO_INCREMENT=1;

