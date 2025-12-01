CREATE TABLE IF NOT EXISTS monitor_model_metrics(
    id INT NOT NULL AUTO_INCREMENT,
    model VARCHAR(50) NOT NULL COMMENT '模型名称, 如 lightgbm_alpha',
    day DATE NOT NULL COMMENT '训练日期，格式YYYY-MM-DD',
    instruments VARCHAR(50) NOT NULL COMMENT '股票集合，如 csi300',
    horizon INT NOT NULL COMMENT '预测时长，如 1,2,5',
    IC Decimal(10, 6) NOT NULL COMMENT 'IC',
    ICIR Decimal(10, 6) NOT NULL COMMENT 'ICIR',
    RIC Decimal(10, 6) NOT NULL COMMENT 'RIC',
    RICIR Decimal(10, 6) NOT NULL COMMENT 'RICIR',
    PRIMARY KEY (id),
    UNIQUE KEY un_code_day (model, day, horizon)
) ENGINE=InnoDB
    DEFAULT CHARSET=utf8mb4
    COLLATE=utf8mb4_unicode_ci
    COMMENT='模型训练指标'
    AUTO_INCREMENT=1;

