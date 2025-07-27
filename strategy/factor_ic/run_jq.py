'''
    聚宽回测代码
'''

from statsmodels.multivariate.factor import Factor as sFactor
import pandas as pd
import numpy as np
import redis
from jqfactor import Factor, calc_factors, analyze_factor
from jqfactor import standardlize, winsorize
from datetime import datetime, timedelta

def get_stocks(dt):
    ''' 股票候选 '''
    # 获取所有股票
    stocks = get_all_securities(types=['stock'], date=dt)
    log.info('总股票数: {}{}'.format(len(stocks), '-' * 50))
    # 排除ST
    stocks = get_extras('is_st', security_list=list(stocks.index),
                        start_date=dt, end_date=dt, df=True)
    stocks = stocks.columns[stocks.eq(False).any()].tolist()
    log.info('排除ST后股票数：{}'.format(len(stocks)))

    stocks_2 = []
    for stock in stocks:
        info = get_security_info(code=stock, date=dt)
        if '退' in info.display_name:
            continue
        stocks_2.append(stock)
    log.info('排除退市股票后：{}'.format(len(stocks_2)))

    # 仅保留主板股票。代码以60/00开头
    stocks_2 = [stock for stock in stocks_2 if stock[0:2] in ['60', '00']]
    log.info('沪深主板股票数：{}'.format(len(stocks_2)))
    return stocks_2

def initialize(context):
    log.info('{}\n函数运行时间（initialize）:{}'.format('-' * 50, context.current_dt.time()))
    # 设置沪深300作为基准
    set_benchmark('000300.XSHG')
    # 设置动态复权(动态前复权)
    set_option('use_real_price', True)
    # 交易成交比例
    set_option('order_volume_ratio', 1.0)
    # 设置佣金/印花税
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0.001,
        open_commission=0.0003, close_commission=0.0003,
        close_today_commission=0, min_commission=5
    ), type='stock')

    # 最大持仓数
    g.stock_num = 20
    # 选股。开盘前运行
    run_weekly(choose_stock, weekday=1, time='before_open', reference_security='000300.XSHG')
    # 交易。开盘时运行
    run_weekly(trade, weekday=1, time='open', reference_security='000300.XSHG')
    # 收盘后运行
    run_weekly(after_market_close, weekday=1, time='after_close', reference_security='000300.XSHG')

def choose_stock(context):
    ''' 选股函数 '''
    log.info('{}\n函数运行时间（choose_stock）:{}'.format('-' * 50, context.current_dt.time()))
    current_dt = context.current_dt

    stocks = get_stocks(current_dt)
    # 股票编号与聚宽股票代码映射关系。如{'000001', '000001.XSHE'}
    stocks_dic = {stock.split('.')[0]: stock for stock in stocks}
    log.info('stocks_dic len: {}'.format(len(stocks_dic)))

    # 从redis中获取股票得分
    r = redis.Redis(host='39.105.18.127', password='Zhao_38013984')
    dt = context.current_dt.strftime('%Y-%m-%d')
    model_name = 'factor_ic'
    key = '{}_{}'.format(model_name, dt)
    log.info('key: {}{}'.format(key, '-' * 50))
    score = r.hgetall(key)
    score = {k.decode(): float(v.decode()) for k, v in score.items()}
    score = dict(sorted(score.items(), key=lambda x: x[1], reverse=True))
    log.info("score len: {}".format(len(score)))
    count = 0
    buy_stocks = {}
    for stock in score.keys():
        if count == g.stock_num:
            break
        code = stock.split('.')[0]
        if code in stocks_dic:
            buy_stocks[stocks_dic[code]] = score[stock]
            count += 1
    for k, v in buy_stocks.items():
        log.info('{}({})\t{}'.format(get_security_info(k, date=current_dt).display_name, k, round(v, 4)))
    buy_stocks = list(buy_stocks.keys())
    log.info('buy_stocks : {}'.format(buy_stocks))
    log.info('buy_stocks len：{}'.format(len(buy_stocks)))
    g.buy_stocks = buy_stocks

def trade(context):
    ''' 交易函数 '''
    log.info('{}\n函数运行时间（trade）:{}'.format('-' * 50, context.current_dt.time()))
    if len(g.buy_stocks) > 0:
        stocks = g.buy_stocks
    else:
        return
    # 出售不在stocks中的股票
    for stock in context.portfolio.positions.keys():
        if stock not in set(stocks):
            log.info('卖出股票：{}({})'.format(get_security_info(stock, date=context.current_dt).display_name, stock))
            order_target(stock, 0)
    # 买入新增股票
    for stock in stocks:
        if stock not in context.portfolio.positions \
                and len(context.portfolio.positions) < g.stock_num \
                and context.portfolio.available_cash > 0:
            cash = context.portfolio.available_cash / (g.stock_num - len(context.portfolio.positions))
            log.info('购买股票：{}({}), 购买金额：{}'.format(
                get_security_info(stock, date=context.current_dt).display_name, stock, round(cash, 4)))
            order_value(stock, cash)
    log.info('当前持仓：')
    pos = context.portfolio.positions
    for s in pos.keys():
        log.info('code: {}, name: {}, price: {}, 总仓位: {}, 可卖标的数: {}, 当前持仓成本: {}, 累计持仓成本: {}'.format(
                s, get_security_info(s, date=context.current_dt).display_name,
                pos[s].price, pos[s].total_amount, pos[s].closeable_amount,
                round(pos[s].avg_cost, 2), round(pos[s].acc_avg_cost, 2)
            )
        )

def after_market_close(context):
    log.info('{}\n函数运行时间(after_market_close)：{}'.format('-' * 50, context.current_dt.time()))
    # 得到当前所有成交记录
    for _trade in get_trades().values():
        log.info('成交记录：{}'.format(_trade))
    log.info('{} 这一天结束'.format(context.current_dt.date().strftime('%Y-%m-%d')))
    print('-' * 50)


