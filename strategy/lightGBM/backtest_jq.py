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
from jqdata import get_trade_days

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
    g.stock_num = 50
    # 每天卖出得分最低的5只股票
    g.drop_k = 5
    # 记录是否为第一次交易
    g.is_first_trade = True
    # 选股。开盘前运行
    run_daily(choose_stock, time='before_open', reference_security='000300.XSHG')
    # 交易。开盘时运行
    run_daily(trade, time='open', reference_security='000300.XSHG')
    # 收盘后运行
    # run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')

def choose_stock(context):
    ''' 选股函数 '''
    log.info('{}\n函数运行时间（choose_stock）:{}'.format('-' * 50, context.current_dt.time()))

    # 从redis中获取股票得分
    r = redis.Redis(host='39.105.18.127', password='Zhao_38013984')
    # dt = (context.current_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    # 上一个交易日
    dt = get_trade_days(end_date=context.current_dt, count=(1 + 1))[0].strftime('%Y-%m-%d')
    print('dt: {}'.format(dt))
    model_name = 'lightGBM'
    key = '{}:{}'.format(model_name, dt)
    log.info('key: {}{}'.format(key, '-' * 50))
    score = r.hgetall(key)
    if len(score) == 0:
        log.info('未获取redis中的数据：{}'.format(key))
        return
    score = {k.decode(): round(float(v.decode()), 6) for k, v in score.items()}
    # 转化为聚宽的股票格式
    score = {((k[2:8]+'.XSHE') if k[0:2] == 'SZ' else (k[2:8]+'.XSHG')): v for k, v in score.items() if k[0:2] in ['SZ', 'SH']}
    log.info("score len: {}".format(len(score)))
    # 过滤st股
    score = {k: v for k, v in score.items() if 'ST' not in get_security_info(k, date=context.current_dt).display_name}
    score = dict(sorted(score.items(), key=lambda x: x[1], reverse=True))
    log.info("filter score len: {}".format(len(score)))
    # print(score)
    g.sorted_stocks_by_score = list(score.keys())
    # print(g.sorted_stocks_by_score[0:10])
    # 保存得分字典
    g.score_dict = score

def trade(context):
    ''' 交易函数 '''
    log.info('{}\n函数运行时间（trade）:{}'.format('-' * 50, context.current_dt.time()))

    # 获取当前持仓股票
    current_positions = list(context.portfolio.positions.keys())

    # 如果是第一天交易，只买入不卖出
    if g.is_first_trade:
        # 买进得分最高的g.stock_num只股票
        cash = context.portfolio.available_cash / g.stock_num
        for stock in g.sorted_stocks_by_score:
            if len(context.portfolio.positions) == g.stock_num:
                break
            log.info('首次交易买入股票：{}({}), 购买金额：{}'.format(
                get_security_info(stock, date=context.current_dt).display_name, stock, round(cash, 4)))
            order_value(stock, cash)
        # 标记第一次交易已完成
        g.is_first_trade = False
    else:
        # 非第一天交易，执行调仓操作

        # 确定要卖出的股票：得分最低的g.drop_k只股票
        # 从当前持仓中筛选出得分最低的g.drop_k只股票
        # 先给当前持仓股票按得分排序
        current_positions_with_scores = [(stock, g.score_dict.get(stock, -float('inf'))) for stock in set(current_positions)]
        current_positions_sorted = sorted(current_positions_with_scores, key=lambda x: x[1])
        sell_stocks = [stock for stock, score in current_positions_sorted[:g.drop_k]]
        print('current_positions_sorted: {}'.format(current_positions_sorted))
        print('sell_stocks: {}'.format('-' * 50))
        for stock in sell_stocks:
            print('code: {}, name: {}, score: {}'.format(stock,
                                                         get_security_info(stock, date=context.current_dt).display_name,
                                                         g.score_dict.get(stock, -float('inf'))))

        # 卖出得分最低的g.drop_k只股票
        for stock in sell_stocks:
            log.info('卖出股票：{}({})'.format(get_security_info(stock, date=context.current_dt).display_name, stock))
            order_target(stock, 0)

        # 确定要买入的股票：得分最高的g.drop_k只股票
        # 先从得分高的股票中筛选出当前未持仓的股票
        candidate_buy_stocks = [stock for stock in g.sorted_stocks_by_score if stock not in set(current_positions)]

        # 买入得分最高的g.drop_k只股票
        buy_stocks = []
        cash_for_buy = context.portfolio.available_cash
        if len(candidate_buy_stocks) > 0 and cash_for_buy > 0:
            cash_per_stock = cash_for_buy / g.drop_k
            pos_size_old = len(context.portfolio.positions)
            pos_size_new = len(context.portfolio.positions)
            for stock in candidate_buy_stocks:
                if len(context.portfolio.positions) == g.stock_num:
                    break
                log.info('买入股票：{}({}), 购买金额：{}'.format(
                    get_security_info(stock, date=context.current_dt).display_name, stock, round(cash_per_stock, 4)))
                order_value(stock, cash_per_stock)
                pos_size_new = len(context.portfolio.positions)
                if pos_size_new > pos_size_old:
                    buy_stocks.append(stock)
                    pos_size_old = pos_size_new

        print('buy_stocks: {}'.format('-' * 50))
        for stock in buy_stocks:
            print('code: {}, name: {}, score: {}'.format(stock,
                                                         get_security_info(stock, date=context.current_dt).display_name,
                                                         g.score_dict.get(stock, -float('inf'))))

    # 显示当前持仓
    pos = context.portfolio.positions
    log.info('当前持仓：{}'.format(len(pos)))
    for s in pos.keys():
        log.info('code: {}, name: {}, score: {}, price: {}, 总仓位: {}, 可卖标的数: {}, 当前持仓成本: {}, 累计持仓成本: {}'.format(
                s, get_security_info(s, date=context.current_dt).display_name,
                g.score_dict.get(s, -float('inf')),
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


