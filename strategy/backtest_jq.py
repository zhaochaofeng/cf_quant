'''
    聚宽回测代码
'''

import redis
import pandas as pd
from jqdata import get_trade_days


def initialize(context):
    log.info('\n{}\n函数运行时间（initialize）:{}'.format('-' * 50, context.current_dt.time()))
    # 设置沪深300作为基准
    set_benchmark('000300.XSHG')
    # 设置动态复权(动态前复权)
    set_option('use_real_price', True)
    # 交易成交比例
    set_option('order_volume_ratio', 1.0)
    # 设置佣金/印花税
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0.001,
        open_commission=0.0005, close_commission=0.0005,
        close_today_commission=0, min_commission=5
    ), type='stock')

    # 最大持仓数
    g.stock_num = 50
    # 每天卖出得分最低股票数
    g.drop_k = 5
    # 选股。开盘前运行
    run_daily(choose_stock, time='before_open', reference_security='000300.XSHG')
    # 交易。开盘时运行
    run_daily(trade, time='open', reference_security='000300.XSHG')
    # 收盘后运行
    # run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')


def choose_stock(context):
    ''' 选股函数 '''
    log.info('{}\n函数运行时间（choose_stock）:{}'.format('-' * 50, context.current_dt.time()))

    # 上一个交易日
    # dt = get_trade_days(end_date=context.current_dt, count=2)[0].strftime('%Y-%m-%d')
    dt = context.current_dt.strftime('%Y-%m-%d')
    model_name = 'lightgbm_alpha:csi300'
    key = '{}:{}'.format(model_name, dt)
    log.info('\nkey: {}{} '.format(key, '-' * 50))
    try:
        # 从redis中获取股票得分
        r = redis.Redis(host='39.105.18.127', password='Zhao_38013984')
        score = r.hgetall(key)
    except Exception as e:
        log.error('redis连接失败：{}'.format(e))
        return
    if not score:
        log.error('未获取redis中的数据：{}'.format(key))
        return
    score = {k.decode(): round(float(v.decode()), 6) for k, v in score.items()}
    # 转化为聚宽的股票格式
    score = {((k[2:8] + '.XSHE') if k[0:2] == 'SZ' else (k[2:8] + '.XSHG')): v for k, v in score.items() if
             k[0:2] in ['SZ', 'SH']}
    log.info("score len: {}".format(len(score)))
    # 过滤st股
    # score = {k: v for k, v in score.items() if 'ST' not in get_security_info(k, date=context.current_dt).display_name}
    # 降序排序
    score = dict(sorted(score.items(), key=lambda x: x[1], reverse=True))
    log.info("filter score len: {}".format(len(score)))
    # 排序后的股票代码
    g.sorted_stocks_by_score = list(score.keys())
    # 得分字典
    g.score_dict = score


def trade(context):
    ''' 交易函数 - 基于 TopkDropoutStrategy 算法逻辑 '''
    log.info('\n{}\n函数运行时间（trade）:{}'.format('-' * 50, context.current_dt.time()))

    if not hasattr(g, 'score_dict') or len(g.score_dict) == 0:
        log.error('无可交易的打分结果，跳过当日交易')
        return

    # 将 score_dict 转换为 pandas Series
    pred_score = pd.Series(g.score_dict)

    # 获取当前持仓股票列表
    current_positions = list(context.portfolio.positions.keys())

    # 算法参数
    topk = g.stock_num  # 持仓股票数量
    n_drop = g.drop_k  # 每次调仓替换的股票数量
    risk_degree = 0.95  # 仓位比例

    # Step 1: last - 当前持仓按照分值从高到低排序
    last = pred_score.reindex(current_positions).sort_values(ascending=False).index

    # Step 2: today - pred_score 排除 last，然后按分值降序后取前 n_drop + topk - len(last) 个元素
    today = pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index
    today = list(today[:n_drop + topk - len(last)])

    # Step 3: comb - 合并 today + last，按分值降序排列
    comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index

    # Step 4: sell - 从 comb 获取最低分的 n_drop 只股票，与 last 取交集
    sell = last[last.isin(list(comb[-n_drop:]))]

    # Step 5: buy - 从 today 中获取前 len(sell) + topk - len(last) 个股票
    buy = today[:len(sell) + topk - len(last)]

    # 打印调试信息
    log.info('当前持仓数量: {}'.format(len(last)))
    log.info('候选买入数量: {}'.format(len(today)))
    log.info('合并后数量: {}'.format(len(comb)))
    log.info('计划卖出数量: {}'.format(len(sell)))
    log.info('计划买入数量: {}'.format(len(buy)))

    print('\nsell_stocks: {}'.format('-' * 50))
    for stock in sell:
        print('code: {}, name: {}, score: {}'.format(
            stock,
            get_security_info(stock, date=context.current_dt).display_name,
            g.score_dict.get(stock, -float('inf'))))

    # Step 6: 先卖后买 - 卖出操作
    for stock in sell:
        position = context.portfolio.positions.get(stock)
        if position and position.closeable_amount > 0:
            log.info('卖出股票：{}({}), score: {}'.format(
                get_security_info(stock, date=context.current_dt).display_name,
                stock,
                g.score_dict.get(stock, -float('inf'))))
            order_target(stock, 0, MarketOrderStyle(1))

    # Step 7: 买入操作
    buy_stocks = []
    if len(buy) > 0:
        # 计算每只股票的买入金额
        value_per_stock = context.portfolio.available_cash * risk_degree / len(buy)

        pos_size_old = len(context.portfolio.positions)
        for stock in buy:
            log.info('买入股票：{}({}), 购买金额：{}, score: {}'.format(
                get_security_info(stock, date=context.current_dt).display_name,
                stock,
                round(value_per_stock, 4),
                g.score_dict.get(stock, -float('inf'))))
            order_value(stock, value_per_stock, MarketOrderStyle(9999))

            # 记录买入成功的股票
            pos_size_new = len(context.portfolio.positions)
            if pos_size_new > pos_size_old:
                buy_stocks.append(stock)
                pos_size_old = pos_size_new

    print('\nbuy_stocks: {}'.format('-' * 50))
    for stock in buy_stocks:
        print('code: {}, name: {}, score: {}'.format(
            stock,
            get_security_info(stock, date=context.current_dt).display_name,
            g.score_dict.get(stock, -float('inf'))))

    # 显示当前持仓
    pos = context.portfolio.positions
    log.info('当前持仓：{}'.format(len(pos)))
    for s in pos.keys():
        log.info('code: {}, name: {}, score: {}, price: {}, 总仓位: {}, 可卖标的数: {}, 当前持仓成本: {}, 累计持仓成本: {}'.format(
            s,
            get_security_info(s, date=context.current_dt).display_name,
            g.score_dict.get(s, -float('inf')),
            pos[s].price,
            pos[s].total_amount,
            pos[s].closeable_amount,
            round(pos[s].avg_cost, 2),
            round(pos[s].acc_avg_cost, 2)
        )
        )


def after_market_close(context):
    log.info('{}\n函数运行时间(after_market_close)：{}'.format('-' * 50, context.current_dt.time()))
    # 得到当前所有成交记录
    for _trade in get_trades().values():
        log.info('成交记录：{}'.format(_trade))
    log.info('{} 这一天结束'.format(context.current_dt.date().strftime('%Y-%m-%d')))
    print('-' * 50)
