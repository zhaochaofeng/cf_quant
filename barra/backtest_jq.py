'''
    聚宽回测代码. 从 MySQL portfolio 表中读取交易指令
'''

# 回测环境自动安装 pymysql（研究环境与回测环境隔离）
import subprocess
import importlib
import sys

try:
    import pymysql
except ImportError:
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'pymysql', '-t', '.'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    importlib.invalidate_caches()
    import pymysql

from sqlalchemy import create_engine, text


def qlib_to_jq_code(qlib_code):
    '''
    将 Qlib 代码格式转为聚宽股票代码格式
    SZ000001 -> 000001.XSHE
    SH600519 -> 600519.XSHG
    '''
    if len(qlib_code) < 8:
        return qlib_code
    prefix = qlib_code[0:2]
    code = qlib_code[2:8]
    if prefix == 'SZ':
        return code + '.XSHE'
    elif prefix == 'SH':
        return code + '.XSHG'
    else:
        log.warn('未知股票前缀: {}'.format(qlib_code))
        return qlib_code


def initialize(context):
    log.info('\n{}\n函数运行时间（initialize）:{}'.format('-' * 50, context.current_dt.time()))
    # 设置沪深300作为基准
    set_benchmark('000300.XSHG')
    # 设置动态复权(动态前复权)
    set_option('use_real_price', True)
    # 设置佣金/印花税
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0.001,
        open_commission=0.0003, close_commission=0.0003,
        close_today_commission=0, min_commission=5
    ), type='stock')

    # 组合名称, 对应 MySQL portfolio 表的 portfolio 字段
    g.portfolio_name = 'default'

    # 选股。开盘前运行
    run_daily(choose_stock, time='before_open', reference_security='000300.XSHG')
    # 交易。开盘时运行
    run_daily(trade, time='open', reference_security='000300.XSHG')
    # 收盘后运行
    run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')


def choose_stock(context):
    ''' 从 MySQL portfolio 表读取当日交易指令 '''
    log.info('\n{}\n函数运行时间（choose_stock）:{}'.format('-' * 50, context.current_dt.time()))

    dt = context.current_dt.strftime('%Y-%m-%d')
    log.info('读取MySQL交易信号: date={}, portfolio={}'.format(dt, g.portfolio_name))

    try:
        engine = create_engine(
            'mysql+pymysql://cf_reader:Zhao_123@47.93.20.118:23306/cf_quant?charset=utf8mb4',
        )
        with engine.connect() as conn:
            sql = text('''
                SELECT qlib_code, trade_shares, direction, price, hold_shares
                FROM portfolio
                WHERE day = :day AND portfolio = :portfolio
                ORDER BY qlib_code
            ''')
            result = conn.execute(sql, {'day': dt, 'portfolio': g.portfolio_name})
            rows = [dict(r) for r in result]
    except Exception as e:
        log.error('MySQL查询失败: {}'.format(e))
        rows = None

    if not rows:
        log.error('未获取到交易信号: date={}, portfolio={}'.format(dt, g.portfolio_name))
        g.trade_orders = None
        return

    log.info('MySQL原始信号数: {}'.format(len(rows)))

    # 转换为聚宽格式, 并过滤无效股票
    trade_orders = []
    skipped = 0
    for row in rows:
        jq_code = qlib_to_jq_code(row['qlib_code'])
        direction = row['direction']
        trade_shares = int(row['trade_shares'])

        # 跳过不需要交易的
        if direction == 'hold' or trade_shares == 0:
            skipped += 1
            continue

        # 验证股票代码有效性
        try:
            sec_info = get_security_info(jq_code, date=context.current_dt)
            if sec_info is None:
                log.warn('无效股票: {} (raw: {})'.format(jq_code, row['qlib_code']))
                skipped += 1
                continue
        except Exception as e:
            log.warn('get_security_info失败: {}: {}'.format(jq_code, e))
            skipped += 1
            continue

        trade_orders.append({
            'stock': jq_code,
            'trade_shares': trade_shares,
            'direction': direction,
            'display_name': sec_info.display_name,
        })

    if skipped > 0:
        log.info('过滤跳过: {} 条'.format(skipped))

    if not trade_orders:
        log.info('当日无有效交易指令')
        g.trade_orders = None
        return

    # 打印买卖清单
    buy_orders = [o for o in trade_orders if o['direction'] == 'buy']
    sell_orders = [o for o in trade_orders if o['direction'] == 'sell']
    if sell_orders:
        print('sell_stocks: {}'.format('-' * 50))
        for o in sell_orders:
            print('code: {}, name: {}, trade_shares: {}'.format(
                o['stock'], o['display_name'], abs(o['trade_shares'])))
    if buy_orders:
        print('buy_stocks: {}'.format('-' * 50))
        for o in buy_orders:
            print('code: {}, name: {}, trade_shares: {}'.format(
                o['stock'], o['display_name'], o['trade_shares']))

    log.info('有效交易信号: 买入{}只, 卖出{}只, 共{}条'.format(
        len(buy_orders), len(sell_orders), len(trade_orders)))
    g.trade_orders = trade_orders


def trade(context):
    ''' 执行交易指令: 先卖后买 '''
    log.info('\n{}\n函数运行时间（trade）:{}'.format('-' * 50, context.current_dt.time()))

    if not hasattr(g, 'trade_orders') or g.trade_orders is None:
        log.error('无交易信号, 跳过当日交易')
        return

    if len(g.trade_orders) == 0:
        log.info('当日无交易')
        return

    # 分离买卖
    sell_orders = [o for o in g.trade_orders if o['direction'] == 'sell']
    buy_orders = [o for o in g.trade_orders if o['direction'] == 'buy']

    # 先卖出
    for o in sell_orders:
        log.info('卖出股票: {}({}), {}股'.format(o['display_name'], o['stock'], abs(o['trade_shares'])))
        order(o['stock'], o['trade_shares'])

    # 后买入
    for o in buy_orders:
        log.info('买入股票: {}({}), {}股'.format(o['display_name'], o['stock'], o['trade_shares']))
        order(o['stock'], o['trade_shares'])

    # 打印当前持仓
    log.info('当前持仓:')
    pos = context.portfolio.positions
    for s in pos.keys():
        log.info('code: {}, name: {}, price: {}, 总仓位: {}, 可卖标的数: {}, 当前持仓成本: {}, 累计持仓成本: {}'.format(
            s, get_security_info(s, date=context.current_dt).display_name,
            pos[s].price, pos[s].total_amount, pos[s].closeable_amount,
            round(pos[s].avg_cost, 2), round(pos[s].acc_avg_cost, 2),
        ))


def after_market_close(context):
    ''' 收盘后记录交易信息 '''
    log.info('\n{}\n函数运行时间（after_market_close）:{}'.format('-' * 50, context.current_dt.time()))

    # 当日成交记录
    trades = get_trades()
    if trades:
        log.info('当日成交数: {}'.format(len(trades)))
        for _trade in trades.values():
            log.info('成交记录: {}'.format(_trade))
    else:
        log.info('当日无成交')
    log.info('{} 这一天结束'.format(context.current_dt.strftime('%Y-%m-%d')))
    print('-' * 50)
