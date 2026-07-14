"""DEAP + qlib GP 因子挖掘 Demo — 验证方案可行性

验证要点：
1. DEAP 注册 qlib 算子 → 生成合法 qlib 表达式字符串
2. qlib 评估表达式 → 截面 IC 计算 fitness
3. DEAP 进化循环 → 产出因子表达式
"""
import multiprocessing

# multiprocessing.set_start_method("fork", force=True)

import random
import operator
import logging
import time
from functools import partial

import numpy as np

logging.getLogger("qlib").setLevel(logging.ERROR)

import qlib
from qlib.data import D

from deap import base, creator, gp, tools, algorithms

import argparse


def get_ret(start_date, end_date, market='csi300'):
    # qlib 多线程 worker 各自创建 logger, 主线程 setLevel 不生效
    # 需要用 Filter 全局屏蔽 WARNING
    for name in ["qlib", "qlib.Max", "qlib.Min"]:
        logging.getLogger(name).addFilter(lambda r: r.levelno >= logging.ERROR)
    config = D.instruments(market=market)
    instruments = D.list_instruments(config, start_time=start_date, end_time=end_date)

    fields = ['$close', '$open', '$high', '$low', '$volume', '$amount']
    df = D.features(instruments, fields, start_time=start_date, end_time=end_date)

    target = df['$close'].groupby('instrument', group_keys=False).apply(lambda x: x.shift(-2) / x.shift(-1) - 1)
    target.dropna(inplace=True)

    print(f"Data: {len(instruments)} instruments, "
          f"{len(df.index.get_level_values('datetime').unique())} dates, "
          f"{len(target)} target samples\n")
    return target, instruments


def cross_sectional_ic(returns_series, factor_series):
    from qlib.contrib.eva.alpha import calc_ic
    return calc_ic(factor_series, returns_series)[0].mean()


def main(args):

    target, instruments = get_ret(args.start_date, args.end_date)

    random.seed(42)
    np.random.seed(42)

    pset = gp.PrimitiveSetTyped("MAIN", [], float, 0)

    # 添加叶子节点
    for f in ['$close', '$open', '$high', '$low', '$volume', '$amount']:
        pset.addTerminal(f, ret_type=float, name=f)
    # 叶子节点常量
    pset.addEphemeralConstant("C", partial(random.uniform, -1, 1), ret_type=float)
    pset.addEphemeralConstant("N", partial(random.randint, 5, 30), ret_type=int)
    # 创建一个接受任意数量参数并返回 None 的匿名函数，等价于：
    # def _dummy(*a):
    #     return None
    _dummy = lambda *a: None

    # Element-wise
    for op in ["Abs", "Log", "Sign"]:
        pset.addPrimitive(_dummy, in_types=[float], ret_type=float, name=op)

    # Pair-wise
    for op in ["Add", "Sub", "Mul", "Div", "Max", "Min", "Gt", "Lt"]:
        pset.addPrimitive(_dummy, in_types=[float, float], ret_type=float, name=op)

    # Rolling (arity=2)
    for op in ["Ref", "Mean", "Std", "Delta", "Rank", "WMA", "EMA"]:
        pset.addPrimitive(_dummy, in_types=[float, int], ret_type=float, name=op)

    pset.addPrimitive(_dummy, in_types=[float, float, int], ret_type=float, name="Corr")
    pset.addPrimitive(_dummy, in_types=[float, float, float], ret_type=float, name="If")

    # 安全网: float → int 转换。确保 generate() 在 int 位置不崩溃
    # Fix: 理解逻辑
    pset.addPrimitive(_dummy, [float], int, name="IntCast")

    if 'FitnessMax' in creator.__dict__:
        del creator.FitnessMax
    if 'Individual' in creator.__dict__:
        del creator.Individual
    # 创建新类
    # FitnessMax：定义计算规则
    # Individual：定义树结构
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    # DEAP 的注册中心。所有遗传操作（生成、评估、选择、交叉、变异）都通过 toolbox.register
    toolbox = base.Toolbox()
    # 表达式
    # genGrow + min_=1: 根节点强制 primitive, 子节点允许 terminal (不依赖 IntCast)
    toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=3)
    # individual 生成器。树 + 表达式
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    # 种群生成器(多个 individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    cache = {}

    def evaluate(individual):
        """评估个体的适应度(Fitness).
        DEAP 的 fitness 统一设计为多目标，Fitness 类的 values 是一个tuple
        所以函数返回tuple，不能是scalar
        """
        expr_str = str(individual)
        if expr_str in cache:
            return (cache[expr_str],)
        try:
            df_r = D.features(instruments, [expr_str],
                              start_time=args.start_date, end_time=args.end_date)
            factor = df_r[expr_str].dropna()
            if len(factor) == 0:
                cache[expr_str] = 0.0
                return (0.0,)
            ic = cross_sectional_ic(target, factor)
            # 常数因子产生 nan (相关系数除以零), 视为无效
            if not np.isfinite(ic):
                ic = 0.0
            cache[expr_str] = ic
            return (ic,)
        except Exception:
            cache[expr_str] = 0.0
            return (0.0,)

    # 适应度评估。qlib 评估方式注册到 deap
    toolbox.register("evaluate", evaluate)
    # 选择算子。选 fitness 最高的进入下一代
    toolbox.register("select", tools.selTournament, tournsize=3)
    # 交叉算子
    toolbox.register("mate", gp.cxOnePoint)
    # 变异子树生成器
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    # 变异算子
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    # 树深度约束
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

    print("=" * 60)
    pop = toolbox.population(n=args.n_pop)
    # 筛选得分最高的表达式(全历史最优)
    hof = tools.HallOfFame(args.hof_size)
    # 注册统计量
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("max", np.max)

    t_start = time.time()
    popu, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=args.n_gen,
        stats=stats, halloffame=hof, verbose=True
    )
    t_total = time.time() - t_start

    print(f"\n完成! 总耗时: {t_total:.1f}s, "
          f"评估表达式数: {len(cache)} (含缓存命中), "
          f"avg IC/expr: {np.mean(list(cache.values())):.4f}\n")

    print("=" * 60)
    for i, ind in enumerate(hof):
        expr_str = str(ind)
        print(f"  [{i}] IC={ind.fitness.values[0]:.4f}  {expr_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GP Demo')
    parser.add_argument('--start-date', type=str, default='2025-06-01')
    parser.add_argument('--end-date', type=str, default='2026-06-01')
    parser.add_argument('--n-pop', type=int, default=10, help='population 大小')
    parser.add_argument('--n-gen', type=int, default=5, help='进化代数')
    parser.add_argument('--hof_size', type=int, default=15, help='Hall of Fame 大小')

    args = parser.parse_args()
    print(args)
    qlib.init(provider_uri='~/.qlib/qlib_data/custom_data_hfq')
    main(args)



