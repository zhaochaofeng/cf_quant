"""分岛进化引擎 — GP + LLM 因子挖掘核心。

关键设计：
- 每个岛独立的 population + toolbox + stats + hof
- 所有岛共享同一个 evaluator（从而共享表达式缓存）
- 岛间环形迁移
- LLM 周期性注入（竞争替换，而非直接注入）
"""

import logging
import operator
import time
from dataclasses import dataclass, field

import numpy as np

from deap import base, creator, gp, tools

logger = logging.getLogger(__name__)


@dataclass
class Island:
    """单个进化岛屿的状态。"""

    id: int
    population: list = field(default_factory=list)
    toolbox: base.Toolbox | None = None
    hof: tools.HallOfFame | None = None
    stats: tools.Statistics | None = None
    logbook: tools.Logbook | None = None

    # 运行时统计
    gen: int = 0
    best_fitness: float = 0.0
    invalid_count: int = 0


class IslandEvolution:
    """分岛遗传编程引擎。

    Usage:
        evaluator = FactorEvaluator(...)
        registry = PrimitiveRegistry()
        pset = registry.build_pset()

        engine = IslandEvolution(pset, evaluator, config)
        engine.setup_islands()
        result = engine.run()
    """

    def __init__(self, pset, evaluator, config, llm_interface=None):
        self.pset = pset
        self.evaluator = evaluator
        self.config = config
        self.llm = llm_interface

        # 将 pset 注入 evaluator（用于 validate 中的表达式解析）
        evaluator.pset = pset

        self.islands: list[Island] = []
        self._gen_start_time: float = 0.0

    # ================================================================
    # 岛屿初始化
    # ================================================================

    def setup_islands(self):
        """为每个岛创建独立的 toolbox 和初始种群。

        注意：每个岛的 toolbox 共享同一个 evaluator（从而共享缓存）。
        """
        # 确保 DEAP creator 类只创建一次
        self._setup_creator()

        for i in range(self.config.n_islands):
            island = Island(id=i)
            island.toolbox = self._build_toolbox()
            island.hof = tools.HallOfFame(5)
            island.stats = tools.Statistics(lambda ind: ind.fitness.values[0])
            island.stats.register("avg", np.nanmean)
            island.stats.register("std", np.nanstd)
            island.stats.register("max", np.nanmax)
            island.logbook = tools.Logbook()
            island.population = island.toolbox.population(n=self.config.n_pop)

            # 初始化种群评估
            self._eval_population(island)

            self.islands.append(island)

        logger.info(
            "分岛初始化完成: %d 岛 × %d 个体, 共 %d 个体",
            self.config.n_islands, self.config.n_pop,
            self.config.n_islands * self.config.n_pop,
        )

    def _setup_creator(self):
        """确保 DEAP creator 类只创建一次。"""
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    def _build_toolbox(self) -> base.Toolbox:
        """构建单个岛的 toolbox。"""
        toolbox = base.Toolbox()

        # 表达式生成
        toolbox.register("expr", gp.genGrow, pset=self.pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # 适应度评估（共享 evaluator）
        toolbox.register("evaluate", self.evaluator.evaluate)

        # 遗传算子
        toolbox.register("select", tools.selTournament, tournsize=self.config.tournsize)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)

        # 树深度约束
        toolbox.decorate("mate", gp.staticLimit(
            key=operator.attrgetter("height"), max_value=self.config.max_height))
        toolbox.decorate("mutate", gp.staticLimit(
            key=operator.attrgetter("height"), max_value=self.config.max_height))

        return toolbox

    # ================================================================
    # 主进化循环
    # ================================================================

    def run(self) -> "EvolutionResult":
        """主进化循环。"""
        logger.info("=" * 60)
        logger.info("开始分岛进化: %d 代", self.config.n_gen)

        t_start = time.time()

        for gen in range(self.config.n_gen):
            self._gen_start_time = time.time()

            # 1. 各岛独立进化一代
            for island in self.islands:
                self._evolve_one_generation(island)
                island.gen = gen + 1

            # 2. 打印代际统计
            self._log_generation(gen)

            # 3. 岛间迁移
            if gen > 0 and gen % self.config.migration_freq == 0:
                self._migrate()

            # 4. LLM 注入
            if self.llm is not None and gen > 0 and gen % self.config.llm_inject_freq == 0:
                self._llm_inject(gen)

        t_total = time.time() - t_start

        logger.info("进化完成! 总耗时: %.1fs, 评估表达式数: %d (含缓存)",
                     t_total, self.evaluator.eval_count)

        return self.collect_result()

    def _evolve_one_generation(self, island: Island):
        """对单个岛执行一代进化（选择 → 交叉 → 变异 → 评估 → 替换）。"""
        pop = island.population
        toolbox = island.toolbox

        # 选择 offspring
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # 交叉
        for i in range(1, len(offspring), 2):
            if np.random.random() < self.config.cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(
                    offspring[i - 1], offspring[i])
                # 清除被修改个体的适应度
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        # 变异
        for i in range(len(offspring)):
            if np.random.random() < self.config.mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # 评估未计算适应度的个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        island.invalid_count = len(invalid_ind)
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        # 替换
        island.population = offspring

        # 更新统计
        island.hof.update(island.population)
        record = island.stats.compile(island.population)
        island.logbook.record(gen=island.gen, **record)

        if island.hof:
            island.best_fitness = island.hof[0].fitness.values[0]

    def _eval_population(self, island: Island):
        """评估整群个体（仅初始化时调用）。

        DEAP 创建种群时 fitness.values 为空，需要首次评估使每个个体有分数。
        不评估则后续选择算子、统计量 compiley 和 HoF 更新无法工作。

        _evolve_one_generation 处理的是交叉变异产生的新个体，初始种群需要单独评估。
        """
        invalid_ind = [ind for ind in island.population if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = island.toolbox.evaluate(ind)
        island.hof.update(island.population)
        record = island.stats.compile(island.population)
        island.logbook.record(gen=0, **record)

    # ================================================================
    # 岛间迁移
    # ================================================================

    def _migrate(self):
        """环形迁移：每岛迁出最佳个体到下一个岛，接收上一个岛的最佳个体。

        每个岛：
        1. 选出本岛 HoF 最佳个体作为 emigrant
        2. 发送给下一个岛（环形：最后一岛 → 第一岛）
        3. 接收上一个岛的 emigrant，替换本岛最差个体
        """
        n = len(self.islands)
        emigrants = []

        for island in self.islands:
            if island.hof:
                emigrant = island.toolbox.clone(island.hof[0])
                emigrants.append(emigrant)
            else:
                emigrants.append(None)

        for i, island in enumerate(self.islands):
            src_idx = (i - 1) % n
            emigrant = emigrants[src_idx]
            if emigrant is None:
                continue

            # 重新评估适应度（确保 fitness 有效）
            if not emigrant.fitness.valid:
                emigrant.fitness.values = island.toolbox.evaluate(emigrant)

            # 替换最差个体
            worst_idx = min(range(len(island.population)),
                            key=lambda j: island.population[j].fitness.values[0])
            island.population[worst_idx] = emigrant

        logger.debug("岛间迁移完成: %d 岛", n)

    # ================================================================
    # LLM 注入
    # ================================================================

    def _llm_inject(self, gen: int):
        """LLM 周期性注入：生成候选因子，竞争替换进入各岛。

        流程：
        1. 收集各岛当前 Top-K 表达式 + 失败案例
        2. 调用 LLM 生成新候选表达式
        3. 对每个候选：评估 → 与岛内底部个体竞争 → 只保留优胜者
        """
        # 收集各岛 Top-3
        top_exprs = []
        for island in self.islands:
            for ind in island.hof:
                top_exprs.append((str(ind), ind.fitness.values[0]))

        # 去重
        seen = set()
        unique_top = []
        for expr, fit in top_exprs:
            if expr not in seen:
                seen.add(expr)
                unique_top.append((expr, fit))

        # 失败案例
        invalid_list = list(self.evaluator.invalid_exprs)[-50:]  # 最近 50 个

        logger.info("LLM 注入 (gen %d): 提供 Top-%d 表达式 + %d 失败案例",
                     gen, len(unique_top), len(invalid_list))

        try:
            candidates = self.llm.generate_candidates(
                top_exprs=unique_top[:10],
                invalid_patterns=invalid_list,
                gen=gen,
            )
        except Exception as e:
            logger.warning("LLM 生成失败: %s", e)
            return

        if not candidates:
            logger.info("LLM 未生成有效候选")
            return

        # 竞争替换：每个候选 vs 岛内底部个体
        accepted = 0
        for expr_str in candidates:
            # 解析为 individual
            try:
                ind = gp.PrimitiveTree.from_string(expr_str, self.pset)
            except Exception:
                continue

            ind = creator.Individual(ind)
            ind.fitness.values = self.evaluator.evaluate(ind)
            fitness = ind.fitness.values[0]

            if fitness <= 1e-8:
                continue

            # 找最弱的岛
            for island in self.islands:
                bottom = min(island.population,
                             key=lambda x: x.fitness.values[0])

                if fitness > bottom.fitness.values[0]:
                    # 替换
                    idx = island.population.index(bottom)
                    island.population[idx] = ind
                    accepted += 1
                    logger.debug("LLM 因子进入岛 %d: fitness=%.4f, %s",
                                 island.id, fitness, expr_str[:80])
                    break  # 一个因子只注入一个岛

        logger.info("LLM 注入结果: %d/%d 候选进入种群",
                     accepted, len(candidates))

    # ================================================================
    # 结果收集
    # ================================================================

    def collect_result(self) -> "EvolutionResult":
        """收集所有岛屿的候选因子。"""
        all_exprs = {}  # expr_str → best fitness

        for island in self.islands:
            for ind in island.hof:
                expr_str = str(ind)
                fitness = ind.fitness.values[0]
                if expr_str not in all_exprs or fitness > all_exprs[expr_str]:
                    all_exprs[expr_str] = fitness

        # 按 fitness 降序
        candidates = sorted(all_exprs.items(), key=lambda x: x[1], reverse=True)

        return EvolutionResult(
            candidates=candidates,
            islands=self.islands,
            total_evals=self.evaluator.eval_count,
            cache_size=len(self.evaluator.cache),
            logbooks=[isl.logbook for isl in self.islands],
        )

    # ================================================================
    # 日志
    # ================================================================

    def _log_generation(self, gen: int):
        elapsed = time.time() - self._gen_start_time
        parts = [f"Gen {gen + 1}/{self.config.n_gen} ({elapsed:.1f}s)"]

        for island in self.islands:
            if island.hof:
                parts.append(
                    f"岛{island.id}: best={island.best_fitness:.4f} "
                    f"invalid={island.invalid_count}"
                )

        parts.append(f"cache={len(self.evaluator.cache)}")
        logger.info(" | ".join(parts))


@dataclass
class EvolutionResult:
    """进化结果。"""

    candidates: list  # [(expr_str, fitness), ...] 按 fitness 降序
    islands: list[Island]
    total_evals: int
    cache_size: int
    logbooks: list
