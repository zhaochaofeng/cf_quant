"""分岛进化引擎 — GP + LLM 因子挖掘核心。

关键设计：
- 每个岛独立的 population + toolbox + stats + hof
- 所有岛共享同一个 evaluator（从而共享表达式缓存）
- 岛间环形迁移
- LLM 周期性注入（竞争替换，而非直接注入）
"""

import operator
import os
import pickle
import random
import time
from dataclasses import dataclass, field

import numpy as np

from deap import base, creator, gp, tools
from utils.dt import time_decorator
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


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
    gen: int = 0      # 当前代数
    best_fitness: float = 0.0      # 当前最佳 fitness
    invalid_count: int = 0   # 本次迭代评估的表达式个数
    economic_descs: dict = field(default_factory=dict)  # expr_str → 经济学含义描述


class IslandEvolution:
    """分岛遗传编程引擎。

    Usage:
        evaluator = FactorEvaluator(...)
        registry = PrimitiveRegistry()
        pset = registry.build_pset()

        engine = IslandEvolution(pset, evaluator, config)
        result = engine.run()  # setup_islands() + 断点逻辑由 run() 内部处理
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
            island.hof = tools.HallOfFame(self.config.n_hof)
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
    # Checkpoint
    # ================================================================

    def _save_checkpoint(self, gen: int):
        """保存当前进化状态到文件。

        DEAP 官方模式：pickle population + hof + logbook + random state。
        toolbox 和 stats 因为包含 lambda/bound method 不序列化，resume 时重建。
        """
        path = os.path.join(self.config.output_dir,
                            f"checkpoint_gen_{gen + 1}.pkl")

        cp = {
            "generation": gen + 1,
            "config_seed": self.config.seed,
            "config_n_islands": self.config.n_islands,
            "config_n_pop": self.config.n_pop,
            "islands": [{
                "population": isl.population,
                "hof": isl.hof,
                "logbook": isl.logbook,
                "gen": isl.gen,
                "best_fitness": isl.best_fitness,
                "economic_descs": isl.economic_descs,
            } for isl in self.islands],
            "evaluator": {
                "cache": self.evaluator.cache,
                "fitness_set": self.evaluator.fitness_set,
                "invalid_exprs": self.evaluator.invalid_exprs,
                "ic_cache": getattr(self.evaluator, "ic_cache", {}),
                "eval_count": self.evaluator._eval_count,
            },
            "random_state": random.getstate(),
            "np_random_state": np.random.get_state(),
        }
        with open(path, "wb") as f:
            pickle.dump(cp, f)
        logger.info("Checkpoint 已保存: gen=%d → %s", gen + 1, path)

    def _load_checkpoint(self, path: str) -> int:
        """从 checkpoint 文件恢复进化状态，返回已完成代数。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint 文件不存在: {path}")

        with open(path, "rb") as f:
            cp = pickle.load(f)

        # 验证核心配置一致性
        for key in ["config_seed", "config_n_islands"]:
            stored = cp.get(key)
            current = getattr(self.config, key.replace("config_", ""))
            if stored != current:
                logger.warning(
                    "Checkpoint %s=%s != config %s=%s，恢复结果可能不一致",
                    key, stored, key.replace("config_", ""), current,
                )

        start_gen = cp["generation"]
        logger.info("加载 Checkpoint: gen=%d, file=%s", start_gen, path)

        # 1. 确保 creator 类存在（pickle 反序列化需要）
        self._setup_creator()

        # 2. 恢复随机状态
        random.setstate(cp["random_state"])
        np.random.set_state(cp["np_random_state"])

        # 3. 恢复 evaluator 缓存
        ev = cp["evaluator"]
        self.evaluator.cache = ev["cache"]
        self.evaluator.fitness_set = ev["fitness_set"]
        self.evaluator.invalid_exprs = ev["invalid_exprs"]
        if hasattr(self.evaluator, "ic_cache"):
            self.evaluator.ic_cache = ev.get("ic_cache", {})
        self.evaluator._eval_count = ev["eval_count"]

        # 4. 重建岛屿（toolbox + stats 重建，其余从 checkpoint 加载）
        self.islands = []
        for i, isl_data in enumerate(cp["islands"]):
            island = Island(id=i)
            island.toolbox = self._build_toolbox()
            island.stats = tools.Statistics(lambda ind: ind.fitness.values[0])
            island.stats.register("avg", np.nanmean)
            island.stats.register("std", np.nanstd)
            island.stats.register("max", np.nanmax)
            island.population = isl_data["population"]
            island.hof = isl_data["hof"]
            island.logbook = isl_data["logbook"]
            island.gen = isl_data["gen"]
            island.best_fitness = isl_data["best_fitness"]
            island.economic_descs = isl_data.get("economic_descs", {})
            self.islands.append(island)

        logger.info("Checkpoint 恢复完成: %d islands, evaluator cache=%d",
                     len(self.islands), len(self.evaluator.cache))
        return start_gen

    # ================================================================
    # 主进化循环
    # ================================================================
    def run(self) -> "EvolutionResult":
        """主进化循环"""
        # 断点恢复分支
        if self.config.resume_from:
            start_gen = self._load_checkpoint(self.config.resume_from)
        else:
            self.setup_islands()
            start_gen = 0

        logger.info("=" * 60)
        logger.info("分岛进化: gen=[%d, %d]", start_gen+1, self.config.n_gen)

        t_start = time.time()
        gen_candidates = {}

        for gen in range(start_gen, self.config.n_gen):
            self._gen_start_time = time.time()

            # 1. 各岛独立进化一代
            for island in self.islands:
                self._evolve_one_generation(island)
                island.gen = gen + 1

            # 2. 打印代际统计
            self._log_generation(gen)

            # 2.5 LLM 经济学含义过滤
            if (self.llm is not None
                    and self.config.enable_economic_check
                    and gen > 0
                    and (gen + 1) % self.config.llm_check_freq == 0):
                self._llm_economic_filter(gen)

            # 3. 岛间迁移
            if gen > 0 and gen % self.config.migration_freq == 0:
                self._migrate()

            # 4. LLM 注入
            if self.llm is not None and gen > 0 and (gen+1) % self.config.llm_inject_freq == 0:
                self._llm_inject(gen)

            # 5. 保存每一代 candidates
            candi = self.collect_result().candidates
            gen_candidates['gen_{}'.format(gen+1)] = candi

            # 6. Checkpoint 保存
            if self.config.checkpoint_freq > 0 and (gen + 1) % self.config.checkpoint_freq == 0:
                self._save_checkpoint(gen)

        # 最终 checkpoint
        if self.config.checkpoint_freq > 0:
            self._save_checkpoint(self.config.n_gen - 1)

        # 6. 最终经济学描述补全（覆盖进化后期新进入 HoF 但未经检查的表达式）
        if self.llm is not None and self.config.enable_economic_check:
            self._fill_missing_descs()

        t_total = time.time() - t_start

        from utils import PickleIO
        PickleIO.write(gen_candidates, f'{self.config.output_dir}/gen_candidates.pkl')
        logger.info("进化完成! 总耗时: %.1fs, 评估表达式数: %d (含缓存)",
                     t_total, self.evaluator.eval_count)

        return self.collect_result()

    def _evolve_one_generation(self, island: Island):
        """对单个岛执行一代进化（选择 → 交叉 → 变异 → 评估 → 替换）。"""
        pop = island.population
        toolbox = island.toolbox

        # 选择 offspring：list[individual]
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

        # 评估未计算 fitness 的个体（批量）
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        island.invalid_count = len(invalid_ind)
        if invalid_ind:
            fitnesses = self.evaluator.evaluate_batch(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

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
        # invalid_ind 是 list[PrimitiveTree]
        invalid_ind = [ind for ind in island.population if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = self.evaluator.evaluate_batch(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        island.hof.update(island.population)
        record = island.stats.compile(island.population)
        island.logbook.record(gen=0, **record)

    # ================================================================
    # 岛间迁移
    # ================================================================
    @time_decorator
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

        logger.info("岛间迁移完成: %d 岛", n)

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

        # 正常表达式
        unique_top = sorted(unique_top, key=lambda x: abs(x[1]), reverse=True)
        unique_top = [(expr, fitness) for expr, fitness in unique_top if abs(fitness)>0.0001]
        valid_list = unique_top[0:50]   # top 50 个
        # 失败案例
        invalid_list = list(self.evaluator.invalid_exprs)[-50:]  # 最近 50 个

        logger.info("LLM 注入 (gen %d): 提供 Top-%d 表达式 + %d 失败案例",
                     gen+1, len(valid_list), len(invalid_list))

        try:
            candidates = self.llm.generate_candidates(
                top_exprs=valid_list,
                invalid_patterns=invalid_list
            )
        except Exception as e:
            logger.warning("LLM 生成失败: %s", e)
            return

        candidates = [c for c in candidates if self.evaluator._passes_qlib_semantic(c)]
        logger.info('valid candidates len: {}'.format(len(candidates)))

        if not candidates:
            logger.info("LLM 未生成有效候选")
            return

        logger.info('\n{} {}'.format('-' * 30, 'pset Primitive'))
        pri_list = self.pset.primitives[float]
        pris = [pri.name for pri in pri_list]
        logger.info(pris)

        logger.info('\n{} {}'.format('-' * 30, 'pset Terminals'))
        ter_list = self.pset.terminals[float]
        ters = [ter.name for ter in ter_list]
        logger.info(ters)

        # 竞争替换：每个候选 vs 岛内底部个体
        accepted = 0
        for expr_str in candidates:
            # 解析为 individual
            try:
                logger.info('expr_str: {}'.format(expr_str))
                ind = gp.PrimitiveTree.from_string(expr_str, self.pset)
            except Exception as e:
                # continue
                raise Exception("Invalid expression: {}, e: {}".format(expr_str, e))

            ind = creator.Individual(ind)
            ind.fitness.values = self.evaluator.evaluate(ind)
            fitness = ind.fitness.values[0]
            logger.info('gen: {}, candidate: {}, fitness: {}'.format(gen, expr_str, fitness))
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

        logger.info("LLM 注入结果: %d/%d 候选进入种群",accepted, len(candidates))

    # ================================================================
    # LLM 经济学含义过滤
    # ================================================================

    def _llm_economic_filter(self, gen: int):
        """LLM 经济学含义过滤：批量评估种群表达式，淘汰无意义个体。"""
        # 1. 收集所有岛的去重表达式（population + hof）
        # 注意：HoF 中的历史最佳可能已被进化替换出 population，
        # 若只扫描 population 会遗漏这些表达式，导致最终候选缺少经济学描述。
        all_exprs = set()
        for island in self.islands:
            for ind in island.population:
                all_exprs.add(str(ind))
            for ind in island.hof:
                all_exprs.add(str(ind))
        expr_list = sorted(all_exprs)

        if not expr_list:
            return

        logger.info("LLM 经济检查 (gen %d): %d 个唯一表达式", gen + 1, len(expr_list))

        # 2. 批量调用 LLM 评估
        try:
            results = self.llm.assess_economic_meaning(expr_list)
        except Exception as e:
            logger.warning("LLM 经济检查失败: %s", e)
            return

        if not results:
            logger.info("LLM 经济检查未返回有效结果，跳过")
            return

        # 3. 建立 expr → (meaningful, desc) 映射
        expr_verdict = {}  # expr_str → (meaningful: bool, desc: str)
        for item in results:
            idx = item.get("id")
            if idx is None or not isinstance(idx, int) or idx >= len(expr_list):
                continue
            expr_str = expr_list[idx]
            meaningful = item.get("meaningful", True)
            desc = item.get("desc", "")
            expr_verdict[expr_str] = (meaningful, desc)

        # 4. 各岛执行过滤 + 存储描述
        total_removed = 0
        for island in self.islands:
            new_pop = []
            removed = 0
            for ind in island.population:
                expr_str = str(ind)
                verdict = expr_verdict.get(expr_str)
                # 没有经济学描述的因子保留，仅删除 meaningful=False 的因子
                if verdict is None:
                    new_pop.append(ind)
                    continue
                meaningful, desc = verdict
                if desc:
                    island.economic_descs[expr_str] = desc
                if meaningful:
                    new_pop.append(ind)
                else:
                    removed += 1
                    self.evaluator.invalid_exprs.add(expr_str)
                    logger.debug("经济过滤淘汰 [岛%d]: %s", island.id, expr_str[:80])

            # 补充新个体维持种群大小
            if removed > 0:
                offspring = [island.toolbox.clone(ind) for ind in
                             island.toolbox.population(n=removed)]
                fitnesses = self.evaluator.evaluate_batch(offspring)
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit
                new_pop.extend(offspring)

            island.population = new_pop
            island.hof.update(island.population)
            total_removed += removed

            # 存储 HoF 中表达式的经济学描述（HoF 个体可能不在 population 中）
            for ind in island.hof:
                expr_str = str(ind)
                if expr_str not in island.economic_descs:
                    verdict = expr_verdict.get(expr_str)
                    if verdict is not None and verdict[1]:
                        island.economic_descs[expr_str] = verdict[1]

        logger.info("LLM 经济检查完成 (gen %d): 淘汰 %d 个体, 补充 %d 新个体",
                     gen + 1, total_removed, total_removed)

    def _fill_missing_descs(self):
        """最终补全：为 HoF 中缺少经济学描述的表达式调用 LLM 获取 desc。

        进化过程中，经济检查仅在特定代触发（llm_check_freq），
        后期新进入 HoF 的表达式或 LLM 注入的候选可能从未被检查。
        此方法在进化结束后一次性补全，确保最终候选都有经济学解释。
        """
        # 收集所有 HoF 中缺少 desc 的表达式
        missing = set()
        for island in self.islands:
            for ind in island.hof:
                expr_str = str(ind)
                if "IntCast" in expr_str:
                    continue
                if expr_str not in island.economic_descs:
                    missing.add(expr_str)

        if not missing:
            return

        expr_list = sorted(missing)
        logger.info("经济学描述补全: %d 个候选缺少描述", len(expr_list))

        try:
            results = self.llm.assess_economic_meaning(expr_list)
        except Exception as e:
            logger.warning("LLM 描述补全失败: %s", e)
            return

        if not results:
            return

        # 存储 desc 到对应岛
        for item in results:
            idx = item.get("id")
            desc = item.get("desc", "")
            if idx is None or idx >= len(expr_list) or not desc:
                continue
            expr_str = expr_list[idx]
            for island in self.islands:
                for ind in island.hof:
                    if str(ind) == expr_str and expr_str not in island.economic_descs:
                        island.economic_descs[expr_str] = desc

        filled = sum(1 for island in self.islands
                     for expr in missing if expr in island.economic_descs)
        logger.info("经济学描述补全完成: %d/%d 个表达式获得描述", filled, len(missing))

    # ================================================================
    # 结果收集
    # ================================================================

    def collect_result(self) -> "EvolutionResult":
        """收集所有岛屿的候选因子。"""
        all_exprs = {}  # expr_str → best fitness
        economic_descs = {}  # expr_str → 经济学含义描述

        for island in self.islands:
            for ind in island.hof:
                expr_str = str(ind)
                if "IntCast" in expr_str:
                    continue
                fitness = ind.fitness.values[0]
                if expr_str not in all_exprs or fitness > all_exprs[expr_str]:
                    all_exprs[expr_str] = fitness
                # 收集经济学描述（各岛可能不同，取非空值）
                if expr_str not in economic_descs and island.economic_descs.get(expr_str):
                    economic_descs[expr_str] = island.economic_descs[expr_str]

        # 按 fitness 降序
        candidates = sorted(all_exprs.items(), key=lambda x: x[1], reverse=True)

        return EvolutionResult(
            candidates=candidates,
            islands=self.islands,
            total_evals=self.evaluator.eval_count,
            cache_size=len(self.evaluator.cache),
            logbooks=[isl.logbook for isl in self.islands],
            economic_descs=economic_descs,
        )

    # ================================================================
    # 日志
    # ================================================================
    @time_decorator
    def _log_generation(self, gen: int):
        elapsed = time.time() - self._gen_start_time
        parts = [f"Gen {gen + 1}/{self.config.n_gen} ({elapsed:.1f}s)"]

        for island in self.islands:
            if island.hof:
                parts.append(
                    f"岛{island.id + 1}: best={island.best_fitness:.4f} "
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
    economic_descs: dict = field(default_factory=dict)  # expr_str → 经济学含义描述
