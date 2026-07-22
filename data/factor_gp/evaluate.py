"""因子适应度评估：语法校验 → qlib 计算 → IC/ICIR → fitness"""

import numpy as np
import pandas as pd

import qlib
from qlib.data import D
from qlib.contrib.eva.alpha import calc_ic
from utils import LoggerFactory
from .conf import ELEM_OPS, PIRE_OPS, ELEM_ROLLING_OPS, PAIR_ROLLING_OPS

logger = LoggerFactory.get_logger(__name__)


def _calc_fitness_worker(args):
    """Worker：因子值 → IC → fitness。模块级函数，用于 multiprocessing_wrapper_same。"""
    (expr_str, factor_values, target_values,
     icir_weight, complexity_penalty, depth) = args

    common_idx = factor_values.index.intersection(target_values.index)
    if len(common_idx) == 0:
        return (expr_str, (0.0,), True)

    pred = factor_values.loc[common_idx]
    label = target_values.loc[common_idx]
    pred = pred[np.isfinite(pred)]
    label = label[np.isfinite(label)]
    common_idx = pred.index.intersection(label.index)
    pred = pred.loc[common_idx]
    label = label.loc[common_idx]

    if len(pred) < 100:
        return (expr_str, (0.0,), True)

    from qlib.contrib.eva.alpha import calc_ic
    ic_series, rank_ic_series = calc_ic(pred, label)
    rank_ic_mean = rank_ic_series.mean()
    # rank_ic_std = rank_ic_series.std()
    # icir = rank_ic_mean / rank_ic_std if rank_ic_std > 1e-12 else 0.0

    fitness = abs(rank_ic_mean) - complexity_penalty * depth
    # fitness = abs(rank_ic_mean) + icir_weight * abs(icir) - complexity_penalty * depth
    if not np.isfinite(fitness):
        fitness = 0.0

    return (expr_str, (fitness,), False)


class FactorEvaluator:
    """因子评估器。

    两层评价：
    - Layer 1: 语法 + qlib 可计算性 + 截面区分度
    - Layer 2: 训练集 RankIC + ICIR + 复杂度惩罚 → fitness

    设计要点：
    - 适应度仅使用训练集，测试集在进化完成后统一评估
    - cache 跨岛屿共享（所有岛用同一个 evaluator 实例）
    - 无效表达式记录到 invalid_exprs，供 LLM 分析
    """

    def __init__(self, instruments, target_train, target_test, config, pset):
        self.instruments = instruments
        self.target_train = target_train
        self.target_test = target_test
        self.config = config
        self.pset = pset

        # 共享缓存
        self.cache: dict[str, tuple] = {}  # expr_str → (fitness_train,)
        self.invalid_exprs: set[str] = set()  # 无效表达式
        self._eval_count: int = 0

    def evaluate(self, individual) -> tuple:
        """计算个体适应度（训练集），返回 DEAP 需要的 tuple。"""
        expr_str = str(individual)
        if expr_str in self.cache:
            return self.cache[expr_str]

        self._eval_count += 1
        fitness = self._compute_fitness(individual, expr_str)
        self.cache[expr_str] = fitness
        return fitness

    def evaluate_batch(self, individuals: list) -> list[tuple]:
        """批量评估多个个体。

        一次 D.features 调用计算所有未缓存表达式，避免反复初始化/读盘。
        缓存命中的直接从 self.cache 取，不进入批量。

        Returns:
            fitness tuple 列表，与原 individuals 顺序一致
        """
        # 筛选合法表达式
        batch_ind = []
        batch_exprs = []
        for ind in individuals:
            expr_str = str(ind)
            if expr_str in self.cache:
                continue
            elif "IntCast" in expr_str or expr_str in self.invalid_exprs:
                self.cache[expr_str] = (0.0,)
            elif not self._passes_qlib_semantic(expr_str):
                self.invalid_exprs.add(expr_str)
                self.cache[expr_str] = (0.0,)
            else:
                batch_ind.append(ind)
                batch_exprs.append(expr_str)

        import time
        df_r = None
        if batch_exprs:
            t  = time.time()
            unique_exprs = list(set(batch_exprs))
            try:
                df_r = D.features(
                    self.instruments, unique_exprs,
                    start_time=self.config.start_date,
                    end_time=self.config.train_end,
                )
            except Exception as e:
                err_msg = "训练集批量 D.features 失败: {}. \n batch_exprs: {}".format(e, batch_exprs)
                logger.error(err_msg)
                raise Exception(err_msg)
            logger.info('\n{}\n qlib 表达式计算完成，耗时：{}s'.format('-' * 50, round(time.time() - t)))

        # 如果因子为NaN占比超过 50% ，则IC 设置为 0.0
        batch_ind_filtered = []
        nan_gt_05 = 0
        for i, expr in enumerate(batch_exprs):
            df_tmp = df_r[expr]
            nan_r = df_tmp.isna().sum() / len(df_tmp)
            if nan_r > 0.5:
                self.cache[expr] = (0.0,)
                nan_gt_05 += 1
            else:
                batch_ind_filtered.append(batch_ind[i])
        logger.info('值为NaN占比超过 50% 的因子数: {}/{}'.format(nan_gt_05, len(batch_exprs)))

        # 并行计算 IC
        t = time.time()
        if batch_ind:
            worker_args = []
            for ind in batch_ind:
                expr_str = str(ind)
                factor = df_r[expr_str].dropna()
                worker_args.append((
                    expr_str, factor, self.target_train,
                    self.config.icir_weight,
                    self.config.complexity_penalty,
                    ind.height,
                ))

            from utils.multiprocess import multiprocessing_wrapper_same
            results = multiprocessing_wrapper_same(
                _calc_fitness_worker,
                worker_args,
                n=self.config.kernels,
                start_method='fork',
            )

            for expr_str, fitness, is_invalid in results:
                if is_invalid:
                    self.invalid_exprs.add(expr_str)
                self.cache[expr_str] = fitness
                self._eval_count += 1

        logger.info('{}\n IC 计算完成(并行)，耗时：{}s'.format('-' * 50, round(time.time() - t)))

        # 按原顺序返回
        return [self.cache[str(ind)] for ind in individuals]

    def _compute_fitness(self, individual, expr_str: str,
                         df_r: pd.DataFrame | None = None) -> tuple:
        """从因子值计算适应度。df_r 为 None 时自行调用 D.features。"""
        try:
            if df_r is not None:
                factor = df_r[expr_str].dropna()
            else:
                df_r = D.features(
                    self.instruments, [expr_str],
                    start_time=self.config.start_date,
                    end_time=self.config.train_end,
                )
                factor = df_r[expr_str].dropna()

            if len(factor) == 0:
                self.invalid_exprs.add(expr_str)
                return (0.0,)

            common_idx = factor.index.intersection(self.target_train.index)
            if len(common_idx) == 0:
                self.invalid_exprs.add(expr_str)
                return (0.0,)

            pred = factor.loc[common_idx]
            label = self.target_train.loc[common_idx]

            pred = pred[np.isfinite(pred)]
            label = label[np.isfinite(label)]
            common_idx = pred.index.intersection(label.index)
            pred = pred.loc[common_idx]
            label = label.loc[common_idx]

            if len(pred) < 100:
                self.invalid_exprs.add(expr_str)
                return (0.0,)

            ic_series, rank_ic_series = calc_ic(pred, label)
            rank_ic_mean = rank_ic_series.mean()
            # rank_ic_std = rank_ic_series.std()
            #
            # icir = rank_ic_mean / rank_ic_std if rank_ic_std > 1e-12 else 0.0

            depth = individual.height
            fitness = (
                abs(rank_ic_mean)
                # + self.config.icir_weight * abs(icir)
                - self.config.complexity_penalty * depth
            )

            if not np.isfinite(fitness):
                fitness = 0.0

            return (fitness,)

        except Exception:
            self.invalid_exprs.add(expr_str)
            return (0.0,)

    # ================================================================
    # 测试集评估（不参与进化）
    # ================================================================

    def evaluate_test(self, expr_str: str) -> dict:
        """在测试集上评估因子，返回完整指标。"""
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                df_r = D.features(
                    self.instruments, [expr_str],
                    start_time=self.config.train_end,
                    end_time=self.config.end_date,
                )
            factor = df_r[expr_str].dropna()

            if len(factor) == 0:
                return {"error": "empty"}

            common_idx = factor.index.intersection(self.target_test.index)
            if len(common_idx) == 0:
                return {"error": "no overlap"}

            pred = factor.loc[common_idx]
            label = self.target_test.loc[common_idx]

            pred = pred[np.isfinite(pred)]
            label = label[np.isfinite(label)]
            common_idx = pred.index.intersection(label.index)
            pred = pred.loc[common_idx]
            label = label.loc[common_idx]

            if len(pred) < 100:
                return {"error": "too few samples"}

            ic, rank_ic = calc_ic(pred, label)

            return {
                "ic_mean": ic.mean(),
                "ic_std": ic.std(),
                "rank_ic_mean": rank_ic.mean(),
                "rank_ic_std": rank_ic.std(),
                "icir": rank_ic.mean() / rank_ic.std() if rank_ic.std() > 1e-12 else 0.0,
                "ic_series": ic,
                "rank_ic_series": rank_ic,
                "n_samples": len(pred),
            }

        except Exception as e:
            return {"error": str(e)}

    # ================================================================
    # 辅助方法
    # ================================================================

    @property
    def eval_count(self) -> int:
        return self._eval_count

    def get_train_ic(self, expr_str: str) -> float:
        """获取训练集 RankIC 均值（从缓存）"""
        if expr_str in self.cache:
            return self.cache[expr_str][0]
        individual = self._parse_expr(expr_str)
        if individual is not None:
            return self.evaluate(individual)[0]
        return 0.0

    def _parse_expr(self, expr_str: str):
        """将表达式字符串解析为 DEAP individual"""
        from deap import gp
        try:
            return gp.PrimitiveTree.from_string(expr_str, self.pset)
        except Exception:
            return None

    # pset 由外部设置（在 primitives.py 构建后注入）
    # pset = None

    # ================================================================
    # qlib 语义预检（基于 DEAP PrimitiveTree，不调用 D.features）
    # ================================================================

    def _passes_qlib_semantic(self, expr_str: str) -> bool:
        """解析表达式并做 qlib 语义预检，返回 True/False。"""
        from deap import gp
        try:
            tree = gp.PrimitiveTree.from_string(expr_str, self.pset)
        except Exception:
            return False
        ok, _ = self._check_qlib_semantics(tree)
        return ok

    def _check_qlib_semantics(self, tree) -> tuple[bool, str]:
        """基于 PrimitiveTree 做 qlib 语义预检，拦截已知不兼容模式。

        PrimitiveTree 是前缀序扁平列表（list-like），Primitive 的子节点
        按 arity 紧随其后。用索引方式遍历。
        """
        from deap import gp

        # 防御：空树
        if len(tree) == 0:
            return True, "ok"

        # 规则 0: 单节点树不能是常量（eg. 0.6973914688286109）
        if len(tree) == 1 and isinstance(tree[0], gp.Terminal):
            if self._node_is_literal(tree[0]):
                return False, f"表达式为常量: {tree[0].name}"

        def _next_idx(idx):
            """返回 subtree 在 tree[idx] 结束后的下一个索引，同时做规则检查。
            返回 (next_idx, ok, reason)"""
            node = tree[idx]
            if isinstance(node, gp.Terminal):
                return idx + 1, True, "ok"

            name = node.name
            arity = node.arity

            # 收集子树的起始索引
            child_starts = []
            pos = idx + 1
            for _ in range(arity):
                child_starts.append(pos)
                pos, ok, reason = _next_idx(pos)
                if not ok:
                    return pos, False, reason

            # ---- 规则检查 ----
            # 规则1: feature 参数不能为常量
            if name in ELEM_OPS + ELEM_ROLLING_OPS:
                child = tree[child_starts[0]]
                if self._node_is_literal(child):
                    return pos, False, (
                        f"{name} 的feature是常量: {self._short(child)}")

            # 规则2: feature_left 参数不能为常量
            if name in PIRE_OPS + PAIR_ROLLING_OPS:
                child = tree[child_starts[0]]
                if self._node_is_literal(child):
                    return pos, False, (
                        f"{name} 的feature_left是常量: {self._short(child)}")

            # 规则3: 除了 Power 算子外，其他算子的 feature_right不能为常量
            if name in set(PIRE_OPS + PAIR_ROLLING_OPS) - set(['Power']):
                child = tree[child_starts[1]]
                if self._node_is_literal(child):
                    return pos, False, (
                        f"{name} 的feature_right是常量: {self._short(child)}")

            # 规则4: Log: 为了保证元素值需要为正。需要配置 Abs 使用
            if name == "Log":
                child = tree[child_starts[0]]
                if child.name != 'Abs':
                    return pos, False, (
                        f"{name} 的feature需要为正数(加 Abs): {self._short(child)}")

            # 规则5: Quantile 的 qscore 为常量，取值 [0, 1]
            # feature 不能为 bool 类型，因为 bool 取值0/1，无法分组
            if name == "Quantile":
                feature = tree[child_starts[0]]
                qscore_node = tree[child_starts[2]]
                if self._node_is_bool(feature):
                    return pos, False, (
                        f"Quantile 的feature不能为布尔类型: {self._short(feature)}")
                if not self._node_is_literal(qscore_node):
                    return pos, False, (
                        f"Quantile 的qscore必须是常量: {self._short(qscore_node)}")
                qs = float(qscore_node.name)
                if not (0 <= qs <= 1):
                    return pos, False, f"Quantile qscore={qs} 不在 [0,1]"

            # 规则6: Rolling算子中，参数N为int类型常量，N>0
            if name in ELEM_ROLLING_OPS + PAIR_ROLLING_OPS:
                if name in ELEM_ROLLING_OPS:
                    child = tree[child_starts[1]]
                else:
                    child = tree[child_starts[2]]

                if not self._node_is_literal(child):
                    return pos, False, (
                        f"{name} 的参数N必须是常量: {self._short(child)}")

                if not self._node_is_int(child):
                    return pos, False, (
                        f"{name} 的参数N必须是整数: {self._short(child)}")

                n = int(child.name)
                if n <= 0:
                    return pos, False, f"{name} 的参数N必须>0"

            # 规则7: If 第一参数（条件）不能为常量，元素为bool 类型
            # feature_left, feature_right 不能为常量
            if name == "If":
                cond = tree[child_starts[0]]
                left = tree[child_starts[1]]
                right = tree[child_starts[2]]
                if self._node_is_literal(cond):
                    return pos, False, (
                        f"If 条件是常量: {self._short(cond)}")
                if not self._node_is_bool(cond):
                    return pos, False, (
                        f"If 的条件必须是布尔: {self._short(cond)}")
                if self._node_is_literal(left) or self._node_is_literal(right):
                    return pos, False, (
                        f"If 的feature_left 或 feature_right是常量: {self._short(left)} 或 {self._short(right)}")

            # 规则8: Not 的feature 必须为 bool 类型
            if name == 'Not':
                child = tree[child_starts[0]]
                if not self._node_is_bool(child):
                    return pos, False, (
                        f"Not 的feature 必须为 bool: {self._short(child)}")

            # 规则 9: And,Or 的 feature_left, feature_right 必须为 bool 类型
            if name in ("And", "Or"):
                left = tree[child_starts[0]]
                right = tree[child_starts[1]]
                if not self._node_is_bool(left) or not self._node_is_bool(right):
                    return pos, False, (
                        f"{name} 的feature_left 或 feature_right必须为 bool: {self._short(left)} 或 {self._short(right)}")

            # 规则10: 除了Not, 单元素算子 featrue 不能为 bool 类型
            if name in (set(ELEM_OPS) - set(['Not'])):
                child = tree[child_starts[0]]
                if self._node_is_bool(child):
                    return pos, False, (
                        f"{name} 的feature必须是 bool: {self._short(child)}")

            # 规则11: 元素对算子（Power, Add, Sub, Mul, Div, Greater, Less）的feature_left, feature_right 类型不能为bool
            if name in ("Power", "Add", "Sub", "Mul", "Div", "Greater", "Less"):
                left = tree[child_starts[0]]
                right = tree[child_starts[1]]
                if self._node_is_bool(left) or self._node_is_bool(right):
                    return pos, False, (
                        f"{name} 的feature_left 或 feature_right不能为 bool: {self._short(left)} 或 {self._short(right)}")

            # 规则11b: Power 指数必须为字面常数且 ∈ {0.5, 2, 3}（防高次幂爆炸+过拟合）
            if name == "Power":
                right = tree[child_starts[1]]
                if not self._node_is_literal(right):
                    return pos, False, (
                        f"Power 指数必须为常量: {self._short(right)}")
                exp = float(right.name)
                if exp not in (0.5, 2.0, 3.0):
                    return pos, False, (
                        f"Power 指数必须∈{{0.5,2,3}}: {exp}")

            # 规则12: 单元素 Rolling算子 feature 不能为 bool 类型
            if name in ELEM_ROLLING_OPS:
                child = tree[child_starts[0]]
                if self._node_is_bool(child):
                    return pos, False, (
                        f"{name} 的feature必须是 bool: {self._short(child)}")

            # 规则13: 元素对 Rolling算子（Corr, Cov）的feature_left, feature_right 不能为 bool 类型
            if name in PAIR_ROLLING_OPS:
                left = tree[child_starts[0]]
                right = tree[child_starts[1]]
                if self._node_is_bool(left) or self._node_is_bool(right):
                    return pos, False, (
                        f"{name} 的feature_left 或 feature_right不能为 bool: {self._short(left)} 或 {self._short(right)}")

            return pos, True, "ok"





        _, ok, reason = _next_idx(0)
        return ok, reason

    @staticmethod
    def _node_is_literal(node) -> bool:
        """节点是否是常量（C 或 N 生成的具体数值）。"""
        from deap import gp
        # 排除非叶子节点
        if not isinstance(node, gp.Terminal):
            return False
        try:
            float(node.name)
            return True
        except ValueError:
            return False

    @staticmethod
    def _node_is_bool(node) -> bool:
        """ 节点是否是 bool 类型。
        """
        from deap import gp
        if isinstance(node, gp.Terminal):
            return False
        if node.name in ("Gt", "Ge", "Lt", "Le", "Eq", "Ne", "And", "Or", "Not"):
            return True
        return False

    @staticmethod
    def _node_is_int(node) -> bool:
        """节点输出类型是否为 int（N 终端或 IntCast）。"""
        from deap import gp
        if isinstance(node, gp.Terminal):
            # 'ret' 是 gp.Terminal 对象类型属性
            return getattr(node, 'ret', None) is int
        # 'name' 是 gp.Primitive 对象函数名称
        return getattr(node, 'name', None) == "IntCast"

    @staticmethod
    def _short(node) -> str:
        from deap import gp
        if isinstance(node, gp.Terminal):
            return node.name
        return f"{node.name}(...)"
