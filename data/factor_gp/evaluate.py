"""因子适应度评估：语法校验 → qlib 计算 → IC/ICIR → fitness"""

import numpy as np
import pandas as pd

import qlib
from qlib.data import D
from qlib.contrib.eva.alpha import calc_ic
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


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

    def __init__(self, instruments, target_train, target_test, config):
        self.instruments = instruments
        self.target_train = target_train
        self.target_test = target_test
        self.config = config

        # 共享缓存
        self.cache: dict[str, tuple] = {}  # expr_str → (fitness_train,)
        self.invalid_exprs: set[str] = set()  # 无效表达式
        self._eval_count: int = 0

    # ================================================================
    # Layer 1: 表达式验证
    # ================================================================

    def validate(self, expr_str: str) -> tuple[bool, str]:
        """验证表达式是否可计算且有截面区分度。

        Returns:
            (is_valid, reason)
        """
        # 1. 语法校验：DEAP 能否解析
        from deap import gp
        try:
            tree = gp.PrimitiveTree.from_string(expr_str, self.pset)
        except Exception as e:
            return False, f"语法错误: {e}"

        # 2. qlib 语义预检（基于 tree，不调用 D.features）
        ok, reason = self._check_qlib_semantics(tree)
        if not ok:
            return False, f"qlib 语义: {reason}"

        # 3. qlib 能否计算
        try:
            df_r = D.features(
                self.instruments, [expr_str],
                start_time=self.config.start_date,
                end_time=self.config.end_date,
            )
        except Exception as e:
            return False, f"qlib 计算失败: {e}"

        # 4. 结果非空且有截面区分度
        factor = df_r[expr_str].dropna()
        if len(factor) == 0:
            return False, "结果为空"
        if factor.std() < 1e-12:
            return False, "截面无区分度 (std=0)"

        return True, "ok"

    # ================================================================
    # Layer 2: 适应度计算
    # ================================================================

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
            logger.info('\n{}\n exprs len: {}'.format('-'* 50, len(unique_exprs)))
            try:
                # with np.errstate(divide="ignore", invalid="ignore"):
                df_r = D.features(
                    self.instruments, unique_exprs,
                    start_time=self.config.start_date,
                    end_time=self.config.train_end,
                )
            except Exception as e:
                err_msg = "训练集批量 D.features 失败: {}. \n batch_exprs: {}".format(e, batch_exprs)
                logger.error(err_msg)
                raise Exception(err_msg)
            logger.info('\n{}\n 表达式计算完成，耗时：{}s'.format('-' * 50, round(time.time() - t)))

        # 逐个计算 fitness
        t = time.time()
        for ind, expr_str in zip(batch_ind, batch_exprs):
            self._eval_count += 1
            fitness = self._compute_fitness(ind, expr_str, df_r)
            self.cache[expr_str] = fitness

        logger.info('{}\n IC 计算完成，耗时：{}s'.format('-' * 50, round(time.time() - t)))

        # 按原顺序返回
        return [self.cache[str(ind)] for ind in individuals]

    def _compute_fitness(self, individual, expr_str: str,
                         df_r: pd.DataFrame | None = None) -> tuple:
        """从因子值计算适应度。df_r 为 None 时自行调用 D.features。"""
        try:
            if df_r is not None:
                factor = df_r[expr_str].dropna()
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
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
            rank_ic_std = rank_ic_series.std()

            icir = rank_ic_mean / rank_ic_std if rank_ic_std > 1e-12 else 0.0

            depth = individual.height
            fitness = (
                abs(rank_ic_mean)
                + self.config.icir_weight * abs(icir)
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
    pset = None

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

        # 规则 0: 单节点树不能是字面常量（eg. 0.6973914688286109）
        if len(tree) == 1 and isinstance(tree[0], gp.Terminal):
            if self._node_is_literal(tree[0]):
                return False, f"表达式为纯字面常量: {tree[0].name}"

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
            # 规则 1: Max/Min 第二参数必须是 int
            if name in ("Max", "Min"):
                snd = tree[child_starts[1]]
                if not self._node_returns_int(snd):
                    return pos, False, (
                        f"Max/Min 第二参数非 int: {self._short(snd)}")

            # 规则 2: Sign/Abs 子节点不能是字面常量
            if name in ("Sign", "Abs"):
                child = tree[child_starts[0]]
                if self._node_is_literal(child):
                    return pos, False, (
                        f"{name} 作用于字面常量: {self._short(child)}")

            # 规则 3: 算术算子两个操作数不能都是布尔表达式
            # Div(bool,bool)→NotImplementedError crash,
            # Add/Mul/Sub(bool,bool)→无意义布尔运算
            if name in ("Add", "Sub", "Mul", "Div"):
                left = tree[child_starts[0]]
                right = tree[child_starts[1]]
                if self._node_is_bool(left) and self._node_is_bool(right):
                    return pos, False, (
                        f"{name} 操作数均为布尔: "
                        f"{self._short(left)}, {self._short(right)}")

            # 规则 4: Corr 前两参数不能是字面常量
            if name == "Corr":
                for i in (0, 1):
                    child = tree[child_starts[i]]
                    if self._node_is_literal(child):
                        return pos, False, (
                            f"Corr 参数{i + 1}是字面常量: {self._short(child)}")

            # 规则 5: If 第一参数（条件）不能是字面常量
            if name == "If":
                cond = tree[child_starts[0]]
                if self._node_is_literal(cond):
                    return pos, False, (
                        f"If 条件是字面常量: {self._short(cond)}")

            # 规则 6: 所有二元算子两个参数不能都是字面常量
            # Add(1,2)→numpy.float64, Gt(1,2)→numpy.bool_ 标量而非 pd.Series
            if name in ("Add", "Sub", "Mul", "Div", "Max", "Min", "Gt", "Lt"):
                left = tree[child_starts[0]]
                right = tree[child_starts[1]]
                if self._node_is_literal(left) and self._node_is_literal(right):
                    return pos, False, (
                        f"{name} 两参数均为字面常量: "
                        f"{self._short(left)}, {self._short(right)}")

            return pos, True, "ok"

        _, ok, reason = _next_idx(0)
        return ok, reason

    @staticmethod
    def _node_is_literal(node) -> bool:
        """节点是否是字面常量（C 或 N 生成的具体数值）。"""
        from deap import gp
        if not isinstance(node, gp.Terminal):
            return False
        try:
            float(node.name)
            return True
        except ValueError:
            return False

    @staticmethod
    def _node_is_bool(node) -> bool:
        """节点是否产生布尔值（Gt/Lt）。"""
        from deap import gp
        if isinstance(node, gp.Terminal):
            return False
        return node.name in ("Gt", "Lt")

    @staticmethod
    def _node_returns_int(node) -> bool:
        """节点输出类型是否为 int（N 终端或 IntCast）。"""
        from deap import gp
        if isinstance(node, gp.Terminal):
            return getattr(node, 'ret', None) is int
        return getattr(node, 'name', None) == "IntCast"

    @staticmethod
    def _short(node) -> str:
        from deap import gp
        if isinstance(node, gp.Terminal):
            return node.name
        return f"{node.name}(...)"
