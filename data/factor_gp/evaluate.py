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
            gp.PrimitiveTree.from_string(expr_str, self.pset)
        except Exception as e:
            return False, f"语法错误: {e}"

        # 2. qlib 能否计算
        try:
            df_r = D.features(
                self.instruments, [expr_str],
                start_time=self.config.start_date,
                end_time=self.config.end_date,
            )
        except Exception as e:
            return False, f"qlib 计算失败: {e}"

        # 3. 结果非空且有截面区分度
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
        """计算个体适应度（训练集），返回 DEAP 需要的 tuple。

        适应度 = |RankIC_train| + icir_weight * |ICIR_train| - complexity_penalty * depth
        """
        expr_str = str(individual)

        # 缓存命中
        if expr_str in self.cache:
            return self.cache[expr_str]

        self._eval_count += 1

        try:
            # qlib 计算因子值（屏蔽 Log(≤0) 等数学无效告警）
            with np.errstate(divide="ignore", invalid="ignore"):
                df_r = D.features(
                    self.instruments, [expr_str],
                    start_time=self.config.start_date,
                    end_time=self.config.train_end,
                )
            factor = df_r[expr_str].dropna()

            if len(factor) == 0:
                self.invalid_exprs.add(expr_str)
                result = (0.0,)
                self.cache[expr_str] = result
                return result

            # 对齐预测值与标签
            common_idx = factor.index.intersection(self.target_train.index)
            if len(common_idx) == 0:
                self.invalid_exprs.add(expr_str)
                result = (0.0,)
                self.cache[expr_str] = result
                return result

            pred = factor.loc[common_idx]
            label = self.target_train.loc[common_idx]

            # 去除 inf/NaN
            pred = pred[np.isfinite(pred)]
            label = label[np.isfinite(label)]
            common_idx = pred.index.intersection(label.index)
            pred = pred.loc[common_idx]
            label = label.loc[common_idx]

            if len(pred) < 100:
                self.invalid_exprs.add(expr_str)
                result = (0.0,)
                self.cache[expr_str] = result
                return result

            # RankIC + ICIR
            ic_series, rank_ic_series = calc_ic(pred, label)
            ic_mean = ic_series.mean()
            rank_ic_mean = rank_ic_series.mean()
            rank_ic_std = rank_ic_series.std()

            if rank_ic_std < 1e-12:
                icir = 0.0
            else:
                icir = rank_ic_mean / rank_ic_std

            # 适应度 = |RankIC| + weight * |ICIR| - penalty * depth
            depth = individual.height
            fitness = (
                abs(rank_ic_mean)
                + self.config.icir_weight * abs(icir)
                - self.config.complexity_penalty * depth
            )

            if not np.isfinite(fitness):
                fitness = 0.0

            result = (fitness,)
            self.cache[expr_str] = result
            return result

        except Exception:
            self.invalid_exprs.add(expr_str)
            result = (0.0,)
            self.cache[expr_str] = result
            return result

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
