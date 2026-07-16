"""后处理筛选：测试集评估 + 低相关筛选 + 报告生成"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FactorScreening:
    """后处理：测试集评估 → 相关性筛选 → 报告生成。"""

    def __init__(self, evaluator, config):
        self.evaluator = evaluator
        self.config = config

    # ================================================================
    # 测试集评估
    # ================================================================

    def evaluate_all_on_test(
        self, candidates: list[tuple[str, float]]
    ) -> pd.DataFrame:
        """对所有候选因子计算测试集指标。

        Args:
            candidates: [(expr_str, train_fitness), ...]

        Returns:
            DataFrame: expr | train_fitness | ic_train | icir_train |
                       ic_test | rank_ic_test | icir_test | depth
        """
        rows = []
        for expr_str, train_fitness in candidates:
            test_result = self.evaluator.evaluate_test(expr_str)

            if "error" in test_result:
                continue

            depth = self._expr_depth(expr_str)

            rows.append({
                "expr": expr_str,
                "train_fitness": train_fitness,
                "train_ic": self.evaluator.get_train_ic(expr_str),
                "test_rank_ic": test_result["rank_ic_mean"],
                "test_icir": test_result["icir"],
                "depth": depth,
                "n_samples": test_result["n_samples"],
                "ic_decay": self.evaluator.get_train_ic(expr_str)
                - test_result["rank_ic_mean"],
            })

        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("无有效候选因子")
            return df

        # 统一方向后计算绝对指标
        df["abs_test_rank_ic"] = df["test_rank_ic"].abs()
        df["abs_train_ic"] = df["train_ic"].abs()

        # 按测试集 |RankIC| 降序
        df = df.sort_values("abs_test_rank_ic", ascending=False).reset_index(drop=True)

        logger.info(
            "测试集评估: %d 个候选, |RankIC| mean=%.4f, max=%.4f",
            len(df), df["abs_test_rank_ic"].mean(), df["abs_test_rank_ic"].max(),
        )
        return df

    # ================================================================
    # 低相关筛选
    # ================================================================

    def filter_by_correlation(
        self, df: pd.DataFrame, threshold: float = None
    ) -> pd.DataFrame:
        """贪心低相关筛选。

        算法：
        1. 按测试集 |RankIC| 降序排列
        2. 逐个检查：如果与已选因子的最大相关性 < threshold，则保留
        3. 输出筛选后的因子集合

        Returns:
            筛选后的 DataFrame（新增 corr_to_selected 列）
        """
        if threshold is None:
            threshold = self.config.corr_threshold

        if df.empty:
            return df

        # 计算因子值矩阵（测试集区间）
        factor_matrix = self._build_factor_matrix(df["expr"].tolist())
        if factor_matrix is None or factor_matrix.shape[1] < 2:
            logger.warning("因子相关性矩阵构建失败或因子太少")
            df["max_corr"] = 0.0
            return df.assign(selected=True)

        # 计算相关性矩阵
        corr_matrix = factor_matrix.corr()

        # 按 |RankIC| 降序，确保是副本
        df = df.sort_values("abs_test_rank_ic", ascending=False).reset_index(drop=True)

        selected_indices = []
        max_corrs = []

        for i in range(len(df)):
            expr_i = df.iloc[i]["expr"]
            if expr_i not in corr_matrix.columns:
                max_corrs.append(0.0)
                continue

            if not selected_indices:
                selected_indices.append(i)
                max_corrs.append(0.0)
                continue

            # 与已选择因子的最大相关性
            selected_exprs = [df.iloc[j]["expr"] for j in selected_indices
                              if df.iloc[j]["expr"] in corr_matrix.columns]
            if not selected_exprs:
                selected_indices.append(i)
                max_corrs.append(0.0)
                continue

            max_corr = corr_matrix.loc[expr_i, selected_exprs].abs().max()
            max_corrs.append(max_corr)

            if max_corr < threshold:
                selected_indices.append(i)

        df["max_corr"] = max_corrs
        df["selected"] = df.index.isin(selected_indices)

        n_selected = df["selected"].sum()
        logger.info(
            "低相关筛选 (threshold=%.2f): %d/%d 保留",
            threshold, n_selected, len(df),
        )

        return df

    # ================================================================
    # 报告生成
    # ================================================================

    def generate_report(
        self, df: pd.DataFrame, output_dir: str = None
    ) -> str:
        """生成因子报告摘要。

        Returns:
            报告文本
        """
        if output_dir is None:
            output_dir = self.config.output_dir

        selected = df[df.get("selected", True)]

        lines = [
            "=" * 60,
            "GP + LLM 因子挖掘报告",
            "=" * 60,
            f"样本区间: {self.config.start_date} ~ {self.config.end_date}",
            f"训练集: ~ {self.config.train_end}",
            f"测试集: {self.config.train_end} ~ {self.config.end_date}",
            "",
            f"候选因子总数: {len(df)}",
            f"筛选后因子数: {len(selected)}",
            f"相关性阈值: {self.config.corr_threshold}",
            "",
            "--- 全量候选统计 ---",
            f"测试集 |RankIC| 均值: {df['abs_test_rank_ic'].mean():.4f}",
            f"测试集 |RankIC| 中位数: {df['abs_test_rank_ic'].median():.4f}",
            f"测试集 |RankIC| 最大值: {df['abs_test_rank_ic'].max():.4f}",
            f"测试集 ICIR 均值: {df['test_icir'].mean():.4f}",
            "",
            "--- Top-10 因子 ---",
        ]

        for i, row in selected.head(10).iterrows():
            expr_str = row["expr"]
            # 截断过长表达式
            display_expr = expr_str if len(expr_str) <= 100 else expr_str[:97] + "..."
            lines.append(
                f"  [{i}] |RankIC|={row['abs_test_rank_ic']:.4f} "
                f"ICIR={row['test_icir']:.4f} "
                f"depth={row['depth']} "
                f"max_corr={row.get('max_corr', 0):.2f}"
            )
            lines.append(f"       {display_expr}")

        lines.append("")
        lines.append("=" * 60)

        report = "\n".join(lines)
        logger.info(report)
        return report

    # ================================================================
    # 辅助方法
    # ================================================================

    def _build_factor_matrix(self, expr_list: list[str]) -> pd.DataFrame | None:
        """构建因子值矩阵（测试集区间），用于计算相关性。"""
        from qlib.data import D

        factor_series = {}
        for expr_str in expr_list:
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    df_r = D.features(
                        self.evaluator.instruments, [expr_str],
                        start_time=self.config.train_end,
                        end_time=self.config.end_date,
                    )
                series = df_r[expr_str].dropna()
                if len(series) > 100:
                    factor_series[expr_str] = series
            except Exception:
                continue

        if len(factor_series) < 2:
            return None

        # 对齐 index
        # 取所有 series 中最近一段时间的共同日期
        matrix = pd.DataFrame(factor_series)
        matrix = matrix.dropna()
        return matrix

    @staticmethod
    def _expr_depth(expr_str: str) -> int:
        """估算表达式嵌套深度。"""
        depth = 0
        max_depth = 0
        for c in expr_str:
            if c == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif c == ')':
                depth -= 1
        return max_depth
