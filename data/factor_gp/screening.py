"""后处理筛选：测试集评估 + 低相关筛选 + 报告生成"""

import numpy as np
import pandas as pd
from utils import DataFrameIO, PickleIO
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class FactorScreening:
    """后处理：测试集评估 → 相关性筛选 → 报告生成。"""

    def __init__(self, evaluator, config, pset):
        self.evaluator = evaluator
        self.config = config
        self.pset = pset

    # ================================================================
    # 测试集评估
    # ================================================================

    def evaluate_all_on_test(
        self, candidates: list[tuple[str, float]]
    ) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
        """对所有候选因子计算测试集指标（批量 D.features）。

        Args:
            candidates: [(expr_str, train_fitness), ...]

        Returns:
            (df, factor_series): df: 表达式的指标
                                 factor_series：dict. (expr, 表达式Series)，
                                 可直接传给 filter_by_correlation 避免重复计算
        """
        from qlib.data import D

        # 收集有效表达式（排除已知无效的）
        valid_candidates = [
            (expr, fit) for expr, fit in candidates
            if expr not in self.evaluator.invalid_exprs
        ]
        if not valid_candidates:
            return pd.DataFrame(), {}

        all_exprs = list(set(expr for expr, _ in valid_candidates))

        try:
            df_r = D.features(
                self.evaluator.instruments, all_exprs,
                start_time=self.config.train_end,
                end_time=self.config.end_date,
            )
        except Exception as e:
            err_msg = f"测试集批量 D.features 失败: %s，回退逐条评估: {e}"
            logger.error(err_msg)
            raise Exception(err_msg)

        # 逐个计算指标，同时保存 factor series
        rows = []
        factor_series = {}
        for expr_str, train_fitness in valid_candidates:
            factor = df_r[expr_str].dropna()
            if len(factor) == 0:
                continue

            common_idx = factor.index.intersection(self.evaluator.target_test.index)
            if len(common_idx) == 0:
                continue

            pred = factor.loc[common_idx]
            label = self.evaluator.target_test.loc[common_idx]
            pred = pred[np.isfinite(pred)]
            label = label[np.isfinite(label)]
            common_idx = pred.index.intersection(label.index)
            pred = pred.loc[common_idx]
            label = label.loc[common_idx]

            if len(pred) < 100:
                continue

            from qlib.contrib.eva.alpha import calc_ic
            ic, rank_ic = calc_ic(pred, label)

            # 保存 factor series（用于相关性计算）
            factor_series[expr_str] = pred

            # depth = self._expr_depth(expr_str)
            from deap import gp
            tree = gp.PrimitiveTree.from_string(expr_str, self.pset)
            rows.append({
                "expr": expr_str,
                "train_fitness": train_fitness,
                "train_ic": self.evaluator.get_train_ic(expr_str),
                "test_rank_ic": rank_ic.mean(),
                "test_icir": rank_ic.mean() / rank_ic.std() if rank_ic.std() > 1e-12 else 0.0,
                "depth": tree.height,
                "n_samples": len(pred),
                "ic_decay": self.evaluator.get_train_ic(expr_str) - rank_ic.mean(),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("无有效候选因子")
            return df, factor_series

        df["abs_test_rank_ic"] = df["test_rank_ic"].abs()
        df["abs_train_ic"] = df["train_ic"].abs()
        df = df.sort_values("abs_test_rank_ic", ascending=False).reset_index(drop=True)

        logger.info(
            "测试集评估: %d 个候选, |RankIC| mean=%.4f, max=%.4f",
            len(df), df["abs_test_rank_ic"].mean(), df["abs_test_rank_ic"].max(),
        )
        return df, factor_series

    # ================================================================
    # 低相关筛选
    # ================================================================

    def filter_by_correlation(
        self, df: pd.DataFrame, threshold: float = None,
        factor_series: dict[str, pd.Series] = None,
    ) -> pd.DataFrame:
        """贪心低相关筛选。

        Args:
            df: 因子指标 DataFrame
            threshold: 相关性阈值
            factor_series: expr→Series 映射，由 evaluate_all_on_test 返回，避免重复 D.features

        Returns:
            筛选后的 DataFrame（新增 max_corr, selected 列）
        """
        if threshold is None:
            threshold = self.config.corr_threshold

        if df.empty:
            return df

        # 直接用传入的 factor_series 构建相关性矩阵
        # matrix: 列名为 表达式名
        if factor_series and len(factor_series) >= 2:
            matrix = pd.DataFrame(factor_series)
            DataFrameIO.write(matrix, f'{self.config.output_dir}/factor_matrix.parquet', type='parquet')
        else:
            logger.warning("因子相关性矩阵构建失败或因子太少")
            df["max_corr"] = 0.0
            return df.assign(selected=True)

        # 计算相关性矩阵
        corr_matrix = matrix.corr(min_periods=30)
        PickleIO.write(corr_matrix, f'{self.config.output_dir}/factor_corr_matrix.pkl')

        # 按 |RankIC| 降序，确保是副本
        df = df.sort_values("abs_test_rank_ic", ascending=False).reset_index(drop=True)

        selected_indices = []
        max_corrs = []

        for i in range(len(df)):
            expr_i = df.iloc[i]["expr"]
            if expr_i not in corr_matrix.columns:
                max_corrs.append(0.0)
                continue

            # 第一个因子
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
        DataFrameIO.write(df, f'{self.config.output_dir}/test_factor_corr.parquet', type='parquet')
        return df

    # ================================================================
    # 报告生成
    # ================================================================

    def generate_report(
        self, df: pd.DataFrame
    ) -> str:
        """生成因子报告摘要
        """

        selected = df[df.get("selected", True)]

        lines = [
            "=" * 60,
            "GP + LLM 因子挖掘报告",
            "=" * 60,
            f"样本区间: [{self.config.start_date}, {self.config.end_date}]",
            f"训练集: [{self.config.start_date}, {self.config.train_end}]",
            f"测试集: ({self.config.train_end} ~ {self.config.end_date}]",
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

    # @staticmethod
    # def _expr_depth(expr_str: str) -> int:
    #     """估算表达式嵌套深度。"""
    #     depth = 0
    #     max_depth = 0
    #     for c in expr_str:
    #         if c == '(':
    #             depth += 1
    #             max_depth = max(max_depth, depth)
    #         elif c == ')':
    #             depth -= 1
    #     return max_depth
