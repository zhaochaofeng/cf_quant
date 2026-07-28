"""GP + LLM 因子挖掘主流水线。

串联四个 Phase：
  Phase 0: qlib 初始化 → 数据加载 → pset 构建
  Phase 1: LLM 子表达式基因提取（可选）
  Phase 2: 分岛进化
  Phase 3: 后处理筛选 + 报告
"""

import os
import argparse
import time

import numpy as np
import pandas as pd
from qlib.data import D

from data.factor_gp.conf import GPConfig
from data.factor_gp.evaluate import FactorEvaluator
from data.factor_gp.evolution import IslandEvolution
from data.factor_gp.primitives import PrimitiveRegistry
from data.factor_gp.screening import FactorScreening
from utils.io_utils import DataFrameIO
from utils.logger import LoggerFactory
from utils import init_qlib
from utils.dt import time_decorator

logger = LoggerFactory.get_logger(__name__)


class GPLlmPipeline:
    """GP + LLM 因子挖掘主流水线"""

    def __init__(self, config: GPConfig, enable_llm: bool = True):
        self.config = config
        self.enable_llm = enable_llm
        self.registry = PrimitiveRegistry()             # 注册算子
        self.evaluator: FactorEvaluator | None = None   # 因子评估器
        self.llm = None
        self._instruments = None     # 股票集合
        self._target_train = None    # 训练集收益率
        self._target_test = None     # 测试集收益率

        os.makedirs(self.config.output_dir, exist_ok=True)

    # ================================================================
    # 主入口
    # ================================================================

    def run(self) -> pd.DataFrame:
        """执行完整流水线，返回筛选后的因子 DataFrame。"""
        t_start = time.time()

        # Phase 0
        self._phase0_init()
        self._build_pset()

        # 构建 evaluator
        self.evaluator = FactorEvaluator(
            instruments=self._instruments,
            target_train=self._target_train,
            target_test=self._target_test,
            config=self.config,
            pset=self.registry.pset_check,
        )

        # Phase 1
        self._phase1_llm_genes()

        # Phase 2
        result = self._phase2_evolve()

        # Phase 3
        df = self._phase3_screening(result)

        logger.info("流水线完成，总耗时: %.1fs", time.time() - t_start)
        return df

    # ================================================================
    # Phase 0: 初始化
    # ================================================================
    def _phase0_init(self):
        """qlib 初始化 + 行情数据加载 + train/test 划分。"""
        logger.info("=" * 60)
        logger.info("Phase 0: 初始化")

        # qlib 初始化
        init_qlib(provider_uri=self.config.provider_uri)
        logger.info("qlib 初始化完成: %s, kernels=%d",
                     self.config.provider_uri, self.config.kernels)

        # 获取股票列表
        cfg = D.instruments(market=self.config.market)
        self._instruments = D.list_instruments(
            cfg, start_time=self.config.start_date, end_time=self.config.end_date)

        # 加载收盘价数据
        df = D.features(
            self._instruments, ['$close'],
            start_time=self.config.start_date, end_time=self.config.end_date)

        # 计算未来收益（T+1 执行滞后）
        target = (df['$close'].groupby('instrument', group_keys=False)
                  .apply(lambda x: x.shift(-2) / x.shift(-1) - 1))
        target.dropna(inplace=True)
        target.name = 'return'

        # train/test 划分
        self._target_train = target[target.index.get_level_values('datetime')
                                    <= self.config.train_end]
        self._target_test = target[target.index.get_level_values('datetime')
                                   > self.config.train_end]
        DataFrameIO.write(self._target_train.to_frame(), f'{self.config.output_dir}/target_train.parquet')
        DataFrameIO.write(self._target_test.to_frame(), f'{self.config.output_dir}/target_test.parquet')
        train_len = len(self._target_train.index.get_level_values('datetime').unique())
        test_len = len(self._target_test.index.get_level_values('datetime').unique())
        logger.info("数据加载: %d instruments, train=%d, test=%d",
                     len(self._instruments), train_len, test_len)
        if train_len == 0 or test_len == 0:
            raise Exception('train_len == 0 or test_len == 0')

    def _build_pset(self):
        """构建 PrimitiveSetTyped。"""
        self.registry.build_pset()

    # ================================================================
    # Phase 1: LLM 子表达式基因提取
    # ================================================================
    def _phase1_llm_genes(self):
        """LLM 从初始优质因子提取子表达式基因，注册到 pset"""
        logger.info("=" * 60)
        logger.info("Phase 1: LLM 子表达式基因提取 ...")

        # FIXME: 后续可以删除
        from data.factor_gp.llm import LLMInterface
        self.llm = LLMInterface()

        '''
        # 加载 Alpha158 因子作为种子
        factor_exprs = self._load_alpha158()
        if not factor_exprs:
            logger.warning("未加载到 Alpha158 因子，跳过基因提取")
            return

        # LLM 提取子表达式基因
        genes_ori = self.llm.extract_sub_expr_genes(factor_exprs)
        if not genes_ori:
            return

        # 保存含经济学描述的基因到本地便于查看（须在下面转换前保存，否则 desc 丢失）
        from utils import PickleIO
        PickleIO.write(genes_ori, f'{self.config.output_dir}/llm_genes.pkl')

        # 检查genes 合法性
        from deap import gp
        genes = {}
        for alias, payload in genes_ori.items():
            expr_str = payload["expr"]
            try:
                tree = gp.PrimitiveTree.from_string(expr_str, self.registry.pset)
            except:
                logger.info('不合法：{}, 理由: {}'.format(expr_str, '不符合 deap 语法格式'))
                continue
            ok, reason = self.evaluator._check_qlib_semantics(tree)
            if not ok:
                logger.info("不合法: {}, 理由：{}".format(expr_str, reason))
                continue
            else:
                genes[alias]= expr_str

        # genes = {
        #     alias: payload["expr"]
        #     for alias, payload in genes.items()
        #     if isinstance(payload, dict)
        #     and self.evaluator._passes_qlib_semantic(payload["expr"])
        # }
        # print(genes)

        if not genes:
            logger.warning("LLM 提取的因子 genes 不合法，跳过注册")
            return
        '''

        genes_df = DataFrameIO.read('./factor_genes.csv', type='csv')
        if genes_df.empty:
            raise Exception('未找到因子基因文件')

        genes_df.drop(columns=['desc'], inplace=True)
        genes = {}
        for i, row in genes_df.iterrows():
            genes[row['name']] = row['expr']

        # 注册到 pset
        self.registry.register_sub_exprs(genes)
        self.llm.set_gene_aliases(self.registry.get_gene_aliases())

        logger.info("子表达式基因注册完成: %d 个基因", len(genes))
        for alias, expr in genes.items():
            logger.info('{} -> {}'.format(alias, expr))

    def _load_alpha158(self) -> list[tuple[str, str]]:
        """从 Alpha158 加载因子表达式。"""
        try:
            from qlib.contrib.data.loader import Alpha158DL
            fields, names = Alpha158DL.get_feature_config()
            return [(name, expr) for name, expr in zip(names, fields)]
        except Exception as e:
            logger.warning("Alpha158 加载失败: %s", e)
            return []

    # ================================================================
    # Phase 2: 分岛进化
    # ================================================================
    @time_decorator
    def _phase2_evolve(self):
        """运行分岛进化。"""
        logger.info("=" * 60)
        logger.info("Phase 2: 分岛进化 ...")

        # 随机种子（resume 时由 _load_checkpoint 恢复，不重新设置）
        if not self.config.resume_from:
            import random
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        # 启动进化
        engine = IslandEvolution(
            pset=self.registry.pset,
            evaluator=self.evaluator,
            config=self.config,
            llm_interface=self.llm,
        )
        result = engine.run()

        # 暂存基因别名（Phase 3 报告可用）
        self._gene_aliases = self.registry.get_gene_aliases()

        return result

    # ================================================================
    # Phase 3: 后处理
    # ================================================================
    @time_decorator
    def _phase3_screening(self, result) -> pd.DataFrame:
        """测试集评估 + 低相关筛选 + 报告 + 持久化。"""
        logger.info("=" * 60)
        logger.info("Phase 3: 后处理筛选")

        import os
        from datetime import datetime
        from utils import PickleIO
        PickleIO.write(result.candidates, f'{self.config.output_dir}/result_candi.pkl')

        screening = FactorScreening(self.evaluator, self.config, self.registry.pset)
        logger.info('训练过程中重复的 fitness 值因子个数: {}'.format(len(self.evaluator.fitness_set)))

        # 通过 fitness 阈值筛选因子
        candidates = []
        fitness_set = set([])  # 相同 fitness 的因子只保留一个
        threshold_count = 0
        fitness_repeat_count = 0

        for expr, fitness in result.candidates:
            if fitness in fitness_set:
                fitness_repeat_count += 1
                continue
            # if fitness < self.config.fitness_threshold:
            #     threshold_count += 1
            #     continue
            fitness_set.add(fitness)

            train_metrics = self.evaluator.get_train_ic_metrics(expr)
            ric = train_metrics['ric']
            if abs(ric) < self.config.ric_threshold:
                threshold_count += 1
                continue
            candidates.append((expr, fitness))
        logger.info('总因子数：{}'.format(len(result.candidates)))
        logger.info('fitness 重复值 过滤因子个数: {}'.format(fitness_repeat_count))
        logger.info('ric threshold 过滤的因子个数: {}'.format(threshold_count))
        logger.info('经过筛选后的因子个数: {}'.format(len(candidates)))

        # 测试集评估（返回 factor_series 供相关性计算复用）
        df, factor_series = screening.evaluate_all_on_test(candidates)
        if df.empty:
            logger.warning("无有效候选因子")
            return df
        # 附加经济学含义描述
        if result.economic_descs:
            df["economic_desc"] = df["expr"].map(result.economic_descs)
        DataFrameIO.write(df, f'{self.config.output_dir}/test_eval.parquet', type='parquet')
        PickleIO.write(factor_series, f'{self.config.output_dir}/test_factor.pkl')

        # 低相关筛选（复用 factor_series，避免重复 D.features）
        df = screening.filter_by_correlation(df, factor_series=factor_series)

        # 生成报告
        report = screening.generate_report(df)

        # ---- 持久化 ----
        output_dir = self.config.output_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 因子结果 CSV
        csv_path = os.path.join(output_dir, f"factors_{ts}.csv")
        # 报告中的表达式列可能很长，保存时处理
        save_df = df.copy()
        if "ic_series" in save_df.columns:
            save_df = save_df.drop(columns=["ic_series"], errors="ignore")
        DataFrameIO.write(save_df, csv_path, type="csv")
        logger.info("因子结果已保存: %s", csv_path)

        # 报告文本
        report_path = os.path.join(output_dir, f"report_{ts}.txt")
        with open(report_path, "w") as f:
            f.write(report)
        logger.info("报告已保存: %s", report_path)

        # 进化日志 JSON
        log_path = os.path.join(output_dir, f"evolution_{ts}.json")
        import json
        log_data = {}
        for i, logbook in enumerate(result.logbooks):
            log_data[f"island_{i}"] = list(logbook)
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2, default=str)
        logger.info("进化日志已保存: %s", log_path)

        return df


# ================================================================
# CLI 入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='GP + LLM 因子挖掘')
    parser.add_argument('--start-date', type=str, default='2025-01-01')
    parser.add_argument('--end-date', type=str, default='2026-05-11')
    parser.add_argument('--train-end', type=str, default='2026-01-31')
    parser.add_argument('--market', type=str, default='csi300')
    parser.add_argument('--n-pop', type=int, default=10, help='每岛种群大小')
    parser.add_argument('--n-gen', type=int, default=4, help='进化代数')
    parser.add_argument('--n-islands', type=int, default=3, help='岛屿数量')
    parser.add_argument('--n-hof', type=int, default=1000, help='每个岛屿筛选表达式数量')
    parser.add_argument('--no-llm', action='store_true', help='禁用 LLM')
    parser.add_argument('--kernels', type=int, default=max(min(os.cpu_count() - 2, 10), 1), help='qlib kernels')
    parser.add_argument('--ric-threshold', type=float, default=0.001, help='因子指标阈值')
    parser.add_argument('--complexity_penalty', type=float, default=0.001, help='表达式深度惩罚系数')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-economic-check', action='store_true', help='禁用经济学含义过滤')
    parser.add_argument('--checkpoint-freq', type=int, default=2, help='每隔 N 代保存 checkpoint，0=禁用')
    parser.add_argument('--resume-from', type=str, default='', help='从指定 checkpoint 文件恢复训练')

    args = parser.parse_args()
    logger.info(f'args: {args}')

    config = GPConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        train_end=args.train_end,
        market=args.market,
        n_pop=args.n_pop,
        n_gen=args.n_gen,
        n_islands=args.n_islands,
        n_hof=args.n_hof,
        kernels=args.kernels,
        ric_threshold=args.ric_threshold,
        complexity_penalty=args.complexity_penalty,
        seed=args.seed,
        enable_economic_check=not args.no_economic_check,
        checkpoint_freq=args.checkpoint_freq,
        resume_from=args.resume_from,
    )

    pipeline = GPLlmPipeline(config, enable_llm=not args.no_llm)
    df = pipeline.run()

    if not df.empty:
        selected = df[df.get("selected", True)]
        print(f"\n最终筛选结果: {len(selected)}/{len(df)} 个因子")


if __name__ == '__main__':
    main()
