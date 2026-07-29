""" 因子基因生成脚本 """
import argparse

from data.factor_gp.llm import LLMInterface
from data.factor_gp.primitives import PrimitiveRegistry
from data.factor_gp.evaluate import FactorEvaluator
from utils import DataFrameIO
from deap import gp
from utils import LoggerFactory
import pandas as pd
from pathlib import Path

logger = LoggerFactory.get_logger(__name__)


def _load_alpha158() -> list[tuple[str, str]]:
    """从 Alpha158 加载因子表达式。"""
    try:
        from qlib.contrib.data.loader import Alpha158DL
        fields, names = Alpha158DL.get_feature_config()
        return [(name, expr) for name, expr in zip(names, fields)]
    except Exception as e:
        logger.warning("Alpha158 加载失败: %s", e)
        return []


def _build_gap_hint(dic: dict, pset) -> str:
    """从已有基因统计复合比例，生成 LLM 缺口提示。"""
    n = len(dic['name'])
    if n == 0:
        return ""

    compound = 0
    for expr in dic['expr']:
        if FactorEvaluator.classify_gene_complexity(expr, pset) == 'compound':
            compound += 1
    ratio = compound / n

    hints = [f"当前已有 {n} 个基因，其中复合基因{compound}个（{ratio:.0%}），简单基因{n-compound}个（{1-ratio:.0%}）。"]
    if ratio < 0.4:
        hints.append("复合基因严重不足，本轮请重点生成包含多字段组合或嵌套算子的复合结构基因。")
    elif ratio < 0.5:
        hints.append("复合基因仍不足（目标≥50%），请优先生成复合结构基因。")
    return " ".join(hints)


def _log_stats(dic: dict, pset, tag: str = ""):
    """输出基因统计摘要。"""
    n = len(dic['name'])
    if n == 0:
        logger.info(f"[{tag}] 基因池为空")
        return

    compound = sum(1 for e in dic['expr']
                   if FactorEvaluator.classify_gene_complexity(e, pset) == 'compound')
    roles = pd.Series(dic['role']).value_counts().to_dict()

    logger.info(f"[{tag}] 总计 {n} 个基因 | "
                f"复合 {compound}（{compound/n:.0%}）| "
                f"简单 {n-compound}（{(n-compound)/n:.0%}）| "
                f"role: {roles}")


def loop(args):
    if args.input is not None and Path(args.input).exists():
        df = DataFrameIO.read(args.input, type='csv')
        dic = df.to_dict(orient='list')
    else:
        dic = {'name': [], 'expr': [], 'desc': [], 'role': []}
    logger.info(f'历史 dic len: {len(dic["name"])}')

    llm = LLMInterface()
    registry = PrimitiveRegistry()
    registry.build_pset()
    evaluator = FactorEvaluator(pset=registry.pset_check)

    # 加载 Alpha158 因子作为种子
    factor_exprs = _load_alpha158()
    if not factor_exprs:
        logger.warning("未加载到 Alpha158 因子，跳过基因提取")
        return

    # 输出已有基因统计
    if len(dic['name']) > 0:
        _log_stats(dic, registry.pset_check, tag="input")

    for i in range(args.n_loop):
        logger.info('\n{}\nLoop: {}...'.format('-' * 50, i+1))
        # 基于当前基因池构建缺口提示
        gap_hint = _build_gap_hint(dic, registry.pset_check)
        if gap_hint:
            logger.info(f'gap_hint: {gap_hint}')

        # LLM 提取子表达式基因
        genes_ori = llm.extract_sub_expr_genes(
            factor_exprs, n_target=args.n_target, gap_hint=gap_hint)
        if not genes_ori:
            return
        logger.info(f'genes_ori len: {len(genes_ori)}')

        new_cnt = 0
        for name, payload in genes_ori.items():
            expr = payload['expr']
            desc = payload['desc']
            role = payload['role']
            if expr == "" or desc == "" or role == "":
                logger.warning(f'some fields is None. name: {name}, expr: {expr}, desc: {desc}, role: {role}')
                continue
            if expr in set(dic['expr']):
                continue
            try:
                tree = gp.PrimitiveTree.from_string(expr, registry.pset_check)
            except:
                logger.info('不合法：{}, 理由: {}'.format(expr, '不符合 deap 语法格式'))
                continue
            ok, reason =  evaluator._check_qlib_semantics(tree)
            if not ok:
                logger.info("不合法: {}, 理由：{}".format(expr, reason))
                continue

            dic['name'].append(name)
            dic['expr'].append(expr)
            dic['desc'].append(desc)
            dic['role'].append(role)
            new_cnt += 1

        logger.info(f'本轮新增 {new_cnt} 个，累计 {len(dic["name"])} 个')
        _log_stats(dic, registry.pset_check, tag=f"loop{i+1}")

    df = pd.DataFrame(dic)
    DataFrameIO.write(df, args.output, type='csv', index=False)

def main():
    parser = argparse.ArgumentParser(description="因子基因生成脚本")
    parser.add_argument('--input', type=str, default=None, help="已经存在的因子基因文件路径")
    parser.add_argument('--output', type=str, default='./out_genes.csv',help="因子基因文件路径")
    parser.add_argument('--n-loop', type=int, default=1, help="循环次数")
    parser.add_argument('--n-target', type=int, default=100, help="目标因子数量")
    args = parser.parse_args()
    logger.info('args: {}'.format(args))

    loop(args)

if __name__ == '__main__':
    main()




