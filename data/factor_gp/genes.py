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


def loop(args):
    if args.input is not None and Path(args.input).exists():
        df = DataFrameIO.read(args.input, type='csv')
        dic = df.to_dict(orient='list')
    else:
        dic = {'name': [], 'expr': [], 'desc': []}
    logger.info(f'历史 dic len: {len(dic["name"])}')

    llm = LLMInterface()
    registry = PrimitiveRegistry()
    registry.build_pset()
    evaluator = FactorEvaluator(pset=registry.pset)

    # 加载 Alpha158 因子作为种子
    factor_exprs = _load_alpha158()
    if not factor_exprs:
        logger.warning("未加载到 Alpha158 因子，跳过基因提取")
        return

    for i in range(args.n_loop):
        logger.info('\n{}\nLoop: {}...'.format('-' * 50, i+1))
        # LLM 提取子表达式基因
        genes_ori = llm.extract_sub_expr_genes(
            factor_exprs, n_target=args.n_target)
        if not genes_ori:
            return
        logger.info(f'genes_ori len: {len(genes_ori)}')

        for name, payload in genes_ori.items():
            expr = payload['expr']
            desc = payload['desc']
            if expr == "" or desc == "":
                logger.warning(f'name: {name}, expr: {expr}, desc: {desc}')
                continue
            if expr in set(dic['expr']):
                continue
            try:
                tree = gp.PrimitiveTree.from_string(expr, registry.pset)
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
        logger.info(f'dic len: {len(dic["name"])}')

    df = pd.DataFrame(dic)
    DataFrameIO.write(df, args.output, type='csv', index=False)

def main():
    parser = argparse.ArgumentParser(description="因子基因生成脚本")
    parser.add_argument('--input', type=str, default=None, help="已经存在的因子基因文件路径")
    parser.add_argument('--output', type=str, default='./out_genes.csv',help="因子基因文件路径")
    parser.add_argument('--n-loop', type=int, default=1, help="循环次数")
    parser.add_argument('--n-target', type=int, default=70, help="目标因子数量")
    args = parser.parse_args()
    logger.info('args: {}'.format(args))

    loop(args)

if __name__ == '__main__':
    main()




