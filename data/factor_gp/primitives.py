"""算子/终端/子表达式基因注册中心。

管理 GP 搜索空间的三类构建块：
1. 基础终端：$close, $open, $high, $low, $volume, $amount
2. 基础算子：Abs, Log, Sign, Add, Sub, Mul, Div, Max, Min, Gt, Lt, Ref, Mean, Std, Delta, Rank, WMA, EMA, Corr, If
3. 子表达式基因：LLM 提取的可复用子表达式

设计要点：
- 子表达式基因作为终端注册（不是算子），避免 DEAP 强类型系统的 arity 匹配问题
- 子表达式基因的终端名称即为 qlib 表达式字符串，嵌入树字符串后 qlib 可直接计算
"""
import copy
import random
from functools import partial

from deap import gp

from .conf import ELEM_OPS, PIRE_OPS, ELEM_ROLLING_OPS, PAIR_ROLLING_OPS, OTHER_OPS
from .conf import BASE_TERMINALS


class PrimitiveRegistry:
    """算子注册中心。

    子表达式基因作为终端注册：
    - 终端 name = qlib 表达式字符串（嵌入树字符串后 qlib 可直接计算）
    - 保留别名映射用于可读性展示
    """

    def __init__(self):
        self.pset: gp.PrimitiveSetTyped | None = None
        self.pset_check: gp.PrimitiveSetTyped | None = None  # 注册量价数据，用于检验表达式语法
        self.sub_expr_genes: dict[str, str] = {}  # 子表达式基因. alias → qlib_expr
        self._gene_counter: int = 0    #  用于记录子表达式基因数量

    # ================================================================
    # pset 构建
    # ================================================================

    def build_pset(self, extra_terminals: bool = False) -> gp.PrimitiveSetTyped:
        """构建完整的 PrimitiveSetTyped。

        Args:
            extra_terminals: 是否包含 $change 等扩展终端
        """
        pset = gp.PrimitiveSetTyped("MAIN", [], float, 0)

        # ---- 终端 ----
        # terminals = list(BASE_TERMINALS)
        # if extra_terminals:
        #     terminals.extend(EXTRA_TERMINALS)
        #
        # for f in terminals:
        #     pset.addTerminal(f, ret_type=float, name=f)

        # 叶子常量
        pset.addEphemeralConstant("C", partial(random.uniform, 0, 1), ret_type=float)
        pset.addEphemeralConstant("N", partial(random.randint, 5, 30), ret_type=int)

        # ---- 算子 ----
        _dummy = lambda *a: None

        # Element-wise (arity=1)
        for op in ELEM_OPS:
            pset.addPrimitive(_dummy, in_types=[float], ret_type=float, name=op)

        # Pair-wise element (arity=2)
        for op in PIRE_OPS:
            pset.addPrimitive(_dummy, in_types=[float, float], ret_type=float, name=op)

        # Rolling element (arity=2: float, int)
        for op in (set(ELEM_ROLLING_OPS) - set(['Quantile'])):
            pset.addPrimitive(_dummy, in_types=[float, int], ret_type=float, name=op)

        # Quantile (arity=3: float, int, float)
        pset.addPrimitive(_dummy, in_types=[float, int, float], ret_type=float, name="Quantile")

        # Pair Rolling (arity=3: float, float, int)
        for op in PAIR_ROLLING_OPS:
            pset.addPrimitive(_dummy, in_types=[float, float, int], ret_type=float, name=op)

        # If (arity=3: float, float, float)
        if OTHER_OPS:
            for op in OTHER_OPS:
                pset.addPrimitive(_dummy, in_types=[float, float, float], ret_type=float, name=op)

        # 类型转换：float → int（Rolling 算子动态窗口必须）
        pset.addPrimitive(_dummy, [float], int, name="IntCast")

        self.pset = pset

        self.pset_check = copy.deepcopy(pset)
        # 注册量价数据
        terminals = list(BASE_TERMINALS)
        for f in terminals:
            self.pset_check.addTerminal(f, ret_type=float, name=f)

        return pset

    # ================================================================
    # 子表达式基因管理
    # ================================================================

    def register_sub_expr(self, alias: str, qlib_expr: str):
        """注册一个子表达式基因。

        Args:
            alias: 可读别名，例如 "bearish_reversal_flag"
            qlib_expr: qlib 表达式字符串，例如 "Greater($open/Ref($close,1)-1,0)*Greater($open/$close-1,0)"
        """
        if alias in self.sub_expr_genes:
            raise ValueError(f"基因别名已存在: {alias}")
        self.sub_expr_genes[alias] = qlib_expr
        self._gene_counter += 1

        # 如果 pset 已构建，动态添加终端
        if self.pset is not None:
            self._add_gene_terminal(self.pset, qlib_expr)

    def register_sub_exprs(self, genes: dict[str, str]):
        """批量注册子表达式基因。"""
        for alias, expr in genes.items():
            self.register_sub_expr(alias, expr)

    def _add_gene_terminal(self, pset: gp.PrimitiveSetTyped, qlib_expr: str):
        """将子表达式基因注册为终端。

        终端 name 设为 qlib 表达式字符串，嵌入树字符串后 qlib 可直接计算。
        """
        safe_expr = f"({qlib_expr})"
        pset.addTerminal(safe_expr, ret_type=float, name=safe_expr)

    @property
    def gene_count(self) -> int:
        return self._gene_counter

    def get_gene_aliases(self) -> list[str]:
        return list(self.sub_expr_genes.keys())
