"""GP + LLM 因子挖掘 — 全局配置"""

from dataclasses import dataclass, field
from config import PROVIDER_URI

@dataclass
class GPConfig:
    # ========== 数据 ==========
    start_date: str = "2018-01-01"
    end_date: str = "2026-05-11"
    train_end: str = "2022-12-31"  # 训练/测试分割点
    market: str = "csi300"

    # ========== GP 参数 ==========
    n_pop: int = 600  # 每岛种群大小（报告：1800/3 岛）
    n_gen: int = 20  # 总进化代数
    n_islands: int = 3  # 岛屿数量
    n_hof: int = 50  # 每个岛屿筛选表达式数量
    cxpb: float = 0.7
    mutpb: float = 0.3
    max_height: int = 8
    tournsize: int = 3    # 竞赛候选数

    # ========== 岛间迁移 ==========
    migration_freq: int = 4  # 每 N 代迁移一次
    migration_n: int = 1  # 每次每岛迁出个体数

    # ========== LLM 注入 ==========
    llm_inject_freq: int = 3  # 每 N 代 LLM 生成一次
    llm_inject_n: int = 5  # 每次每岛生成数
    enable_economic_check: bool = True  # 开关：是否启用 LLM 经济学含义检查
    llm_check_freq: int = 3  # 每 N 代执行一次经济学含义检查
    # LLM API 配置（model/base_url/api_key）从 config.yaml llm_deepseek 读取

    # ========== 适应度 ==========
    complexity_penalty: float = 0.005  # 表达式深度惩罚系数
    icir_weight: float = 0.5  # ICIR 在 fitness 中的权重

    # ========== 筛选 ==========
    corr_threshold: float = 0.70  # 相关性阈值
    fitness_threshold: float = 0.03

    # ========== 路径 ==========
    output_dir: str = "data"

    # ========== 随机种子 ==========
    seed: int = 42

    # ========== qlib ==========
    provider_uri: str = PROVIDER_URI
    kernels: int = 1     # qlib并行数 / 因子fitness计算并行数


@dataclass
class EvolutionState:
    """进化过程中需要持久化的状态，支持断点恢复"""

    generation: int = 0
    cache: dict = field(default_factory=dict)  # expr_str → fitness
    invalid_exprs: set = field(default_factory=set)
    llm_history: list = field(default_factory=list)  # LLM 生成历史


# 基础终端（qlib 字段引用）
BASE_TERMINALS = [
    "$close", "$open", "$high", "$low", "$volume", "$amount",
]

# 可选的扩展终端
EXTRA_TERMINALS = [
    "$change",  # 涨跌幅
]

# 单元素 算子（4个）"Not"
ELEM_OPS = ["Abs", "Sign", "Log"]

# 元素对 算子(15个)  'Gt', 'Ge', 'Lt', 'Le', 'Eq', 'Ne', 'And', 'Or'
PIRE_OPS = ['Power', 'Add', 'Sub', 'Mul', 'Div',
            'Greater', 'Less']

# 单元素 Rolling 算子 (22个) 'IdxMax', 'IdxMin'
ELEM_ROLLING_OPS = ['Ref', 'Mean', 'Sum', 'Std', 'Var', 'Skew', 'Kurt',
                    'Max',  'Min', 'Quantile', 'Med', 'Mad',
                    'Rank', 'Count', 'Delta', 'Slope', 'Rsquare','Resi', 'WMA', 'EMA']

# 元素对 Rolling 算子（2个）
PAIR_ROLLING_OPS = [
    'Corr', 'Cov'
]

# 其他（1个） "If",
OTHER_OPS = [

]

