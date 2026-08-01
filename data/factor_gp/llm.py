"""LLM 接口层 — 子表达式基因提取 + 周期性因子注入。

通过 config.yaml 中的 llm_deepseek 配置对接 DeepSeek API（OpenAI 兼容）。

设计原则：
1. 提示词包含完整的算子列表 + 字段语义 + 子表达式基因列表
2. 要求 LLM 返回结构化输出（JSON），便于解析
3. 失败案例驱动：将无效表达式填入 prompt，避免 LLM 重复错误
4. 竞争替换：LLM 生成候选 → qlib 计算 IC → 仅 IC 优于底部个体才进入种群
"""

import json
import re
import time

import httpx
from typing import Optional

from utils import get_config
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)

# ================================================================
# 提示词模板
# ================================================================

# qlib 算子参考（供 LLM 理解可用操作）
#
# 以下算子已从搜索空间移除（bool/逻辑族：信息损失+过拟合风险，见第12章四方针），
# 不在 QLIB_OPERATORS_REF 中暴露给 LLM，LLM 不应生成：
#   单目: Not                （bitwise_not，逻辑非，仅适用 bool）
#   双目: Gt,Ge,Lt,Le,Eq,Ne  （比较返回 0/1，丢失偏离幅度，压低 IC）
#         And,Or             （bitwise_and/or，多重布尔叠加=过拟合温床）
#   三元: If                 （需 bool 条件；纯连续空间改用 Greater/Less/Sign 软条件）
QLIB_OPERATORS_REF = """
## 可用算子

### 单目算子 (float → float)
- Abs(x): 绝对值
- Log(x): 自然对数
- Sign(x): 符号函数

### 双目算子 (float, float → float)
- Add(x, y): x + y
- Sub(x, y): x - y
- Mul(x, y): x * y
- Div(x, y): x / y
- Power(x, y): x 的 y 次幂
- Greater(x, y): 取 x 和 y 中的较大值
- Less(x, y): 取 x 和 y 中的较小值
- Gt(x, y): x > y 比较，返回 1.0（真）或 0.0（假），布尔型 0/1 输出
- Lt(x, y): x < y 比较，返回 1.0（真）或 0.0（假），布尔型 0/1 输出

### 滚动算子 (float, int → float)
- Ref(x, d): d 日前的 x 值
- Mean(x, d): d 日滚动均值
- Sum(x, d): d 日滚动求和
- Std(x, d): d 日滚动标准差
- Var(x, d): d 日滚动方差
- Skew(x, d): d 日滚动偏度
- Kurt(x, d): d 日滚动峰度
- Max(x, d): d 日滚动最大值
- IdxMax(x, d): d 日内最大值出现的位置（1-based，最近=1），即 Aroon-Up 分量
- Min(x, d): d 日滚动最小值
- IdxMin(x, d): d 日内最小值出现的位置（1-based，最近=1），即 Aroon-Down 分量
- Med(x, d): d 日滚动中位数
- Mad(x, d): d 日滚动平均绝对偏差
- Rank(x, d): d 日内 x 的排名百分位
- Count(x, d): d 日内非 NaN 元素个数
- Delta(x, d): x - Ref(x, d)
- Slope(x, d): d 日滚动线性回归斜率
- Rsquare(x, d): d 日滚动线性回归 R²
- Resi(x, d): d 日滚动线性回归残差
- WMA(x, d): d 日加权移动平均
- EMA(x, d): d 日指数移动平均

### 特殊滚动算子
- Quantile(x, d, q): d 日滚动 q 分位数（q 为 0-1 之间的浮点数，如 0.5 表示中位数）

### 双变量滚动算子 (float, float, int → float)
- Corr(x, y, d): x 与 y 在 d 日内的滚动相关系数
- Cov(x, y, d): x 与 y 在 d 日内的滚动协方差

### 可用字段
- $close: 收盘价
- $open: 开盘价
- $high: 最高价
- $low: 最低价
- $volume: 成交量
- $amount: 成交额

### 注意事项
- 所有表达式必须用英文逗号和括号
- 滚动窗口参数必须是整数常量
- 运算符名称大小写敏感，首字母大写
"""

SUB_EXPR_PROMPT_TEMPLATE = """你是一位量化金融研究员，擅长从量价因子中提取可复用的子表达式（"因子基因"）。

{operators_ref}

## 任务
分析以下优质量价因子表达式，提取可复用的子表达式基因。基因是 GP 搜索空间的叶子节点，GP 用算子在基因上组合搜索完整因子。

## 基因设计原则
1. 金融语义原子性：一个基因且仅刻画一种具体的交易行为或量价状态（如“高开阴线”、“极端放量”、“隔夜波动占比”）。不能把“高开后回落+放量”揉在一起，那是两个基因的协同工作。
2. 表达式深度（Deep）≤ 5层：若基因本身深度已达6~7层，GP再往上叠加2~3层，总深度极易超过10层，该因子会出现“结构臃肿”。
3. 可被LLM自然命名：研究员或LLM能用一个金融名词（如 bearish_reversal_flag）精准概括其含义，而非只能用公式描述。这确保了LLM在周期性注入时，能理解“用这个基因去组合什么”。
4. 语法必须用 DEAP 前缀格式：算子用 Add, Sub, Mul, Div, Greater, Less, Power, Ref, EMA, Std, Corr, Cov 等名称，**严禁使用 +, -, *, /, <, > 等中缀符号**（解析器无法识别）。
5. 量纲一致：加减两侧需同量纲（价格±价格、收益率±收益率），禁止 Add($close, $volume) 这类量纲不匹配。
6. 算子参数约束：Power 指数必须是浮点字面量 0.5、2.0 或 3.0 之一（写整数 2 会被类型检查拒绝）；滚动窗口参数必须是 int 常量（建议 5-20）；除法分母可加 1e-12 防零除。
7. 去冗余：确保基因之间信息有差异，避免重复提取语义相近的子表达式，避免生成与已经存在的基因重复。
8. 英文命名。

## 建议覆盖的基因类别（尽量兼顾多类，勿集中在单一类）
- 收益率类：日收益、隔夜收益、日内收益、收益波动
- 量能变化类：超额放量、量比、天量异动
- K线形态类：实体比例、上下影线、高开阴线
- 价格位置类：收盘在区间位置、近期高低点相对位置
- 波动率类：滚动标准差、隔夜/日内波动比、振幅
- 趋势类：线性斜率、趋势线性度、趋势偏离、距高低点天数
- 路径效率类：净变动/累计波动、路径平滑度
- 量价协同类：收益与放量相关性、量价协方差、量价背离

## 基因功能维度（每个基因按在组合中的角色标注）
- state_detection：状态识别型——判断市场处于什么环境中（如在不在高位、波动是大是小、是否放量），通常作 GP 组合的"条件开关"或环境调制项
- response_intensity：响应强度型——衡量特定行为发生时的激烈程度（如涨跌幅度、放量程度、影线长度），通常作 GP 组合的"乘数"或"惩罚项"
每个基因需在 role 字段中标注所属功能维度。

## 反例（不要提取）
- 过简：Div($close, $open)、Sub($high, $low)、Mean($close, 20) 单独使用
- 量纲不匹配：Add($close, $volume)
- 近乎恒等：Corr($close, $high, 10)
- 完整因子：基因是构件，不是可直接选股的成品
- 笛卡尔积枚举：对同一算子+字段枚举全部窗口（如 Mean($close,5), Mean($close,10), Mean($close,20), Mean($close,60) 不得同时出现）

## 参考基因示例（均为合法 DEAP 前缀格式）
- Mul(Greater(Sub(Div($open, Ref($close, 1)), 1), 0), Greater(Sub(Div($open, $close), 1), 0)) —— 高开阴线
- Power(Greater(Sub(Div($volume, EMA($volume, 20)), 1.0), 0.0), 2.0) —— 极端放量
- Div(Std(Sub(Div($open, Ref($close, 1)), 1), 20), Std(Sub(Div($close, $open), 1), 20)) —— 隔夜波动占比

## 参考价量因子
{factors}

## 已经存在的基因
{genes}

## 返回格式
只返回一个 JSON 对象，key 是子表达式名称，value 是 {{"expr": ..., "desc": ..., "role": ...}}。
- expr: DEAP 前缀格式的 qlib 表达式
- desc: 中文金融含义。包含因子所在类别、金融逻辑描述及所刻画的市场行为，不超过 80 字
- role: 基因功能维度，取值 "state_detection"（状态识别型）或 "response_intensity"（响应强度型）

示例：
{{
  "bearish_reversal_flag": {{"expr": "Mul(Greater(Sub(Div($open, Ref($close, 1)), 1), 0), Greater(Sub(Div($open, $close), 1), 0))", "desc": "K线形态类-高开阴线：高开幅度与日内跌幅的乘积，反映隔夜情绪或开盘冲动被日内卖压主导后的短期反转压力", "role": "state_detection"}},
  "volume_surge_squared": {{"expr": "Power(Greater(Sub(Div($volume, EMA($volume, 20)), 1), 0), 2.0)", "desc": "量能变化类-超额放量平方：仅对超出均量的部分做非线性放大，捕捉极端放量背后的交易拥挤与情绪集中释放", "role": "response_intensity"}}
}}

建议提取 {n_target} 个左右、覆盖上述多类别的基因。

只返回 JSON，不要加其他文字说明。"""

ECONOMIC_CHECK_PROMPT_TEMPLATE = """你是一位量化金融研究员，擅长判断量价因子的经济学含义。

{operators_ref}

## 任务
判断以下每个因子表达式是否具有可解释的经济学含义。

## 判断标准
1. 量纲一致性：加减运算两侧应具有相同量纲（如价格+价格、收益率+收益率），价格+成交量属于量纲不匹配
2. 金融逻辑可解释性：因子应能对应某种市场行为或经济逻辑（如动量、反转、量价背离、波动率聚集等）
3. 非平凡构造：如 Corr($close,$high,10) 构造上近乎完全相关，无信息增量
4. 表达式必须存在明显经济含义，边界情况视为无经济含义

## 待评估表达式
{exprs}

## 返回格式
只返回一个 JSON 数组，每个元素对应一个表达式：
{{"expr": ..., "meaningful": ... , "desc": ...}}
- expr: 表达式原文，必须与上方"待评估表达式"中的字符串完全一致
- meaningful: 是否具有经济学含义（true/false）
- desc: 中文经济学含义说明。如果没有明显经济含义则设置为""。

### desc 编写规则（重要）
desc 需遵循"拆解→解释→逻辑"三段式结构，简洁精炼：

**a. 拆解表达式**：用中文描述表达式的各个组成部分，说明每个算子/子表达式在计算什么。让不熟悉 qlib 语法的读者也能理解表达式的运算逻辑。
**b. 解释数值含义**：说明"值高代表什么、值低代表什么"，或说明"值高/值低的驱动因素是什么"。如果因子方向已知（IC 正/负），可注明。
**c. 点明金融逻辑**：一句话概括核心的金融机制或市场行为，让读者理解因子为什么会有预测能力。

### desc 示例（参考格式与精炼度）
1. `Cov($low/$close, EMA($close/Ref($close,1)-1, 20)/Std($close/Ref($close,1)-1, 20), 20)`
   desc: "质量动量：low/close（日内下探幅度）与风险调整动量（EMA(ret)/Std(ret)）的20日协方差。值高=趋势走强的同时路径平滑、下探不深，区分趋势质量与噪声型上涨"

2. `EMA(Power(($close-$open)/$open + ($high-Greater($open,$close))/$open, 3), 20)`
   desc: "短线情绪过热反转：实体涨跌幅+上影线比例的三次方（极端K线非线性放大）做EMA平滑。值高=近期反复强上冲长上影，买盘情绪强但上方抛压同步释放，IC为负，捕捉短期过热后的回吐"

3. `EMA(($close/Ref($close,1)-1)*($volume/Ref($volume,1)-1) - ($open-Ref($close, 1)) / (Ref($close, 1) + 1e-12), 20)`
   desc: "剔除跳空后的量价拥挤：(收益率×成交量变化率)减去隔夜跳空幅度。第一项=盘中量价共振，第二项=隔夜信息冲击。剔除跳空后的盘中量价同步越强→资金拥挤→短期越易回落（IC为负）"

只返回 JSON 数组，不要加其他文字说明。"""

GENERATE_PROMPT_TEMPLATE = """你是一位量化金融研究员，擅长设计量价选股因子。

{operators_ref}

## 任务
基于以下当前表现最好的因子表达式，设计 {n} 个新的量价因子表达式。

## 当前最佳因子（含适应度分数）
{top_exprs}

## 失败案例（请避免类似结构）
{invalid_exprs}

## 设计要求
1. 每个因子必须具有清晰的金融逻辑（如：反转效应、动量质量、量价共振、情绪过热等）
2. 因子表达式必须是合法的 qlib 表达式语法
3. 尝试组合不同的金融逻辑，而非简单重复现有因子
4. 优先使用 5-20 日的滚动窗口
5. 表达式中不要包含子表达式基因（{gene_aliases}）——这些是已注册的内部符号

## 返回格式
只返回一个 JSON 对象，包含 "factors" 数组，每个元素有 "expr" 和 "logic" 字段。
示例：
{{"factors": [{{"expr": "Corr($close/Ref($close,1)-1, $volume/Ref($volume,1)-1, 20)", "logic": "量价相关性"}}]}}

只返回 JSON，不要加其他文字说明。"""


# ================================================================
# LLM 接口
# ================================================================

class LLMInterface:
    """LLM 集成：子表达式基因提取 + 周期性因子注入。

    对接 OpenAI 兼容 API（本地 Qwen / 远程 API）。
    """

    def __init__(self, model: str = None, base_url: str = None, api_key: str = None,
                 temperature: float = 0.7):
        cfg = get_config().get("llm_deepseek", {})
        self.model = model or "deepseek-v4-pro"
        self.base_url = base_url or cfg.get("base_url", "https://api.deepseek.com")
        self.api_key = api_key or cfg.get("api_key", "")
        self.temperature = temperature
        self._client = None
        self._gene_aliases: list[str] = []

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=httpx.Timeout(120.0, read=300.0),
            )
        return self._client

    # ================================================================
    # 子表达式基因提取
    # ================================================================

    def extract_sub_expr_genes(
        self,
        template_factors: list[tuple[str, str]],
        exist_genes: list[str],
        n_target: int = 50,
        # gap_hint: str = "",
    ) -> dict[str, dict]:
        """从初始优质因子中提取子表达式基因。

        Args:
            template_factors:  模版因子. [(name, qlib_expr), ...]
                例如 [("KMID", "($close-$open)/$open"), ...]
            exist_genes: 已有基因列表，避免生成相同基因
            n_target: 期望提取的基因数量，写入提示词引导 LLM 产出规模。
                默认 2 * len(factor_exprs)（参考报告：29 因子→68 基因≈2.3/因子）。
            gap_hint: 从已有基因统计的缺口提示，非空时注入 prompt 引导 LLM 补齐。

        Returns:
            {"gene_name": {"expr": qlib_sub_expr, "desc": str, "role": str}, ...}
        """

        # 构建模版因子列表文本
        factors_text = "\n".join(
            f"- {name}: `{expr}`" for name, expr in template_factors
        )
        # 已有基因文本
        genes_text = "\n".join(f"- `{gene}`" for gene in exist_genes)

        # 缺口提示：非空时追加换行，空时为空字符串
        # hint = f"\n## 补充要求\n{gap_hint}" if gap_hint else ""

        prompt = SUB_EXPR_PROMPT_TEMPLATE.format(
            operators_ref=QLIB_OPERATORS_REF,
            factors=factors_text,
            genes=genes_text,
            n_target=n_target,
            # gap_hint=hint,
        )
        logger.info('\n{}\n{}'.format('-' * 30, prompt))

        response = self._call_llm(prompt)
        genes = self._parse_json_response(response)

        if not genes:
            logger.warning("LLM 未提取到有效子表达式基因")
            return {}

        # 验证每个基因是合法 qlib 表达式
        valid_genes = {}
        for name, payload in genes.items():
            # 兼容旧格式（裸字符串）与新格式（{expr, desc}）
            if isinstance(payload, str):
                expr, desc, role = payload, "", ""
            elif isinstance(payload, dict):
                expr = payload.get("expr", "")
                desc = payload.get("desc", "")
                role = payload.get("role", "")
            else:
                continue
            if not isinstance(expr, str) or len(expr) < 3:
                continue
            # 检查不包含非法字符
            if re.search(r'[^{}"]', expr) and not re.search(r'[{}]', expr):
                valid_genes[name] = {"expr": expr, "desc": desc, "role": role}

        logger.info("LLM 提取了 %d 个子表达式基因（%d 有效）",
                     len(genes), len(valid_genes))
        return valid_genes

    # ================================================================
    # 周期性候选因子生成
    # ================================================================

    def generate_candidates(
        self,
        top_exprs: list[tuple[str, float]],
        invalid_patterns: list[str],
    ) -> list[str]:
        """周期性生成新候选因子。

        Args:
            top_exprs: [(expr_str, fitness), ...] 当前最佳因子
            invalid_patterns: 最近失败的表达式
            gen: 当前代数

        Returns:
            新 qlib 表达式字符串列表
        """
        # 构建 Top 因子文本
        top_text = "\n".join(
            f"{i+1}. fitness={fitness:.4f} | `{expr[:120]}`"
            for i, (expr, fitness) in enumerate(top_exprs)
        )
        if not top_text:
            top_text = "（暂无）"

        # 构建失败案例文本
        if invalid_patterns:
            invalid_text = "\n".join(
                f"- `{expr[:100]}`" for expr in invalid_patterns
            )
        else:
            invalid_text = "（暂无）"

        prompt = GENERATE_PROMPT_TEMPLATE.format(
            operators_ref=QLIB_OPERATORS_REF,
            n=5,
            top_exprs=top_text,
            invalid_exprs=invalid_text,
            gene_aliases=", ".join(self._gene_aliases) if self._gene_aliases else "无",
        )
        print('\n{}\n{}'.format('-' * 50, prompt))

        response = self._call_llm(prompt)
        data = self._parse_json_response(response)

        if not data:
            logger.warning("LLM 生成候选因子时返回无效 JSON")
            return []

        factors = data.get("factors", [])
        if not isinstance(factors, list):
            factors = []

        candidates = []
        for item in factors:
            if isinstance(item, dict) and "expr" in item:
                expr = item["expr"].strip()
                if expr and self._validate_expr(expr):
                    candidates.append(expr)
                    logger.debug(
                        "LLM 候选: %s | %s",
                        item.get("logic", "?"),
                        expr[:80],
                    )

        logger.info("LLM 生成 %d 候选因子（%d 有效）",
                     len(factors), len(candidates))
        print('\n{}\n{}'.format('-' * 50, candidates))
        return candidates

    # ================================================================
    # 进化总结
    # ================================================================

    def summarize_evolution(
        self,
        all_candidates: list[tuple[str, dict]],
    ) -> str:
        """进化结束后，让 LLM 总结有效模式和失败经验。"""
        # 构建因子列表
        factors_text = "\n".join(
            f"- `{expr[:120]}` | IC={info.get('rank_ic_mean', 0):.4f} | "
            f"ICIR={info.get('icir', 0):.4f}"
            for expr, info in all_candidates[:30]
        )

        prompt = f"""你是一位量化金融研究员。以下是一次遗传编程因子挖掘的结果。

{QLIB_OPERATORS_REF}

## 最终候选因子
{factors_text}

## 任务
总结本次因子挖掘的有效模式和关键发现：
1. 哪些量价逻辑在横截面上表现较好？
2. 哪些因子结构虽然常见但效果不佳？
3. 对下一轮因子挖掘的建议。

用中文回答，控制在 200 字以内。"""

        return self._call_llm(prompt)

    # ================================================================
    # 经济学含义批量评估
    # ================================================================

    def assess_economic_meaning(self, exprs: list[str]) -> list[dict]:
        """批量评估表达式的经济学含义。

        Args:
            exprs: qlib 表达式字符串列表

        Returns:
            [{"id": int, "meaningful": bool, "desc": str}, ...]
        """
        exprs_text = "\n".join(f"{i}. `{expr}`" for i, expr in enumerate(exprs))

        prompt = ECONOMIC_CHECK_PROMPT_TEMPLATE.format(
            operators_ref=QLIB_OPERATORS_REF,
            exprs=exprs_text,
        )
        logger.info('\n{}\n{}'.format('-' * 30, prompt))

        response = self._call_llm(prompt)
        data = self._parse_json_response(response)

        if not data:
            logger.warning("LLM 经济检查返回无效 JSON")
            return []

        # 兼容返回数组或 {"results": [...]} 包装
        if isinstance(data, dict):
            results = data.get("results", [])
        elif isinstance(data, list):
            results = data
        else:
            return []

        valid = []
        for item in results:
            if isinstance(item, dict) and "expr" in item:
                valid.append({
                    "expr": item["expr"],
                    "meaningful": item.get("meaningful", True),
                    "desc": item.get("desc", ""),
                })

        logger.info("LLM 经济检查: %d 表达式 → %d 有效评估", len(exprs), len(valid))
        return valid

    # ================================================================
    # 内部方法
    # ================================================================

    def _call_llm(self, prompt: str, system: str = None) -> str:
        """调用 LLM，带重试。"""
        if system is None:
            system = "你是一位专业的量化金融研究员，擅长量价因子设计。回复简洁准确。"

        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    stream=False,
                    reasoning_effort="high",
                    extra_body={"thinking": {"type": "enabled"}}
                )
                content = response.choices[0].message.content or ""
                if not content.strip():
                    logger.warning("LLM 返回空内容 (attempt %d)", attempt + 1)
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    raise ValueError("LLM 连续返回空内容")
                return content

            except Exception as e:
                logger.warning("LLM 调用失败 (attempt %d): %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise

        return ""

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """从 LLM 回复中提取 JSON。"""
        if not response or not response.strip():
            return None

        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown code block 中提取
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试处理被截断的 markdown 代码块（有开头 ```json 但无闭合 ```）
        stripped = response.strip()
        if stripped.startswith("```"):
            inner = re.sub(r'^```(?:json)?\s*', '', stripped)
            if inner.strip():
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    pass
                match = re.search(r'\{[\s\S]*\}', inner)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except json.JSONDecodeError:
                        pass

        # 尝试找到第一个 { ... } 块
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("无法解析 LLM 回复为 JSON: %s", response[:200])
        return None

    def _validate_expr(self, expr: str) -> bool:
        """快速检查表达式是否看起来合法。

        注意：这只是初步检查——真正的验证由 FactorEvaluator.validate() 完成。
        """
        if not expr:
            return False
        # 检查括号匹配
        if expr.count('(') != expr.count(')'):
            return False
        # 检查是否包含不可打印字符
        if any(ord(c) < 32 and c not in '\t\n' for c in expr):
            return False
        # 不能包含基因别名（防止 LLM 引用未注册的符号）
        for alias in self._gene_aliases:
            if alias in expr:
                return False
        return True

    def set_gene_aliases(self, aliases: list[str]):
        """设置已注册的子表达式基因别名列表（供 prompt 使用）。"""
        self._gene_aliases = aliases
