"""LLM 接口层 — 子表达式基因提取 + 周期性因子注入。

通过 config.yaml 中的 llm_deepseek 配置对接 DeepSeek API（OpenAI 兼容）。

设计原则：
1. 提示词包含完整的算子列表 + 字段语义 + 子表达式基因列表
2. 要求 LLM 返回结构化输出（JSON），便于解析
3. 失败案例驱动：将无效表达式填入 prompt，避免 LLM 重复错误
4. 竞争替换：LLM 生成候选 → qlib 计算 IC → 仅 IC 优于底部个体才进入种群
"""

import json
import logging
import re
import time

import httpx
from typing import Optional

from utils import get_config

logger = logging.getLogger(__name__)

# ================================================================
# 提示词模板
# ================================================================

# qlib 算子参考（供 LLM 理解可用操作）
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
- Max(x, y): 取最大值
- Min(x, y): 取最小值
- Gt(x, y): x > y 则为 1, 否则 0
- Lt(x, y): x < y 则为 1, 否则 0

### 滚动算子 (float, int → float)
- Ref(x, d): d 日前的 x 值
- Mean(x, d): d 日滚动均值
- Std(x, d): d 日滚动标准差
- Delta(x, d): x - Ref(x, d)
- Rank(x, d): d 日内 x 的排名百分位
- WMA(x, d): d 日加权移动平均
- EMA(x, d): d 日指数移动平均

### 三元算子
- Corr(x, y, d): x 与 y 在 d 日内的滚动相关系数
- If(cond, a, b): cond > 0 则返回 a, 否则返回 b

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

SUB_EXPR_PROMPT_TEMPLATE = """你是一位量化金融研究员，擅长从量价因子中提取可复用的子表达式。

{operators_ref}

## 任务
分析以下优质量价因子表达式，提取其中可复用的子表达式（"因子基因"）。

## 要求
1. 每个子表达式必须是可以独立计算的合法 qlib 表达式
2. 子表达式应具有明确的金融含义（如：高开阴线、天量异动、隔夜波动率等）
3. 不要提取过于简单的子表达式（如单独的 $close、Mean($close,20) 等）
4. 用英文 snake_case 为每个子表达式命名

## 参考因子
{factors}

## 返回格式
只返回一个 JSON 对象，key 是子表达式名称，value 是 qlib 表达式字符串。
示例：
{{"bearish_reversal": "Gt($open/Ref($close,1)-1,0)*Gt($open/$close-1,0)"}}

只返回 JSON，不要加其他文字说明。"""

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
        factor_exprs: list[tuple[str, str]],
    ) -> dict[str, str]:
        """从初始优质因子中提取子表达式基因。

        Args:
            factor_exprs: [(name, qlib_expr), ...]
                例如 [("KMID", "($close-$open)/$open"), ...]

        Returns:
            {"gene_name": "qlib_sub_expr", ...}
        """
        # 构建因子列表文本
        factors_text = "\n".join(
            f"- {name}: `{expr}`" for name, expr in factor_exprs[:30]
        )

        prompt = SUB_EXPR_PROMPT_TEMPLATE.format(
            operators_ref=QLIB_OPERATORS_REF,
            factors=factors_text,
        )

        response = self._call_llm(prompt)
        genes = self._parse_json_response(response)

        if not genes:
            logger.warning("LLM 未提取到有效子表达式基因")
            return {}

        # 验证每个基因是合法 qlib 表达式
        valid_genes = {}
        for name, expr in genes.items():
            if not isinstance(expr, str):
                continue
            if len(expr) < 3:
                continue
            # 检查不包含非法字符
            if re.search(r'[^{}"]', expr) and not re.search(r'[{}]', expr):
                valid_genes[name] = expr

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
        gen: int,
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
            for i, (expr, fitness) in enumerate(top_exprs[:10])
        )
        if not top_text:
            top_text = "（暂无）"

        # 构建失败案例文本
        if invalid_patterns:
            invalid_text = "\n".join(
                f"- `{expr[:100]}`" for expr in invalid_patterns[:10]
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
    # 内部方法
    # ================================================================

    def _call_llm(self, prompt: str, system: str = None) -> str:
        """调用 LLM，带重试。"""
        if system is None:
            system = "你是一位专业的量化金融研究员，擅长量价因子设计。回复简洁准确。"

        for attempt in range(3):
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
