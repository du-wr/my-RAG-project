import inspect
import re
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


def build_system_prompt(instruction: str = "", example: str = "", pydantic_schema: str = "") -> str:
    delimiter = "\n\n---\n\n"
    schema = ""
    if pydantic_schema:
        schema = (
            "你的回答必须是合法的 JSON，并严格遵循下面的字段顺序与结构：\n"
            f"```\n{pydantic_schema}\n```"
        )
    if example:
        example = delimiter + example.strip()
    if schema:
        schema = delimiter + schema.strip()
    return instruction.strip() + schema + example


class InteractiveUnifiedAnswerPrompt:
    instruction = """
你是一个基于年报检索上下文的问答助手。你的任务是只根据给定上下文回答问题，不能使用外部知识，不能自行脑补。
要求：
1. 只依据上下文中明确出现的信息回答。
2. 公司主体、时间范围、字段口径必须和问题一致。
3. 目录页、标题页、页眉页脚、弱相关背景描述不能单独作为最终证据。
4. 如果上下文不能直接支持答案，返回 N/A，不要猜测。
5. answer_type 必须根据最终答案的实际形态填写，只能是 text、number、boolean、list、na 之一。
6. final_answer 如果是列表，必须返回字符串列表；如果是数值，返回数字；如果是布尔结论，返回 true 或 false；如果证据不足，返回 N/A。
7. 输出必须是 JSON。
8. step_by_step_analysis 必须简洁，控制在 3 到 5 句之内，总长度尽量不超过 120 个汉字。
9. reasoning_summary 必须是一句简短总结，尽量不超过 50 个汉字。
10. 不要复述大段上下文原文，不要抄表格，不要输出超长列表。
11. 你会收到“题型提示”和“期望答案形态”，必须优先遵守，不要自行切换题型。

当题型提示为 number，且期望答案形态为 single_numeric_value 时，额外遵守以下规则：
12. final_answer 只能返回单个 JSON 数字或 N/A，不能返回带单位的字符串，不能返回整句解释，不能返回多个年份并列结果。
13. 如果上下文同时给出多个年份或多个期间的数值，默认只返回报告期当期值；若无法明确哪一个是报告期当期值，则返回 N/A。
14. 对金额类指标（如营业收入、净利润、研发投入、总资产、负债、现金流、账面价值、余额、总额），若上下文单位为亿元、万元、千元，应统一换算为“元”后再返回数值。
15. 对每股收益、比率、比例、增长率等非金额类指标，不要按元换算，直接返回原指标数值。
16. 若数值口径、单位或币种不一致，且无法在上下文中统一，则返回 N/A。
17. 若 answer_type=number，应尽量同时填写 answer_unit 和 unit_basis；其中 answer_unit 表示最终答案单位，unit_basis 表示单位换算依据。
"""

    user_prompt = (
        "下面是检索得到的上下文：\n"
        '"""\n'
        "{context}\n"
        '"""\n\n'
        "---\n\n"
        "识别到的公司：{companies}\n\n"
        "题型提示：{question_kind_hint}\n"
        "期望答案形态：{answer_shape_hint}\n\n"
        '问题："{question}"\n'
    )

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="简洁分析过程，3 到 5 句，说明答案如何从上下文得出，总长度尽量不超过 120 个汉字")
        reasoning_summary: str = Field(description="一句简短总结，说明为什么可以或不可以回答，尽量不超过 50 个汉字")
        relevant_pages: List[int] = Field(description="直接支持答案的页码列表")
        answer_type: Literal["text", "number", "boolean", "list", "na"] = Field(description="最终答案的类型")
        final_answer: Union[List[str], bool, float, int, str, Literal["N/A"]] = Field(
            description="最终答案；若 answer_type=number，必须是单个数值，不得返回多年份文本或带单位长句；若证据不足返回 N/A"
        )
        answer_unit: Optional[str] = Field(default=None, description="最终答案单位；仅当 answer_type=number 时尽量填写，例如 元、元/股、%、个")
        unit_basis: Optional[str] = Field(default=None, description="单位口径或换算依据；仅当 answer_type=number 时尽量填写，例如 原文为万元，已换算为元")

    pydantic_schema = re.sub(r"^ {8}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = """
示例 1（字段题）：
{
  "step_by_step_analysis": "1. 我先确认问题询问的是公司网址。2. 在上下文中查找公司基本信息页。3. 第 4 页出现了公司网址字段，并且字段值明确。4. 该字段与问题完全一致，因此可以直接返回。",
  "reasoning_summary": "上下文存在公司网址的直接字段证据，可以作答。",
  "relevant_pages": [4],
  "answer_type": "text",
  "final_answer": "www.example.com"
}

示例 2（数值题）：
{
  "step_by_step_analysis": "1. 问题要求单个营业收入数值。2. 上下文表格同时列出报告期与上年同期。3. 我只取报告期当期值 837031.82。4. 该页单位为万元，因此换算为元。5. 最终返回单个数值。",
  "reasoning_summary": "取报告期营业收入并按万元换算为元。",
  "relevant_pages": [12],
  "answer_type": "number",
  "final_answer": 8370318200,
  "answer_unit": "元",
  "unit_basis": "原文为万元，已换算为元"
}
"""

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)
