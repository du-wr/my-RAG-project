import inspect
import re
from typing import Literal

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


class QuestionKindClassificationPrompt:
    instruction = """
你是一个年报问答任务的题型分类器。你的任务是只根据问题文本判断问题所属的题型，不要回答问题本身。

可选题型只有以下五种：
1. name：单个名称或字段值抽取，例如法定代表人、公司网址、股票代码、注册地址。
2. number：单个数值或金额抽取，例如营业收入、净利润、研发投入、每股收益。
3. boolean：判断是否披露、是否存在、是否发生等真假型问题。
4. names：要求返回多个名称、项目或列表的问题，例如“有哪些主要产品”“列出核心技术人员”。
5. comparative：涉及两家公司及以上的比较、筛选、最高/最低/更高/更低问题。

分类要求：
1. 只做题型判断，不要补充外部知识。
2. comparative 优先级最高；只要问题要求在多家公司之间比较或筛选，就归为 comparative。
3. confidence 返回 0 到 1 之间的小数，表示你对分类结果的把握。
4. reasoning_summary 用一句中文简述分类依据，尽量不超过 40 个汉字。
5. 输出必须是 JSON。
"""

    user_prompt = (
        '问题："{question}"\n'
        "识别到的公司数量：{company_count}\n"
        "识别到的公司：{companies}\n"
    )

    class AnswerSchema(BaseModel):
        predicted_kind: Literal["name", "number", "boolean", "names", "comparative"] = Field(description="预测题型")
        confidence: float = Field(description="0 到 1 之间的分类置信度")
        reasoning_summary: str = Field(description="一句中文简述分类依据")

    pydantic_schema = re.sub(r"^ {8}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = """
示例：
{
  "predicted_kind": "number",
  "confidence": 0.96,
  "reasoning_summary": "问题要求提取单个财务指标数值。"
}
"""

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)
