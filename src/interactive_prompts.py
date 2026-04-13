import inspect
import re
from typing import List, Literal, Union

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
"""

    user_prompt = (
        "下面是检索得到的上下文：\n"
        '"""\n'
        "{context}\n"
        '"""\n\n'
        "---\n\n"
        "识别到的公司：{companies}\n\n"
        '问题："{question}"\n'
    )

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="详细分析过程，至少 4 步，说明答案如何从上下文得出")
        reasoning_summary: str = Field(description="简短总结，说明为什么可以或不可以回答")
        relevant_pages: List[int] = Field(description="直接支持答案的页码列表")
        answer_type: Literal["text", "number", "boolean", "list", "na"] = Field(description="最终答案的类型")
        final_answer: Union[List[str], bool, float, int, str, Literal["N/A"]] = Field(
            description="最终答案；若证据不足返回 N/A"
        )

    pydantic_schema = re.sub(r"^ {8}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = """
示例：
{
  "step_by_step_analysis": "1. 我先确认问题询问的是公司网址。2. 在上下文中查找公司基本信息页。3. 第 4 页出现了公司网址字段，并且字段值明确。4. 该字段与问题完全一致，因此可以直接返回。",
  "reasoning_summary": "上下文存在公司网址的直接字段证据，可以作答。",
  "relevant_pages": [4],
  "answer_type": "text",
  "final_answer": "www.example.com"
}
"""

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)
