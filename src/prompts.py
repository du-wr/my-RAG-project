import inspect
import re
from typing import List, Literal, Union

from pydantic import BaseModel, Field


def build_system_prompt(instruction: str = "", example: str = "", pydantic_schema: str = "") -> str:
    delimiter = "\n\n---\n\n"
    schema = ""
    if pydantic_schema:
        schema = (
            "你的回答必须是 JSON，并严格遵循下面的字段顺序与结构：\n"
            f"```\n{pydantic_schema}\n```"
        )
    if example:
        example = delimiter + example.strip()
    if schema:
        schema = delimiter + schema.strip()
    return instruction.strip() + schema + example


class RephrasedQuestionsPrompt:
    instruction = """
你是一个比较题改写助手。你的任务是把一个涉及多家公司的比较问题，改写成多条可以分别提问单家公司的独立问题。

要求：
1. 改写后的问题必须保留原问题的时间范围、指标口径和主体限制。
2. 只能把比较题拆成单公司问题，不能补充原问题没有提供的新信息。
3. 公司名称必须与原问题中的写法完全一致。
4. 输出必须覆盖所有给定公司，不能遗漏。
"""

    class RephrasedQuestion(BaseModel):
        """单家公司对应的改写问题。"""

        company_name: str = Field(description="公司名称，必须与原问题中的写法完全一致")
        question: str = Field(description="针对该公司的独立问题")

    class RephrasedQuestions(BaseModel):
        """改写后的问题列表。"""

        questions: List["RephrasedQuestionsPrompt.RephrasedQuestion"] = Field(description="每家公司的独立问题列表")

    pydantic_schema = """
class RephrasedQuestion(BaseModel):
    \"\"\"单家公司对应的改写问题。\"\"\"
    company_name: str = Field(description="公司名称，必须与原问题中的写法完全一致")
    question: str = Field(description="针对该公司的独立问题")

class RephrasedQuestions(BaseModel):
    \"\"\"改写后的问题列表。\"\"\"
    questions: List['RephrasedQuestionsPrompt.RephrasedQuestion'] = Field(description="每家公司的独立问题列表")
"""

    example = """
示例：
原问题："Apple 和 Microsoft 在 2022 年谁的营业收入更高？"
公司列表：["Apple", "Microsoft"]

输出：
{
  "questions": [
    {
      "company_name": "Apple",
      "question": "Apple 在 2022 年的营业收入是多少？"
    },
    {
      "company_name": "Microsoft",
      "question": "Microsoft 在 2022 年的营业收入是多少？"
    }
  ]
}
"""

    user_prompt = '原始比较问题："{question}"\n\n涉及公司：{companies}'
    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextSharedPrompt:
    instruction = """
你是一个基于检索增强生成（RAG）的年报问答系统。你只能根据提供的年报上下文作答，不能使用外部知识，也不能自行猜测。

通用规则：
1. 只依据上下文中明确出现的信息回答问题。
2. 问题中的时间范围、公司主体、指标口径和字段含义必须严格匹配。
3. 目录页、页眉页脚、章节标题、图注、弱相关背景描述不能单独作为最终证据。
4. 如果上下文只有相近字段、相近概念或不完整线索，不能据此推断答案。
5. 如果证据不足，必须返回 N/A。
6. relevant_pages 必须优先填写真正直接支撑答案的页面，而不是仅提到关键词的页面。
"""

    user_prompt = """
下面是检索得到的上下文：
\"\"\"
{context}
\"\"\"

---

问题：
"{question}"
"""


class AnswerWithRAGContextNamePrompt:
    instruction = (
        AnswerWithRAGContextSharedPrompt.instruction
        + """

补充规则：
1. 这类问题通常是字段抽取题，例如法定代表人、公司网址、电子邮箱、股票代码等。
2. 只有当字段名和字段值在上下文中明确对应时，才能作答。
3. 不能用相近字段替代目标字段，例如不能把董事长、总经理、联系人替代为法定代表人。
4. 如果问的是名称类答案，必须保持与原文写法一致。
"""
    )
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="详细分析过程，至少 5 步，说明字段如何与问题精确匹配。")
        reasoning_summary: str = Field(description="简短总结，说明为什么可以或不可以得出答案。")
        relevant_pages: List[int] = Field(description="直接支撑答案的页码列表。")
        final_answer: Union[str, Literal["N/A"]] = Field(description='最终答案，若证据不足则返回 "N/A"。')

    pydantic_schema = re.sub(r"^ {8}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = """
示例：
{
  "step_by_step_analysis": "1. 问题要求查找法定代表人，这是一个严格字段，不能与董事长或总经理混淆。2. 我先检查上下文中是否出现“法定代表人”字段。3. 在第 3 页发现“法定代表人：冉兴”的直接表述。4. 该字段名与问题完全一致，且字段值明确。5. 因此可以直接提取“冉兴”作为最终答案。",
  "reasoning_summary": "上下文存在“法定代表人：冉兴”的直接证据，可以精确作答。",
  "relevant_pages": [3],
  "final_answer": "冉兴"
}
"""

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextNumberPrompt:
    instruction = (
        AnswerWithRAGContextSharedPrompt.instruction
        + """

补充规则：
1. 这类问题通常是财务指标或数量字段题，必须严格核对指标名称、单位、币种和时间范围。
2. 只有上下文直接给出目标数值时才能回答，不能自行进行复杂推导。
3. 如果上下文明确写出单位，例如元、万元、百万元、亿元，应按问题所需口径返回统一数值。
4. 若口径不一致、币种不明或字段含义不清，应返回 N/A。
"""
    )
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="详细分析过程，至少 5 步，重点说明指标、单位和口径判断。")
        reasoning_summary: str = Field(description="简短总结，说明数值来源和口径。")
        relevant_pages: List[int] = Field(description="直接支撑数值的页码列表。")
        final_answer: Union[float, int, Literal["N/A"]] = Field(description='最终数值答案，若证据不足则返回 "N/A"。')

    pydantic_schema = re.sub(r"^ {8}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = """
示例：
{
  "step_by_step_analysis": "1. 问题询问营业收入，这是标准财务指标。2. 我先查找上下文中的“营业收入”字段。3. 在第 12 页发现营业收入为 837,031.82，单位为万元。4. 问题要求返回数值，因此需要按单位换算。5. 837,031.82 万元换算为 8,370,318,200 元，因此返回该整数值。",
  "reasoning_summary": "上下文直接给出了营业收入和单位，按万元换算为元后返回。",
  "relevant_pages": [12],
  "final_answer": 8370318200
}
"""

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextBooleanPrompt:
    instruction = (
        AnswerWithRAGContextSharedPrompt.instruction
        + """

补充规则：
1. 只有在上下文存在明确肯定或明确否定证据时，才能判断 True 或 False。
2. 仅出现相关关键词、目录标题、章节名、背景介绍，不足以判断为 True。
3. 如果上下文明确写出“未披露”“未发生”“未实施”“不适用”等，可以判断为 False。
4. 如果没有足够证据支持 True 或 False，也必须返回 False 以外的推断结论时，请保持严格，只根据明确证据作答。
"""
    )
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="详细分析过程，至少 5 步，重点说明支持 True 或 False 的证据。")
        reasoning_summary: str = Field(description="简短总结，说明判断依据。")
        relevant_pages: List[int] = Field(description="直接支撑布尔判断的页码列表。")
        final_answer: bool = Field(description="最终答案，只能是 True 或 False。")

    pydantic_schema = re.sub(r"^ {8}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = """
示例：
{
  "step_by_step_analysis": "1. 问题询问年报是否披露研发投入相关信息。2. 我检查上下文中与研发投入、研发费用、研发人员相关的页面。3. 第 45 页明确出现“研发投入金额”“研发投入占营业收入比例”等字段。4. 这些字段属于直接披露，不是仅在目录或标题中提及。5. 因此可以判断答案为 True。",
  "reasoning_summary": "上下文明确列出了研发投入相关字段，因此答案为 True。",
  "relevant_pages": [45],
  "final_answer": true
}
"""

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextNamesPrompt:
    instruction = (
        AnswerWithRAGContextSharedPrompt.instruction
        + """

补充规则：
1. 这类问题要求返回多个名称时，必须先判断问题要的是人名、机构名、产品名还是职位名。
2. 返回结果必须保持原文写法，不得改写。
3. 返回列表时需要去重，尽量保持与原文出现顺序一致。
4. 如果只能找到部分结果，但无法确认列表是否完整，应返回 N/A。
"""
    )
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="详细分析过程，至少 5 步，重点说明列表项的筛选依据。")
        reasoning_summary: str = Field(description="简短总结，说明为什么可以得到该列表。")
        relevant_pages: List[int] = Field(description="直接支撑列表答案的页码列表。")
        final_answer: Union[List[str], Literal["N/A"]] = Field(description='最终答案，返回名称列表；若无法确认完整性则返回 "N/A"。')

    pydantic_schema = re.sub(r"^ {8}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = """
示例：
{
  "step_by_step_analysis": "1. 问题要求的是职位名称，不是人名。2. 我先定位高管任命或职务变动相关页面。3. 第 89 页列出了人员及其新任职位。4. 我只保留职位字段，不保留姓名，并按原文顺序去重。5. 因此可以得到最终职位列表。",
  "reasoning_summary": "上下文明确列出了新任高管对应的职位名称，可以按原文返回列表。",
  "relevant_pages": [89],
  "final_answer": [
    "执行副总裁",
    "临时首席运营官"
  ]
}
"""

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class ComparativeAnswerPrompt:
    instruction = """
你是一个比较题汇总助手。你会收到多家公司已经回答出来的单公司结果，需要基于这些结果给出最终比较结论。

要求：
1. 只能依据给定结果比较，不能补充外部信息。
2. 若某家公司结果为 N/A，或口径、币种、时间范围不一致，应排除该公司。
3. 如果无法形成有效比较，返回 N/A。
4. 如果问题要求返回公司名称，必须保持与原问题中的写法一致。
5. 如果出现并列第一、并列最低等情况，且题目没有指定如何处理，应返回 N/A。
"""

    user_prompt = """
下面是各家公司已经得到的单公司答案：
\"\"\"
{context}
\"\"\"

---

原始比较问题：
"{question}"
"""

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="详细分析过程，至少 5 步，说明比较时如何过滤不可比结果。")
        reasoning_summary: str = Field(description="简短总结，说明最终比较结论。")
        relevant_pages: List[int] = Field(description="比较题不直接返回页码，保持空列表。")
        final_answer: Union[str, Literal["N/A"]] = Field(description='最终比较结论，通常为公司名称；无法比较时返回 "N/A"。')

    pydantic_schema = re.sub(r"^ {8}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = """
示例：
{
  "step_by_step_analysis": "1. 问题要求比较多家公司的总资产。2. 我先检查每家公司的单公司答案是否存在明确数值。3. 对于 N/A 或口径不一致的结果先排除。4. 对剩余公司按总资产数值进行比较。5. 最终选择数值最小的公司名称作为答案。",
  "reasoning_summary": "过滤不可比结果后，剩余公司中总资产最小者即为最终答案。",
  "relevant_pages": [],
  "final_answer": "B 公司"
}
"""

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerSchemaFixPrompt:
    system_prompt = """
你是一个 JSON 修复器。你的任务是把原始模型输出修复为合法 JSON。

要求：
1. 输出必须以 { 开始，以 } 结束。
2. 只能输出 JSON 本体，不能添加任何解释、前后缀或 Markdown 代码块。
3. 不要改变字段语义，只修复格式问题。
"""

    user_prompt = """
下面是目标 system prompt，其中定义了期望的 JSON 结构：
\"\"\"
{system_prompt}
\"\"\"

---

下面是格式不正确的模型输出，请修复为合法 JSON：
\"\"\"
{response}
\"\"\"
"""


class RerankingPrompt:
    system_prompt_rerank_single_block = """
你是一个检索结果重排序器。你会收到一个问题和一个候选文本块。

任务：
1. 判断该文本块是否能直接帮助回答问题。
2. 给出简短理由。
3. 输出 0 到 1 之间的 relevance_score。

要求：
1. 只能根据给定文本块判断，不能补充外部知识。
2. 目录页、标题页、页眉页脚、图注、弱相关背景页不能给高分。
3. 0 表示完全无关，1 表示高度相关且可直接支持答案。
"""

    system_prompt_rerank_multiple_blocks = """
你是一个检索结果重排序器。你会收到一个问题和多个候选文本块。

任务：
1. 分别判断每个文本块与问题的相关性。
2. 给出每个文本块的简短理由。
3. 输出 0 到 1 之间的 relevance_score。

要求：
1. 只能根据给定文本块判断，不能补充外部知识。
2. 目录页、标题页、页眉页脚、图注、弱相关背景页不能给高分。
3. 评分时优先考虑“是否直接支撑答案”，而不是“是否只包含相关关键词”。
"""


class RetrievalRankingSingleBlock(BaseModel):
    """单个候选文本块的相关性判断结果。"""

    reasoning: str = Field(description="说明该文本块与问题之间关系的简短分析")
    relevance_score: float = Field(description="0 到 1 之间的相关性分数，1 表示高度相关")


class RetrievalRankingMultipleBlocks(BaseModel):
    """多个候选文本块的相关性判断结果。"""

    block_rankings: List[RetrievalRankingSingleBlock] = Field(description="每个候选文本块对应的相关性判断结果")
