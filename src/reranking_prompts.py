def build_system_prompt(instruction: str = "", example: str = "") -> str:
    delimiter = "\n\n---\n\n"
    if example:
        return instruction.strip() + delimiter + example.strip()
    return instruction.strip()


class RerankingPrompt:
    instruction_single_block = """
你是文档检索重排序助手。你的任务是判断一个候选文本块对用户问题是否直接有帮助。
要求：
1. 只根据问题与候选文本块本身进行判断。
2. 优先给直接回答问题、包含明确字段值、结论或关键证据的文本块高分。
3. 对只有目录、标题、页眉页脚、弱相关背景描述的文本块给低分。
4. relevance_score 必须在 0 到 1 之间。
5. 输出必须是 JSON。
"""

    instruction_multiple_blocks = """
你是文档检索重排序助手。你的任务是对多个候选文本块分别给出相关性评分。
要求：
1. 逐块独立判断，不要把一个文本块中的信息转移给另一个文本块。
2. 优先给直接回答问题、包含明确字段值、结论或关键证据的文本块高分。
3. 对目录、标题、页眉页脚、弱相关背景描述给低分。
4. 每个 block 都要返回 relevance_score 和 reasoning。
5. 输出必须是 JSON。
"""

    example_single_block = """
示例：
{
  "relevance_score": 0.92,
  "reasoning": "该文本块直接给出了公司网址字段和字段值，与问题完全对应。"
}
"""

    example_multiple_blocks = """
示例：
{
  "block_rankings": [
    {
      "relevance_score": 0.91,
      "reasoning": "该文本块直接给出问题所需字段值。"
    },
    {
      "relevance_score": 0.18,
      "reasoning": "该文本块只包含目录信息，没有直接证据。"
    }
  ]
}
"""

    system_prompt_rerank_single_block = build_system_prompt(instruction_single_block, example_single_block)
    system_prompt_rerank_multiple_blocks = build_system_prompt(instruction_multiple_blocks, example_multiple_blocks)
