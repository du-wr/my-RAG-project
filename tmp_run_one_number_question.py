from pathlib import Path

from src.pipeline import Pipeline, configs


def main():
    root_path = Path(r"D:\35174\Desktop\RAG_System_Learning\data\test_set")
    question_text = "浪潮信息归属于上市公司股东的净利润是多少？"

    pipeline = Pipeline(root_path, run_config=configs["qwen_max_ce_top5_rag_only"])
    processor = pipeline._build_questions_processor()
    processor.parallel_requests = 1

    answer = processor.answer_single_question(question_text)
    print(answer)


if __name__ == "__main__":
    main()
