from pathlib import Path

from src.pipeline import Pipeline, configs


def main():
    root_path = Path(r"D:\35174\Desktop\RAG_System_Learning\data\test_set")
    questions = [
        "浪潮信息归属于上市公司股东的净利润是多少？",
        "TCL智家、博众精工、浪潮信息、燕麦科技和精智达中，哪家营业收入最高？",
        "TCL智家、博众精工、浪潮信息、燕麦科技和精智达中，哪些公司同时披露了公司网址和法定代表人信息？",
    ]

    pipeline = Pipeline(root_path, run_config=configs["qwen_max_ce_top5_rag_only"])
    processor = pipeline._build_questions_processor()
    processor.parallel_requests = 1

    for question in questions:
        answer = processor.answer_single_question(question)
        print("\nQUESTION:", question)
        print("ANSWER:", answer)


if __name__ == "__main__":
    main()
