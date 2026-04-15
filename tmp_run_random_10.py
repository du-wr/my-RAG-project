import json
import random
from pathlib import Path

from src.pipeline import Pipeline, configs


def main():
    root_path = Path(r"D:\35174\Desktop\RAG_System_Learning\data\test_set")
    config_name = "qwen_max_ce_top5_rag_only"
    random_seed = 20260415
    sample_size = 10

    questions_path = root_path / "questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    rng = random.Random(random_seed)
    sampled_questions = rng.sample(questions, sample_size)

    print(f"Running random sample with seed={random_seed}, size={sample_size}, config={config_name}")
    for index, item in enumerate(sampled_questions, start=1):
        print(f"{index:02d}. {item.get('text')}")

    pipeline = Pipeline(root_path, run_config=configs[config_name])
    processor = pipeline._build_questions_processor()
    output_path = root_path / "answers_qwen_max_ce_top5_rag_only_random10.json"
    result = processor.process_questions_list(sampled_questions, output_path=str(output_path), submission_file=False)

    print("\nSample statistics:")
    print(json.dumps(result["statistics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
