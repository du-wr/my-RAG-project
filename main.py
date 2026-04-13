import click
from pathlib import Path

from src.dataset_preparation import prepare_dataset_from_source
from src.pipeline import Pipeline, configs


@click.group()
def cli():
    pass


@cli.command("prepare-dataset")
@click.option(
    "--source-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
def prepare_dataset(source_dir: Path):
    # 以当前工作目录作为单个数据集根目录，便于 test_set/newtest 独立运行。
    root_path = Path.cwd()
    result = prepare_dataset_from_source(root_path=root_path, source_dir=source_dir)
    click.echo(f"Prepared dataset with {result['pdf_count']} PDF files")
    click.echo(f"pdf_reports: {result['pdf_reports_dir']}")
    click.echo(f"subset.csv: {result['subset_csv']}")
    click.echo(f"questions.json: {result['questions_json']}")


@cli.command("process-mineru-reports")
@click.option(
    "--source-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--config",
    type=click.Choice(["qwen_default"]),
    default="qwen_default",
)
def process_mineru_reports(source_dir: Path, config: str):
    root_path = Path.cwd()
    pipeline = Pipeline(root_path, run_config=configs[config])
    click.echo(f"Processing MinerU reports from {source_dir} (config={config})...")
    pipeline.process_mineru_reports(source_dir)


@cli.command("process-questions")
@click.option(
    "--config",
    type=click.Choice(
        [
            "qwen_turbo",
            "qwen_max_rerank",
            "qwen_plus_rerank",
            "qwen_max_llm_top5",
            "qwen_max_ce_top5",
        ]
    ),
    default="qwen_max_rerank",
)
def process_questions(config: str):
    root_path = Path.cwd()
    pipeline = Pipeline(root_path, run_config=configs[config])
    click.echo(f"Processing questions (config={config})...")
    pipeline.process_questions()


@cli.command("interactive-qa")
@click.option(
    "--config",
    type=click.Choice(
        [
            "qwen_turbo",
            "qwen_max_rerank",
            "qwen_plus_rerank",
            "qwen_max_llm_top5",
            "qwen_max_ce_top5",
        ]
    ),
    default="qwen_max_llm_top5",
)
@click.option(
    "--schema",
    type=click.Choice(["auto", "name", "number", "boolean", "names"]),
    default="auto",
)
def interactive_qa(config: str, schema: str):
    # ?????????????????? data\test_set??????????????? subset ???
    root_path = Path.cwd()
    pipeline = Pipeline(root_path, run_config=configs[config])
    click.echo(f"Interactive QA started (config={config}, schema={schema})")
    click.echo("?????????????? exit?quit ? ?? ???")

    while True:
        question_text = click.prompt("??", prompt_suffix="> ", default="", show_default=False).strip()
        if not question_text or question_text.lower() in {"exit", "quit"}:
            click.echo("Interactive QA stopped.")
            break

        resolved_schema = None if schema == "auto" else schema
        answer = pipeline.answer_single_question(question_text, schema=resolved_schema)

        if "error" in answer:
            click.echo(f"[ERROR] {answer['error']}")
            continue

        click.echo(f"[Schema] {answer.get('resolved_kind')}")
        resolved_companies = answer.get('resolved_companies') or answer.get('resolved_company_names') or []
        if resolved_companies:
            click.echo(f"[Companies] {', '.join(resolved_companies)}")
        click.echo(f"[Route] {answer.get('route')}")
        click.echo(f"[Status] {answer.get('status')}")
        click.echo(f"[Answer] {answer.get('value')}")
        references = answer.get('references', [])
        if references:
            pages = ', '.join(str(ref.get('page_index')) for ref in references)
            click.echo(f"[Pages] {pages}")
        detail = answer.get('detail') or {}
        summary = detail.get('reasoning_summary')
        if summary:
            click.echo(f"[Reasoning] {summary}")
        click.echo('---')


if __name__ == "__main__":
    cli()
