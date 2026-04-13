import os
from typing import Dict, List, Literal, Type

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

import src.interactive_prompts as interactive_prompts


class OpenAICompatibleProcessor:
    def __init__(self, provider: Literal["openai", "qwen"] = "qwen", default_model: str | None = None):
        self.provider = provider
        self.default_model = default_model or self._get_default_model()
        self.llm = self._set_up_llm()
        self.response_data = {}

    def _get_default_model(self) -> str:
        if self.provider == "qwen":
            return os.getenv("QWEN_CHAT_MODEL", "qwen-max")
        return "gpt-4o-2024-08-06"

    def _set_up_llm(self):
        load_dotenv()
        kwargs = {"timeout": None, "max_retries": 2}
        if self.provider == "qwen":
            kwargs["api_key"] = os.getenv("QWEN_API_KEY")
            base_url = os.getenv("QWEN_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
        else:
            kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        return OpenAI(**kwargs)

    def send_message(
        self,
        model: str | None = None,
        temperature: float = 0,
        system_content: str = "You are a helpful assistant.",
        human_content: str = "Hello!",
        is_structured: bool = False,
        response_format: Type[BaseModel] | None = None,
    ):
        model = model or self.default_model
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content},
            ],
        }
        params["temperature"] = temperature

        if is_structured:
            completion = self.llm.beta.chat.completions.parse(
                **params,
                response_format=response_format,
            )
            content = completion.choices[0].message.parsed.model_dump()
        else:
            completion = self.llm.chat.completions.create(**params)
            content = completion.choices[0].message.content

        self.response_data = {
            "model": completion.model,
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        }
        print(self.response_data)
        return content


class APIProcessor:
    def __init__(self, provider: Literal["openai", "qwen"] = "qwen"):
        self.provider = provider.lower()
        self.processor = OpenAICompatibleProcessor(provider=self.provider)
        self.response_data = {}

    def send_message(
        self,
        model: str | None = None,
        temperature: float = 0,
        system_content: str = "You are a helpful assistant.",
        human_content: str = "Hello!",
        is_structured: bool = False,
        response_format: Type[BaseModel] | None = None,
    ):
        result = self.processor.send_message(
            model=model,
            temperature=temperature,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
        )
        self.response_data = self.processor.response_data
        return result

    def _build_rag_context_prompts(self, schema: str):
        import src.prompts as prompts

        if schema == "name":
            return (
                prompts.AnswerWithRAGContextNamePrompt.system_prompt,
                prompts.AnswerWithRAGContextNamePrompt.AnswerSchema,
                prompts.AnswerWithRAGContextNamePrompt.user_prompt,
            )
        if schema == "number":
            return (
                prompts.AnswerWithRAGContextNumberPrompt.system_prompt,
                prompts.AnswerWithRAGContextNumberPrompt.AnswerSchema,
                prompts.AnswerWithRAGContextNumberPrompt.user_prompt,
            )
        if schema == "boolean":
            return (
                prompts.AnswerWithRAGContextBooleanPrompt.system_prompt,
                prompts.AnswerWithRAGContextBooleanPrompt.AnswerSchema,
                prompts.AnswerWithRAGContextBooleanPrompt.user_prompt,
            )
        if schema == "names":
            return (
                prompts.AnswerWithRAGContextNamesPrompt.system_prompt,
                prompts.AnswerWithRAGContextNamesPrompt.AnswerSchema,
                prompts.AnswerWithRAGContextNamesPrompt.user_prompt,
            )
        if schema == "comparative":
            return (
                prompts.ComparativeAnswerPrompt.system_prompt,
                prompts.ComparativeAnswerPrompt.AnswerSchema,
                prompts.ComparativeAnswerPrompt.user_prompt,
            )
        raise ValueError(f"Unsupported schema: {schema}")

    def get_answer_from_rag_context(self, question: str, rag_context: str, schema: str, model: str):
        system_prompt, response_format, user_prompt = self._build_rag_context_prompts(schema)
        answer_dict = self.send_message(
            model=model,
            system_content=system_prompt,
            human_content=user_prompt.format(context=rag_context, question=question),
            is_structured=True,
            response_format=response_format,
        )
        return answer_dict

    def get_rephrased_questions(self, original_question: str, companies: List[str]) -> Dict[str, str]:
        import src.prompts as prompts

        answer_dict = self.send_message(
            system_content=prompts.RephrasedQuestionsPrompt.system_prompt,
            human_content=prompts.RephrasedQuestionsPrompt.user_prompt.format(
                question=original_question,
                companies=", ".join([f'"{company}"' for company in companies]),
            ),
            is_structured=True,
            response_format=prompts.RephrasedQuestionsPrompt.RephrasedQuestions,
        )
        return {item["company_name"]: item["question"] for item in answer_dict["questions"]}

    def get_interactive_answer_from_rag_context(self, question: str, rag_context: str, companies: List[str], model: str):
        answer_dict = self.send_message(
            model=model,
            system_content=interactive_prompts.InteractiveUnifiedAnswerPrompt.system_prompt,
            human_content=interactive_prompts.InteractiveUnifiedAnswerPrompt.user_prompt.format(
                context=rag_context,
                question=question,
                companies=", ".join(companies) if companies else "??????",
            ),
            is_structured=True,
            response_format=interactive_prompts.InteractiveUnifiedAnswerPrompt.AnswerSchema,
        )
        return answer_dict
