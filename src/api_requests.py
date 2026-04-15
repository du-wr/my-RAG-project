import json
import os
from typing import Dict, List, Literal, Type

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

import src.interactive_prompts as interactive_prompts
import src.question_classification_prompts as question_classification_prompts


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

    def _get_provider_extra_body(self) -> dict:
        """为不同兼容提供方补充可选参数，默认关闭 Qwen 的思考模式以降低长输出。"""
        if self.provider != "qwen":
            return {}

        thinking_raw = os.getenv("QWEN_ENABLE_THINKING", "false").strip().lower()
        if thinking_raw in {"1", "true", "yes", "on"}:
            return {"enable_thinking": True}
        return {"enable_thinking": False}

    @staticmethod
    def _extract_json_object(raw_content: str) -> dict:
        """尽量从模型输出中提取 JSON 对象。"""
        raw_content = (raw_content or "").strip()
        if not raw_content:
            return {}

        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            pass

        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start >= 0 and end > start:
            candidate = raw_content[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _infer_answer_type(final_answer):
        if final_answer == "N/A" or final_answer is None:
            return "na"
        if isinstance(final_answer, bool):
            return "boolean"
        if isinstance(final_answer, (int, float)):
            return "number"
        if isinstance(final_answer, list):
            return "list"
        return "text"

    def _repair_structured_payload(self, payload: dict, response_format: Type[BaseModel]) -> dict:
        """对缺字段或半结构化结果做兜底修复，避免整题直接失败。"""
        payload = payload if isinstance(payload, dict) else {}
        field_names = set(response_format.model_fields.keys())

        interactive_fields = {
            "step_by_step_analysis",
            "reasoning_summary",
            "relevant_pages",
            "answer_type",
            "final_answer",
            "answer_unit",
            "unit_basis",
        }
        if interactive_fields.issubset(field_names):
            final_answer = payload.get("final_answer", "N/A")
            repaired = {
                "step_by_step_analysis": str(payload.get("step_by_step_analysis") or payload.get("analysis") or "模型未返回完整分析，已按现有字段兜底解析。"),
                "reasoning_summary": str(payload.get("reasoning_summary") or payload.get("summary") or "模型未返回完整总结。"),
                "relevant_pages": payload.get("relevant_pages") if isinstance(payload.get("relevant_pages"), list) else [],
                "answer_type": payload.get("answer_type") or self._infer_answer_type(final_answer),
                "final_answer": final_answer,
                "answer_unit": payload.get("answer_unit") or None,
                "unit_basis": payload.get("unit_basis") or None,
            }
            return response_format.model_validate(repaired).model_dump()

        return response_format.model_validate(payload).model_dump()

    def send_message(
        self,
        model: str | None = None,
        temperature: float = 0,
        system_content: str = "You are a helpful assistant.",
        human_content: str = "Hello!",
        is_structured: bool = False,
        response_format: Type[BaseModel] | None = None,
        max_tokens: int | None = None,
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
        if max_tokens is not None:
            # 同时传递两种常见上限字段，兼容不同 OpenAI 兼容网关的参数实现。
            params["max_tokens"] = max_tokens
            params["max_completion_tokens"] = max_tokens
        extra_body = self._get_provider_extra_body()
        if extra_body:
            params["extra_body"] = extra_body

        if is_structured:
            try:
                completion = self.llm.beta.chat.completions.parse(
                    **params,
                    response_format=response_format,
                )
                content = completion.choices[0].message.parsed.model_dump()
            except Exception:
                fallback_messages = [
                    {"role": "system", "content": system_content + "\n\n再次强调：只能输出完整 JSON，不能缺字段，不能输出超长文本。"},
                    {"role": "user", "content": human_content},
                ]
                completion = self.llm.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=fallback_messages,
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                    max_completion_tokens=max_tokens,
                    extra_body=extra_body or None,
                )
                raw_content = completion.choices[0].message.content or "{}"
                payload = self._extract_json_object(raw_content)
                content = self._repair_structured_payload(payload, response_format)
        else:
            completion = self.llm.chat.completions.create(**params)
            content = completion.choices[0].message.content

        self.response_data = {
            "requested_model": model,
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
        max_tokens: int | None = None,
    ):
        result = self.processor.send_message(
            model=model,
            temperature=temperature,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            max_tokens=max_tokens,
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

    @staticmethod
    def _resolve_answer_shape_hint(question_kind_hint: str | None) -> str:
        """将题型提示映射为更明确的答案形态，帮助统一 prompt 收敛输出。"""
        mapping = {
            "name": "single_field_text",
            "number": "single_numeric_value",
            "boolean": "boolean_value",
            "names": "string_list",
            "comparative": "single_company_or_company_list_or_na",
        }
        return mapping.get(question_kind_hint or "", "unknown")

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

    def classify_question_kind(self, question: str, companies: List[str], model: str):
        """使用轻量模型对问题做题型分类，为统一问答 prompt 提供显式 hint。"""
        answer_dict = self.send_message(
            model=model,
            system_content=question_classification_prompts.QuestionKindClassificationPrompt.system_prompt,
            human_content=question_classification_prompts.QuestionKindClassificationPrompt.user_prompt.format(
                question=question,
                company_count=len(companies),
                companies=", ".join(companies) if companies else "未识别公司",
            ),
            is_structured=True,
            response_format=question_classification_prompts.QuestionKindClassificationPrompt.AnswerSchema,
            max_tokens=int(os.getenv("QUESTION_KIND_MAX_OUTPUT_TOKENS", "200")),
        )
        return answer_dict

    def get_interactive_answer_from_rag_context(
        self,
        question: str,
        rag_context: str,
        companies: List[str],
        model: str,
        question_kind_hint: str | None = None,
    ):
        interactive_max_tokens = int(os.getenv("INTERACTIVE_MAX_OUTPUT_TOKENS", "800"))
        answer_dict = self.send_message(
            model=model,
            system_content=interactive_prompts.InteractiveUnifiedAnswerPrompt.system_prompt,
            human_content=interactive_prompts.InteractiveUnifiedAnswerPrompt.user_prompt.format(
                context=rag_context,
                question=question,
                companies=", ".join(companies) if companies else "未识别公司",
                question_kind_hint=question_kind_hint or "unknown",
                answer_shape_hint=self._resolve_answer_shape_hint(question_kind_hint),
            ),
            is_structured=True,
            response_format=interactive_prompts.InteractiveUnifiedAnswerPrompt.AnswerSchema,
            max_tokens=interactive_max_tokens,
        )
        return answer_dict
