"""모델 응답을 Pydantic 스키마로 받는다. 제공자의 structured output이 안 되면 텍스트에서 JSON을 찾는다."""

from __future__ import annotations

import json
import re
from typing import TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError

Schema = TypeVar("Schema", bound=BaseModel)


def parse_json(schema: type[Schema], text: str) -> Schema:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Reply has no JSON object: {text[:200]!r}")
    return schema.model_validate(json.loads(match.group(0)))


async def structured_call(llm: BaseChatModel, schema: type[Schema], messages: list[BaseMessage]) -> Schema:
    try:
        result = await llm.with_structured_output(schema).ainvoke(messages)
        if isinstance(result, dict):
            result = schema.model_validate(result)
        if isinstance(result, schema):
            return result
    except (NotImplementedError, OutputParserException, ValidationError, ValueError):
        pass  # thinking 모델이 빈 응답을 내거나 형식을 못 맞춘 경우. 아래에서 다시 묻는다.
    response = await llm.ainvoke(messages)
    content = response.content if isinstance(response.content, str) else json.dumps(response.content)
    return parse_json(schema, content)
