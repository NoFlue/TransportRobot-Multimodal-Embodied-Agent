"""모델 프로필로 VLM 클라이언트를 만든다. VLM 추론 서버는 OpenRouter 또는 로컬 Ollama다."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from agent.config import ModelProfile, Settings


def create_chat_model(settings: Settings, profile: ModelProfile) -> BaseChatModel:
    provider = settings.provider_for(profile)
    if provider.type == "openrouter":
        from langchain_openai import ChatOpenAI

        credentials = settings.openrouter_credentials(provider)
        headers = {
            key: value
            for key, value in (("HTTP-Referer", credentials.http_referer), ("X-Title", credentials.app_title))
            if value
        }
        return ChatOpenAI(
            model=profile.model,
            base_url=provider.base_url,
            api_key=credentials.api_key.get_secret_value(),
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            timeout=profile.timeout_seconds,
            max_retries=1,
            default_headers=headers or None,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=profile.model,
        base_url=provider.base_url,
        temperature=profile.temperature,
        num_predict=profile.max_tokens,
        num_ctx=profile.context_length,
        reasoning=profile.thinking,
        client_kwargs={"timeout": profile.timeout_seconds},
    )
