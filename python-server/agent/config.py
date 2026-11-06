"""config/app.yaml과 config/models.yaml을 읽어 실행 설정을 만든다."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class ConfigError(RuntimeError):
    pass


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class UnityConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8001"
    request_timeout_seconds: float = 15.0
    action_timeout_seconds: float = 90.0


class AgentConfig(BaseModel):
    max_steps: int = 30
    max_command_seconds: float = 900.0
    near_radius_meters: float = 3.0
    replan_after_failures: int = 3
    max_consecutive_failures: int = 6
    ground_unknown_objects: bool = True
    # 객체 종류(label 접두어)의 사용자 언어 이름. 명령을 영어로 다시 쓸 때 쓴다.
    type_names: dict[str, list[str]] = Field(default_factory=dict)


class JudgeConfig(BaseModel):
    enabled: bool = True


class LoggingConfig(BaseModel):
    runs_dir: str = "runs"
    save_images: bool = True


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    unity: UnityConfig = Field(default_factory=UnityConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class ProviderConfig(BaseModel):
    type: Literal["openrouter", "ollama"]
    base_url: str
    credentials_file: str | None = None


class ModelProfile(BaseModel):
    name: str = ""
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    context_length: int | None = None
    thinking: bool | None = None
    timeout_seconds: float = 120.0


class OpenRouterCredentials(BaseModel):
    api_key: SecretStr
    http_referer: str | None = None
    app_title: str | None = None


class ModelsConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    profiles: dict[str, ModelProfile]
    active_profile: str
    judge_profile: str | None = None
    planner_profile: str | None = None


class Settings(BaseModel):
    config_dir: Path
    app: AppConfig
    models: ModelsConfig
    profile: ModelProfile
    judge_profile: ModelProfile
    planner_profile: ModelProfile | None = None

    def provider_for(self, profile: ModelProfile) -> ProviderConfig:
        try:
            return self.models.providers[profile.provider]
        except KeyError as exc:
            raise ConfigError(f"Unknown provider '{profile.provider}' in profile '{profile.name}'") from exc

    def openrouter_credentials(self, provider: ProviderConfig) -> OpenRouterCredentials:
        if not provider.credentials_file:
            raise ConfigError("OpenRouter provider needs credentials_file")
        path = self.config_dir / provider.credentials_file
        if not path.exists():
            raise ConfigError(f"{path} is missing. Copy openrouter.example.yaml to {path.name} and set api_key.")
        return OpenRouterCredentials.model_validate(_read_yaml(path))


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _pick_profile(models: ModelsConfig, name: str) -> ModelProfile:
    if name not in models.profiles:
        raise ConfigError(f"Unknown model profile '{name}'. Choose one of: {', '.join(models.profiles)}")
    return models.profiles[name].model_copy(update={"name": name})


def load_settings(
    config_dir: Path | None = None,
    profile: str | None = None,
    judge_profile: str | None = None,
    planner_profile: str | None = None,
) -> Settings:
    config_dir = (config_dir or DEFAULT_CONFIG_DIR).resolve()
    app = AppConfig.model_validate(_read_yaml(config_dir / "app.yaml"))
    models = ModelsConfig.model_validate(_read_yaml(config_dir / "models.yaml"))
    active = _pick_profile(models, profile or models.active_profile)
    judge_name = judge_profile or models.judge_profile
    judge = _pick_profile(models, judge_name) if judge_name else active
    planner_name = planner_profile or models.planner_profile
    planner = _pick_profile(models, planner_name) if planner_name and planner_name != "none" else None
    return Settings(
        config_dir=config_dir, app=app, models=models, profile=active, judge_profile=judge, planner_profile=planner
    )
