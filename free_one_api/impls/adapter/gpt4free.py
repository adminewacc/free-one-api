# 上游库:xtekky/gpt4free。
# 原版实现的 provider 挑选逻辑会在所有 provider(含图像/音频类)里乱试,
# 一旦选中 PollinationsImage 之类就会 402/不支持流式而整体失败。
# 这里改为:精选纯文本、免鉴权 provider 优先,逐个回退,首个 chunk 之前失败就换下一个。

import logging
import random
import re
import traceback
import typing

import pkg_resources

using_g4f_version = pkg_resources.get_distribution("g4f").version


class FakeResponse:

    def json(self):
        return {
            "info": {
                "version": using_g4f_version
            }
        }


def repl(*args, **kwargs):
    return FakeResponse()


import requests

old_get = requests.get
requests.get = repl

import g4f

requests.get = old_get

from free_one_api.entities import request, response

from ...models import adapter
from ...models.adapter import llm
from ...entities import request, response, exceptions
from ...models.channel import evaluation


# 命中这些关键字的 provider 一律跳过(它们产出图片/音频/视频,不是聊天补全)。
MEDIA_KEYWORDS = (
    "image", "audio", "video", "tts", "speech", "upscale", "vision",
    "flux", "dalle", "diffusion", "midjourney", "stable", "sdxl", "voice",
)

# 优先尝试的纯文本 provider(免鉴权、实测相对稳定)。
DEFAULT_PROVIDERS = [
    "PollinationsAI",
    "Cloudflare",
    "Groq",
    "Gemini",
    "GeminiPro",
    "CohereForAI_C4AI_Command",
    "DeepInfra",
    "HuggingSpace",
    "LMArena",
    "MetaAI",
    "CopilotApp",
    "GLM",
    "Airforce",
    "AnyProvider",
]

DEFAULTS = {
    "upstream_model": "gpt-3.5-turbo",
    "models": ["gpt-3.5-turbo", "gpt-4"],
    "timeout_secs": 180,
    "providers": None,
}


@adapter.llm_adapter
class GPT4FreeAdapter(llm.LLMLibAdapter):
    _response_prefix_pattern = re.compile(r"\$@\$v=[^$]+\$@\$")

    upstream_model: str = DEFAULTS["upstream_model"]
    timeout_secs: int = DEFAULTS["timeout_secs"]

    @classmethod
    def _filter_response_text(cls, text):
        if not isinstance(text, str):
            return text
        return cls._response_prefix_pattern.sub("", text)

    @classmethod
    def name(cls) -> str:
        return "xtekky/gpt4free"

    @classmethod
    def description(cls) -> str:
        return "Use xtekky/gpt4free to access lots of GPT providers (text only, no auth)."

    def supported_models(self) -> list[str]:
        models = self._setting("models")
        if isinstance(models, str):
            models = [m.strip() for m in models.split(",") if m.strip()]
        return list(models or DEFAULTS["models"])

    def function_call_supported(self) -> bool:
        return False

    def stream_mode_supported(self) -> bool:
        return True

    def multi_round_supported(self) -> bool:
        return True

    @classmethod
    def config_comment(cls) -> str:
        return ('无需任何鉴权,直接保存即可。可选配置:\n'
                '{\n'
                '  "upstream_model": "gpt-3.5-turbo",\n'
                '  "providers": ["PollinationsAI", "Cloudflare", "Groq"]\n'
                '}\n\n'
                'providers 留空则自动按内置优先级挑选纯文本 provider,并在失败时逐个回退。\n'
                'gpt4free 稳定性一般,建议与其它渠道一起配置,由多渠道路由兜底。\n'
                '参考 https://github.com/xtekky/gpt4free')

    @classmethod
    def supported_path(cls) -> str:
        return "/v1/chat/completions"

    def __init__(self, config: dict, eval: evaluation.AbsChannelEvaluation):
        self.config = config if isinstance(config, dict) else {}
        self.eval = eval
        self._use_provider = None

    # ------------------------------------------------------------------ 配置

    def _setting(self, key: str):
        value = self.config.get(key) if isinstance(self.config, dict) else None
        if value is None or value == "":
            value = getattr(type(self), key, None)
        if value is None or value == "":
            value = DEFAULTS.get(key)
        return value

    def _timeout_secs(self) -> int:
        try:
            return int(self._setting("timeout_secs"))
        except (TypeError, ValueError):
            return DEFAULTS["timeout_secs"]

    def _upstream_model(self, requested: str) -> str:
        # g4f 多数 provider 不认 gpt-4 这类名字,统一换成配置的上游模型名。
        return str(self._setting("upstream_model") or requested)

    def get_config(self) -> dict:
        return dict(self.config)

    def _model_for_provider(self, provider) -> str:
        """g4f 各 provider 只认自己的模型名,配置的名字不被支持时退回它自己的第一个模型。"""
        configured = str(self._setting("upstream_model") or "").strip()
        try:
            raw = getattr(provider, "models", None)
            supported = [m for m in list(raw) if isinstance(m, str) and m.strip()] if raw else []
        except Exception:
            supported = []

        if configured and (not supported or configured in supported):
            return configured
        if supported:
            # 名字不被支持时优先用它自己的 default_model,再退回交给 g4f 自选(None 实测最稳)。
            default_model = getattr(provider, "default_model", None)
            if isinstance(default_model, str) and default_model.strip():
                return default_model
            return None
        return configured or "gpt-3.5-turbo"

    def _candidate_providers(self) -> list:
        configured = self.config.get("providers") if isinstance(self.config, dict) else None
        if isinstance(configured, str):
            configured = [n.strip() for n in configured.split(",") if n.strip()]

        if configured:
            out = []
            for name in configured:
                provider = getattr(g4f.Provider, str(name), None)
                if provider is not None and provider not in out:
                    out.append(provider)
            if out:
                return out

        curated = []
        for name in DEFAULT_PROVIDERS:
            provider = getattr(g4f.Provider, name, None)
            if provider is not None and provider not in curated:
                curated.append(provider)

        # 兜底:g4f 里其余"自称可用 + 纯文本 + 免鉴权"的 provider
        rest = []
        try:
            for provider in g4f.Provider.__providers__:
                if provider in curated:
                    continue
                name = provider.__name__.lower()
                if any(keyword in name for keyword in MEDIA_KEYWORDS):
                    continue
                if not getattr(provider, "working", False):
                    continue
                if getattr(provider, "needs_auth", False):
                    continue
                if not hasattr(provider, "create_async_generator"):
                    continue
                rest.append(provider)
        except Exception:
            pass
        rest.sort(key=lambda p: p.__name__)

        return curated + rest

    # ------------------------------------------------------------------ 探活

    async def test(self) -> typing.Union[bool, str]:
        last_error = "no provider"
        for provider in self._candidate_providers()[:8]:
            try:
                gen = provider.create_async_generator(
                    model=self._model_for_provider(provider),
                    messages=[{"role": "user", "content": "reply with: hi"}],
                    timeout=25,
                )
                async for chunk in gen:
                    if isinstance(chunk, str) and chunk.strip():
                        self._use_provider = provider
                        return True, ""
            except Exception as e:
                last_error = "{}: {}".format(provider.__name__, e)
                continue
            last_error = "{} 无输出".format(provider.__name__)
        return False, "没有可用的 g4f provider(" + str(last_error) + ")"

    # ------------------------------------------------------------------ 主流程

    async def query(self, req: request.Request) -> typing.AsyncGenerator[response.Response, None]:
        timeout = self._timeout_secs()
        last_error = "no provider"

        candidates = self._candidate_providers()
        if self._use_provider is not None and self._use_provider in candidates:
            candidates = [self._use_provider] + [p for p in candidates if p is not self._use_provider]

        for provider in candidates:
            try:
                gen = provider.create_async_generator(
                    model=self._model_for_provider(provider),
                    messages=req.messages,
                    timeout=timeout,
                )
            except Exception as e:
                last_error = "{}: {}".format(provider.__name__, e)
                continue

            emitted = False
            try:
                async for chunk in gen:
                    if not isinstance(chunk, str) or not chunk:
                        continue
                    emitted = True
                    yield response.Response(
                        id=random.randint(0, 1000000000),
                        finish_reason=response.FinishReason.NULL,
                        normal_message=self._filter_response_text(chunk),
                        function_call=None,
                    )
            except Exception as e:
                last_error = "{}: {}".format(provider.__name__, e)
                if not emitted:
                    logging.warning("g4f provider %s failed, trying next: %s", provider.__name__, e)
                    continue
                logging.warning("g4f provider %s failed mid-stream: %s", provider.__name__, e)
                yield response.Response(
                    id=random.randint(0, 1000000000),
                    finish_reason=response.FinishReason.STOP,
                    normal_message="",
                    function_call=None,
                )
                return

            if emitted:
                self._use_provider = provider
                yield response.Response(
                    id=random.randint(0, 1000000000),
                    finish_reason=response.FinishReason.STOP,
                    normal_message="",
                    function_call=None,
                )
                return

            last_error = "{} 无输出".format(provider.__name__)

        logging.error("g4f exhausted all providers: %s", last_error)
        raise exceptions.QueryHandlingError(
            502,
            "no_provider_available",
            "gpt4free 没有可用 provider: " + str(last_error),
            type="api_error",
        )
