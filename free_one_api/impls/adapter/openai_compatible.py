"""通用转发适配器:任何 OpenAI 兼容上游都能作为一个渠道接进来。

设计目标是"个人多项目共用一个端点,上游再怎么限流也不把错误丢给调用方":

1. 模型池轮换 —— 上游按"每个免费模型每分钟 1 次"限流时,撞 429 就换池子里的下一个模型;
2. 多 key 轮换 —— 同一个模型先把配置里的每把 key 试一遍,再换模型(每个账号/每把 key
   是独立配额);
3. 上游状态码原样回传,429/503 才算"可重试",参数错误直接抛(换什么都不管用)。

配置项:
  base_url            上游根地址(可带或不带 /v1,不含 /chat/completions)
  api_key             单 key(旧写法)
  api_keys            多 key 列表,优先于 api_key;用于撞限流时轮换
  models              本渠道对外暴露(客户端可填)的模型名
  model_map           客户端模型名 -> 上游模型名
  default_model       没有映射、也没有模型池时的上游模型名
  pass_through_model  true 则把客户端模型名原样传上游
  free_model_pool     轮换用的上游模型池(撞 429 时依次尝试)
  max_model_rotations 池里最多换几个模型(默认 4)
  timeout_secs        单次请求超时
  extra_headers       额外请求头
"""

import json
import logging
import random
import typing

import aiohttp

from ...models import adapter
from ...models.adapter import llm
from ...entities import request, response, exceptions
from ...models.channel import evaluation


DEFAULTS = {
    "base_url": "https://text.pollinations.ai/openai",
    "api_key": "",
    "api_keys": [],
    "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "model_map": {},
    "default_model": "openai",
    "pass_through_model": False,
    "free_model_pool": [],
    "max_model_rotations": 4,
    "timeout_secs": 300,
    "extra_headers": {},
}

# 这些状态码才值得换个模型/换把 key 重试。
RETRYABLE_STATUS = (429, 503)

# 免费聚合网关经常用 4xx 表达"这个模型/这条渠道暂时不可用"(例:unoRouter 的
# get_channel_failed 表示该 key 的计费组没有可用上游)。这些也该换组合重试,
# 否则一把配置有问题的 key 会把整个请求打死。
RETRYABLE_MARKERS = (
    "rate limit",
    "rate-limit",
    "too many requests",
    "queue full",
    "get_channel_failed",
    "no available provider",
    "overloaded",
    "try again",
)


@adapter.llm_adapter
class OpenAICompatibleAdapter(llm.LLMLibAdapter):
    """把任意 OpenAI 兼容上游接入 free-one-api。"""

    base_url: str = DEFAULTS["base_url"]
    api_key: str = DEFAULTS["api_key"]
    default_model: str = DEFAULTS["default_model"]
    timeout_secs: int = DEFAULTS["timeout_secs"]
    pass_through_model: bool = DEFAULTS["pass_through_model"]

    @classmethod
    def name(cls) -> str:
        return "openai/compatible"

    @classmethod
    def description(cls) -> str:
        return "Forward to any OpenAI-compatible endpoint (base_url + optional api_key(s))."

    def function_call_supported(self) -> bool:
        return True

    def stream_mode_supported(self) -> bool:
        return True

    def multi_round_supported(self) -> bool:
        return True

    @classmethod
    def supported_path(cls) -> str:
        return "/v1/chat/completions"

    @classmethod
    def config_comment(cls) -> str:
        return (
            "配置示例:\n"
            "{\n"
            '  "base_url": "https://api.unorouter.com/v1",\n'
            '  "api_keys": ["sk-...", "sk-..."],\n'
            '  "models": ["unorouter-free"],\n'
            '  "pass_through_model": true,\n'
            '  "free_model_pool": ["deepseek-v4.1-flash:free", "gpt-4o:free"],\n'
            '  "max_model_rotations": 4\n'
            "}\n\n"
            "base_url 不带 /v1/chat/completions。api_keys 留空则不发 Authorization(无鉴权服务)。\n"
            "常见上游:\n"
            "  无鉴权: https://text.pollinations.ai/openai\n"
            "  Groq:   https://api.groq.com/openai   pass_through_model=true\n"
            "  Gemini: https://generativelanguage.googleapis.com/v1beta/openai\n"
            "  本地:   http://127.0.0.1:11434/v1 (Ollama)"
        )

    def __init__(self, config: dict, eval: evaluation.AbsChannelEvaluation):
        # 每次请求现取现用,管理页改完配置立即生效,不需要重启。
        self.config = config if isinstance(config, dict) else {}
        self.eval = eval

    # ------------------------------------------------------------------ 配置读取

    def _setting(self, key: str):
        """渠道配置 -> 类属性(config.yaml 全局默认) -> 内置默认值。"""
        value = self.config.get(key) if isinstance(self.config, dict) else None
        if value is None or value == "":
            value = getattr(type(self), key, None)
        if value is None or value == "":
            value = DEFAULTS.get(key)
        return value

    def _as_list(self, value) -> list:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def get_config(self) -> dict:
        """生效配置(数据库持久化用的就是这个)。

        这里绝对不能脱敏:dump_adapter() 的结果会被 database/sqlite.py 写回
        channel 表,回显用 "***" 会把真实 key 抹掉(踩过一次)。
        对外脱敏在 router/redact.py + WebAPIGroup 出口做。
        """
        return dict(self.config)

    def _key_candidates(self) -> list:
        """请求可用的 key 列表(去重,保留顺序);无鉴权上游返回 [""]。"""
        keys = self._as_list(self.config.get("api_keys") if isinstance(self.config, dict) else None)
        single = self._setting("api_key")

        if single:
            keys.insert(0, str(single).strip())

        seen, out = set(), []
        for key in keys:
            if key and key not in seen:
                seen.add(key)
                out.append(key)

        return out or [""]

    def _free_model_pool(self) -> list:
        return self._as_list(
            self.config.get("free_model_pool") if isinstance(self.config, dict) else None
        )

    def _max_rotations(self) -> int:
        try:
            return max(1, int(self._setting("max_model_rotations")))
        except (TypeError, ValueError):
            return DEFAULTS["max_model_rotations"]

    def _timeout_secs(self) -> int:
        try:
            return int(self._setting("timeout_secs"))
        except (TypeError, ValueError):
            return DEFAULTS["timeout_secs"]

    def _resolve_model(self, requested: str) -> str:
        mapping = self._setting("model_map")
        if isinstance(mapping, dict) and requested in mapping:
            return str(mapping[requested])
        if self._setting("pass_through_model"):
            return requested
        return str(self._setting("default_model") or requested)

    def _model_candidates(self, requested: str) -> list:
        """本次请求依次尝试的上游模型名。

        配了模型池就按池轮换(客户端点名的模型若在池里则排最前);没配则沿用
        model_map / pass_through / default_model 的老逻辑。
        """
        pool = self._free_model_pool()

        if pool:
            head = [requested] if requested in pool else []
            tail = [model for model in pool if model != requested]
            random.shuffle(tail)
            return (head + tail)[: self._max_rotations()]

        return [self._resolve_model(requested)]

    def supported_models(self) -> list[str]:
        models = self._as_list(self._setting("models"))
        mapping = self._setting("model_map")
        if isinstance(mapping, dict):
            models = list(mapping.keys()) + models
        models = self._free_model_pool() + models

        seen, out = set(), []
        for model in (models or DEFAULTS["models"]):
            if model and model not in seen:
                seen.add(model)
                out.append(model)
        return out

    def _url(self) -> str:
        base = str(self._setting("base_url") or DEFAULTS["base_url"]).rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def _headers(self, api_key: str) -> dict:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        extra = self._setting("extra_headers")
        if isinstance(extra, dict):
            headers.update({str(k): str(v) for k, v in extra.items()})
        return headers

    def _build_payload(self, req: request.Request, stream: bool, model: str) -> dict:
        payload = {
            "model": model,
            "messages": req.messages,
            "stream": stream,
        }
        tools = getattr(req, "tools", None)
        tool_choice = getattr(req, "tool_choice", None)
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        elif getattr(req, "functions", None):
            payload["functions"] = req.functions
        return payload

    @staticmethod
    def _mask(secret: str) -> str:
        if len(secret) > 14:
            return secret[:8] + "..." + secret[-4:]
        return "***" if secret else "(none)"

    @staticmethod
    def _is_retryable(err) -> bool:
        """只有限流/上游过载/渠道不可用值得换模型换 key;参数错误换哪个都一样。"""
        if getattr(err, "retryable", False):
            return True
        return getattr(err, "status_code", None) in RETRYABLE_STATUS

    @classmethod
    def _body_is_retryable(cls, body: str) -> bool:
        lowered = (body or "").lower()
        return any(marker in lowered for marker in RETRYABLE_MARKERS)

    # ------------------------------------------------------------------ 响应解析

    @staticmethod
    def _parse_tool_call(raw_calls: list):
        if not raw_calls:
            return None
        call = raw_calls[0] or {}
        function = call.get("function") or {}
        raw_arguments = function.get("arguments")
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            try:
                arguments = json.loads(raw_arguments) if raw_arguments else {}
            except (TypeError, ValueError):
                arguments = {"_raw": raw_arguments}
            if not isinstance(arguments, dict):
                arguments = {"_raw": arguments}
        return response.FunctionCall(function.get("name") or "", arguments, call.get("id"))

    async def _iter_stream(self, upstream, resp_id: int):
        """把上游 SSE 转成本项目的 Response 流。"""
        pending: dict = {}
        async for raw_line in upstream.content:
            line = raw_line.decode("utf-8", "ignore").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue

            choices = obj.get("choices") or []
            if not choices:
                continue

            delta = (choices[0] or {}).get("delta") or {}
            content = delta.get("content")
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            if content:
                yield response.Response(
                    id=resp_id,
                    finish_reason=response.FinishReason.NULL,
                    normal_message=content,
                    function_call=None,
                )

            for call in (delta.get("tool_calls") or []):
                key = call.get("index", 0)
                entry = pending.setdefault(key, {"id": None, "name": "", "arguments": ""})
                if call.get("id"):
                    entry["id"] = call["id"]
                function = call.get("function") or {}
                if function.get("name"):
                    entry["name"] = function["name"]
                if function.get("arguments"):
                    entry["arguments"] += function["arguments"]

        if pending:
            calls = [{
                "id": pending[k]["id"],
                "function": {"name": pending[k]["name"], "arguments": pending[k]["arguments"]},
            } for k in sorted(pending.keys())]
            tool_call = self._parse_tool_call(calls)
            if tool_call is not None and tool_call.function_name:
                yield response.Response(
                    id=resp_id,
                    finish_reason=response.FinishReason.FUNCTION_CALL,
                    normal_message=None,
                    function_call=tool_call,
                )
                return

        yield response.Response(
            id=resp_id,
            finish_reason=response.FinishReason.STOP,
            normal_message="",
            function_call=None,
        )

    # ------------------------------------------------------------------ 探活

    async def test(self) -> typing.Union[bool, str]:
        model = self._model_candidates(str(self._setting("default_model")))[0]
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
        }
        timeout = aiohttp.ClientTimeout(total=60)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._url(), json=payload, headers=self._headers(self._key_candidates()[0])
                ) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        return False, "上游 {} 返回 {}: {}".format(self._url(), resp.status, body[:300])
                    try:
                        data = json.loads(body)
                    except ValueError:
                        return False, "上游返回非 JSON: " + body[:200]
                    if not (data.get("choices") or []):
                        return False, "上游响应没有 choices: " + body[:200]
                    return True, ""
        except Exception as e:
            return False, "连不上 {}: {}".format(self._url(), e)

    # ------------------------------------------------------------------ 主流程

    async def _stream_once(self, model: str, api_key: str, req: request.Request, resp_id: int):
        """单次尝试:用指定的模型 + key 打上游。"""
        payload = self._build_payload(req, stream=True, model=model)
        timeout = aiohttp.ClientTimeout(total=self._timeout_secs(), sock_read=self._timeout_secs())

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self._url(), json=payload, headers=self._headers(api_key)
            ) as upstream:
                if upstream.status != 200:
                    body = await upstream.text()
                    error = exceptions.QueryHandlingError(
                        upstream.status,
                        "upstream_error",
                        "上游 {} 返回 {}: {}".format(self._url(), upstream.status, body[:500]),
                        type="api_error",
                    )
                    error.retryable = (
                        upstream.status in RETRYABLE_STATUS or self._body_is_retryable(body)
                    )
                    raise error

                ctype = (upstream.headers.get("Content-Type") or "").lower()
                if "event-stream" in ctype:
                    async for item in self._iter_stream(upstream, resp_id):
                        yield item
                    return

                # 上游忽略了 stream=true,按一次性响应处理。
                body = await upstream.text()
                try:
                    data = json.loads(body)
                except ValueError:
                    raise exceptions.QueryHandlingError(
                        502, "upstream_bad_body",
                        "上游返回无法解析的内容: " + body[:300], type="api_error",
                    )

                choices = data.get("choices") or []
                if not choices:
                    raise exceptions.QueryHandlingError(
                        502, "upstream_empty", "上游响应没有 choices: " + body[:300],
                        type="api_error",
                    )

                message = choices[0].get("message") or {}
                tool_call = self._parse_tool_call(message.get("tool_calls") or [])
                text = message.get("content") or ""

                if tool_call is not None:
                    yield response.Response(
                        id=resp_id,
                        finish_reason=response.FinishReason.FUNCTION_CALL,
                        normal_message=text or None,
                        function_call=tool_call,
                    )
                    return

                yield response.Response(
                    id=resp_id,
                    finish_reason=response.FinishReason.STOP,
                    normal_message=text,
                    function_call=None,
                )

    async def query(self, req: request.Request) -> typing.AsyncGenerator[response.Response, None]:
        """按 (模型 × key) 依次尝试,只有可重试的错误才继续换下一个。"""
        resp_id = random.randint(0, 1000000000)
        models = self._model_candidates(req.model)
        keys = self._key_candidates()

        # 模型在外层、key 在内层:同一个模型先把所有 key 试一遍(每把 key 一份配额),
        # 全撞限流再换下一个模型。
        attempts = [(model, key) for model in models for key in keys]
        last_error = None

        for index, (model, api_key) in enumerate(attempts):
            has_next = index < len(attempts) - 1
            gen = self._stream_once(model, api_key, req, resp_id)
            first = None

            try:
                first = await gen.__anext__()
            except StopAsyncIteration:
                first = None
            except Exception as e:
                last_error = e
                if has_next and self._is_retryable(e):
                    logging.warning(
                        "openai/compatible: 模型 %s + key %s 限流/过载(%s),尝试下一个组合",
                        model, self._mask(api_key), getattr(e, "code", None) or e,
                    )
                    continue
                raise

            if first is not None:
                yield first

            async for item in gen:
                yield item

            return

        if last_error is not None:
            raise last_error

        raise exceptions.QueryHandlingError(
            502, "upstream_unreachable", "上游没有任何可用模型", type="api_error",
        )
