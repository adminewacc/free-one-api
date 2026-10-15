import json
import logging
import random
import typing

import aiohttp

from ...models import adapter
from ...models.adapter import llm
from ...entities import request, response, exceptions
from ...models.channel import evaluation


# 通用转发适配器:任何 OpenAI 兼容上游都能作为一个渠道接进来。
# 用途:把官方免费额度(Pollinations 无鉴权 / Groq / Gemini / Cerebras / OpenRouter 免费模型)
# 或自建服务(Ollama、vLLM、one-api)统一挂到 /v1/chat/completions 后面,
# 配合多渠道路由实现"一个 key 打所有上游、单个挂了自动换下一个"。
#
# 配置项:
#   base_url          上游根地址,不含 /v1/chat/completions
#   api_key           可选,留空则不发 Authorization 头(无鉴权服务用)
#   models            本渠道对外暴露(客户端可填)的模型名列表
#   model_map         客户端模型名 -> 上游模型名,优先于其它规则
#   default_model     没有映射时的上游模型名
#   pass_through_model true 则把客户端模型名原样传给上游(上游模型名与客户端一致时用)
#   timeout_secs      单次请求超时
#   extra_headers     额外请求头

DEFAULTS = {
    "base_url": "https://text.pollinations.ai/openai",
    "api_key": "",
    "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "model_map": {},
    "default_model": "openai",
    "pass_through_model": False,
    "timeout_secs": 300,
    "extra_headers": {},
    # 模型池:上游按"每个免费模型 1 次/分钟"限流时,撞到 429 就池内换一个再试。
    "free_model_pool": [],
    "max_model_rotations": 4,
}


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
        return "Forward to any OpenAI-compatible endpoint (base_url + optional api_key)."

    def supported_models(self) -> list[str]:
        models = self._setting("models")
        if isinstance(models, str):
            models = [m.strip() for m in models.split(",") if m.strip()]
        mapping = self._setting("model_map") or {}
        if isinstance(mapping, dict):
            models = list(mapping.keys()) + list(models or [])
        models = list(self._free_model_pool()) + list(models or [])
        seen, out = set(), []
        for m in (models or DEFAULTS["models"]):
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def function_call_supported(self) -> bool:
        return True

    def stream_mode_supported(self) -> bool:
        return True

    def multi_round_supported(self) -> bool:
        return True

    @classmethod
    def config_comment(cls) -> str:
        return ('配置示例:\n'
                '{\n'
                '  "base_url": "https://text.pollinations.ai/openai",\n'
                '  "api_key": "",\n'
                '  "models": ["gpt-4o-mini", "gpt-4o"],\n'
                '  "default_model": "openai",\n'
                '  "pass_through_model": false\n'
                '}\n\n'
                'base_url 不带 /v1/chat/completions。api_key 留空则不发 Authorization(无鉴权服务)。\n'
                '常见上游:\n'
                '  无鉴权: https://text.pollinations.ai/openai\n'
                '  Groq:   https://api.groq.com/openai   apikey=sc-...  pass_through_model=true\n'
                '  Cerebras: https://api.cerebras.ai/v1  apikey=csk-...\n'
                '  Gemini: https://generativelanguage.googleapis.com/v1beta/openai  apikey=AIza...\n'
                '  本地:   http://127.0.0.1:11434/v1 (Ollama)')

    @classmethod
    def supported_path(cls) -> str:
        return "/v1/chat/completions"

    def __init__(self, config: dict, eval: evaluation.AbsChannelEvaluation):
        # 每次请求现取现用,管理页改完配置立即生效。
        self.config = config if isinstance(config, dict) else {}
        self.eval = eval

    # ------------------------------------------------------------------ 配置

    def _setting(self, key: str):
        value = self.config.get(key) if isinstance(self.config, dict) else None
        if value is None or value == "":
            value = getattr(type(self), key, None)
        if value is None or value == "":
            value = DEFAULTS.get(key)
        return value

    def _base_url(self) -> str:
        return str(self._setting("base_url") or DEFAULTS["base_url"]).rstrip("/")

    def _url(self) -> str:
        base = self._base_url()
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def _timeout_secs(self) -> int:
        try:
            return int(self._setting("timeout_secs"))
        except (TypeError, ValueError):
            return DEFAULTS["timeout_secs"]

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        token = self._setting("api_key")
        if token:
            headers["Authorization"] = "Bearer " + str(token)
        extra = self._setting("extra_headers")
        if isinstance(extra, dict):
            headers.update({str(k): str(v) for k, v in extra.items()})
        return headers

    def get_config(self) -> dict:
        return dict(self.config)

    def _resolve_model(self, requested: str) -> str:
        mapping = self._setting("model_map")
        if isinstance(mapping, dict) and requested in mapping:
            return str(mapping[requested])
        if self._setting("pass_through_model"):
            return requested
        return str(self._setting("default_model") or requested)

    def _free_model_pool(self) -> list:
        pool = self.config.get("free_model_pool") if isinstance(self.config, dict) else None
        if isinstance(pool, str):
            pool = [m.strip() for m in pool.split(",") if m.strip()]
        return [str(m).strip() for m in (pool or []) if str(m).strip()]

    def _max_rotations(self) -> int:
        try:
            return max(1, int(self._setting("max_model_rotations")))
        except (TypeError, ValueError):
            return DEFAULTS["max_model_rotations"]

    def _model_candidates(self, requested: str) -> list:
        """返回本次请求要依次尝试的上游模型名。

        配了模型池就按池轮换(客户端点的模型若在池里则排最前);没配则沿用
        model_map / pass_through / default_model 的老逻辑。
        """
        pool = self._free_model_pool()

        if pool:
            head = [requested] if requested in pool else []
            tail = [m for m in pool if m != requested]
            random.shuffle(tail)
            return (head + tail)[:self._max_rotations()]

        return [self._resolve_model(requested)]

    @staticmethod
    def _is_retryable(err) -> bool:
        """只有限流/上游过载值得换模型重试;参数错误换多少个模型都是一样的错。"""
        return getattr(err, "status_code", None) in (429, 503)

    def _build_payload(self, req: request.Request, stream: bool, model: str = None) -> dict:
        payload = {
            "model": model or self._resolve_model(req.model),
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

    # ------------------------------------------------------------------ 解析

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
        finish = None
        async for raw_line in upstream.content:
            line = raw_line.decode("utf-8", "ignore").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
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
            choice = choices[0] or {}
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
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
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]

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
        payload = {
            "model": str(self._setting("default_model")),
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
        }
        payload["model"] = self._resolve_model(payload["model"])
        timeout = aiohttp.ClientTimeout(total=60)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self._url(), json=payload, headers=self._headers()) as resp:
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

    async def _stream_once(self, model: str, req: request.Request, resp_id: int) -> typing.AsyncGenerator[response.Response, None]:
        """单次尝试:用一个指定模型打上游,原样产出 Response 流。"""
        payload = self._build_payload(req, stream=True, model=model)
        timeout = aiohttp.ClientTimeout(
            total=self._timeout_secs(),
            sock_read=self._timeout_secs(),
        )
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self._url(), json=payload, headers=self._headers()) as upstream:
                    if upstream.status != 200:
                        body = await upstream.text()
                        raise exceptions.QueryHandlingError(
                            upstream.status,
                            "upstream_error",
                            "上游 {} 返回 {}: {}".format(self._url(), upstream.status, body[:500]),
                            type="api_error",
                        )
                    ctype = (upstream.headers.get("Content-Type") or "").lower()
                    if "event-stream" in ctype:
                        async for item in self._iter_stream(upstream, resp_id):
                            yield item
                        return
                    # 上游忽略了 stream=true,按一次性响应处理
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
        except exceptions.QueryHandlingError:
            raise
        except Exception as e:
            logging.warning("openai/compatible 调用上游失败: %s", e)
            raise exceptions.QueryHandlingError(
                502,
                "upstream_unreachable",
                "调用上游 {} 失败: {}".format(self._url(), e),
                type="api_error",
            ) from e

    async def query(self, req: request.Request) -> typing.AsyncGenerator[response.Response, None]:
        """按模型池轮换:某个免费模型撞到 429 就换下一个,而不是把错误直接丢给调用方。"""
        resp_id = random.randint(0, 1000000000)
        candidates = self._model_candidates(req.model)
        last_error = None

        for index, model in enumerate(candidates):
            gen = self._stream_once(model, req, resp_id)
            first = None

            try:
                first = await gen.__anext__()
            except StopAsyncIteration:
                first = None
            except Exception as e:
                last_error = e
                has_next = index < len(candidates) - 1
                if has_next and self._is_retryable(e):
                    logging.warning(
                        "openai/compatible: %s 限流/过载(%s),换下一个模型重试",
                        model,
                        getattr(e, "code", None) or e,
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
            502,
            "upstream_unreachable",
            "上游没有任何可用模型",
            type="api_error",
        )
