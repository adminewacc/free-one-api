#!/usr/bin/env python3
"""一键初始化个人 AI 网关:建好免费渠道 + 生成个人 API Key。

服务起来之后跑一次即可(重复跑不会重复建):
    docker compose exec free-one-api python3 scripts/personal_setup.py
或者本机源码部署时:
    python3 scripts/personal_setup.py

可用环境变量覆盖:
    FOA_BASE    默认 http://127.0.0.1:3000
    FOA_CONFIG  默认 ./data/config.yaml
"""
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("FOA_BASE", "http://127.0.0.1:3000").rstrip("/")
CONFIG_PATH = os.environ.get("FOA_CONFIG", "./data/config.yaml")
KEY_NAME = "personal"

# 免鉴权渠道:开箱可用,不花一分钱。想换 provider 改 providers 列表即可。
CHANNELS = [
    {
        "name": "pollinations-free",
        "adapter": {
            "type": "openai/compatible",
            "config": {
                "base_url": "https://text.pollinations.ai/openai",
                "api_key": "",
                "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                "default_model": "openai",
                "pass_through_model": False,
                "timeout_secs": 180,
            },
        },
    },
    {
        "name": "gpt4free-free",
        "adapter": {
            "type": "xtekky/gpt4free",
            # 只保留实测可用的免鉴权 provider;全挂时自动换下一个。
            "config": {
                "providers": ["Gemini", "HuggingSpace", "CohereForAI_C4AI_Command"],
                "models": ["gpt-3.5-turbo", "gpt-4"],
                "timeout_secs": 120,
            },
        },
    },
]


def unorouter_channel():
    """有 UNOROUTER_API_KEY 就顺带建一个 unoRouter 免费渠道(模型池轮换抗限流)。"""
    key = os.environ.get("UNOROUTER_API_KEY", "").strip()
    if not key:
        return None
    pool = [
        "deepseek-v4.1-flash:free", "gpt-4o:free", "gemini-3.6-flash:free",
        "glm-4.7-flash:free", "claude-haiku-5-5:free", "nemotron-3-super-120b-a12b:free",
        "qwen3.6-plus:free", "gpt-oss-120b:free", "minimax-m2.7:free", "gemma-4-26b:free",
    ]
    return {
        "name": "unorouter-free",
        "adapter": {"type": "openai/compatible", "config": {
            "base_url": "https://api.unorouter.com/v1",
            "api_key": key,
            "models": ["unorouter-free"],
            "default_model": pool[0],
            "pass_through_model": True,
            "timeout_secs": 180,
            # 上游按"每个免费模型 1 次/分钟"限流;撞 429 就池内换一个,见适配器 max_model_rotations
            "free_model_pool": pool,
            "max_model_rotations": 4,
        }},
    }


def read_admin_token() -> str:
    if not os.path.exists(CONFIG_PATH):
        sys.exit("找不到 %s,先启动一次服务让它生成配置。" % CONFIG_PATH)
    raw = open(CONFIG_PATH, encoding="utf-8").read()
    try:
        import yaml  # noqa
        token = ((yaml.safe_load(raw) or {}).get("router") or {}).get("token")
    except Exception:
        match = re.search(r"^\s*token:\s*[\"']?([^\"'\n]+)[\"']?\s*$", raw, re.M)
        token = match.group(1).strip() if match else None
    if not token:
        sys.exit("读不到 router.token,检查 %s。" % CONFIG_PATH)
    return hashlib.md5(str(token).encode()).hexdigest()


def call(path, method="GET", body=None, token=None):
    payload = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=payload, method=method)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    if payload:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode() or "{}")


def wait_ready(timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/ping", timeout=5) as resp:
                if resp.read().decode().strip() == "pong":
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main():
    if not wait_ready():
        sys.exit("等不到 %s 就绪,先确认服务已启动。" % BASE)

    admin = read_admin_token()
    print("已连接 %s" % BASE)

    channels = (call("/api/channel/list", token=admin).get("data") or [])
    by_name = {c["name"]: c for c in channels}

    specs = list(CHANNELS)
    extra = unorouter_channel()
    if extra:
        specs.append(extra)

    for spec in specs:
        existed = by_name.get(spec["name"])
        body = {
            "id": existed["id"] if existed else 0,
            "name": spec["name"],
            "adapter": spec["adapter"],
            "model_mapping": {},
            "enabled": True,
            "latency": existed.get("latency", 0) if existed else 0,
        }
        if existed:
            res = call("/api/channel/update/%d" % existed["id"], "PUT", body, admin)
            action = "更新"
        else:
            res = call("/api/channel/create", "POST", body, admin)
            action = "新建"
        ok = res.get("code") == 0
        print("  [%s] %s -> %s" % ("ok" if ok else "失败", action, spec["name"]))
        if not ok:
            print("        ", res.get("message"))

    keys = (call("/api/key/list", token=admin).get("data") or [])
    existing = next((k for k in keys if k["name"] == KEY_NAME), None)
    if existing:
        raw = (call("/api/key/raw/%d" % existing["id"], token=admin).get("data") or {}).get("key")
    else:
        created = call("/api/key/create", "POST", {"name": KEY_NAME}, admin)
        raw = (created.get("data") or {}).get("raw")
        if created.get("code") != 0:
            print("  [!] 建 Key 失败:", created.get("message"))
            raw = None

    models = []
    try:
        models = [m["id"] for m in call("/v1/models", token=raw).get("data", [])]
    except Exception:
        pass

    print("\n" + "=" * 62)
    print("个人 API 地址: %s/v1" % BASE)
    print("个人 API Key : %s" % (raw or "(未拿到,去管理页手动建)"))
    print("可用模型     : %s" % (", ".join(models) or "(见 /v1/models)"))
    print("=" * 62)
    print("""
客户端填法(OpenAI 兼容):
  base_url = %s/v1
  api_key  = 上面那串
  model    = %s

Python:
  from openai import OpenAI
  client = OpenAI(base_url="%s/v1", api_key="%s")
  print(client.chat.completions.create(
      model="%s",
      messages=[{"role": "user", "content": "hi"}]).choices[0].message.content)

管理页(加渠道 / 看日志 / 发 Key): %s/
""" % (BASE, models[0] if models else "gpt-4o-mini", BASE, raw or "sk-...",
       models[0] if models else "gpt-4o-mini", BASE))


if __name__ == "__main__":
    main()
