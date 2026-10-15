# Personal AI gateway (free-one-api, patched)

A self-hosted, OpenAI-compatible endpoint you can point your own projects at — one URL, one
key, several free upstreams behind it, with automatic failover when one dies.

Based on [RockChinQ/free-one-api](https://github.com/RockChinQ/free-one-api) (AGPL-3.0).

## What this branch changes

| # | Patch | Why |
|---|---|---|
| 1 | **New `openai/compatible` adapter** | Lets any OpenAI-compatible upstream be a channel: keyless Pollinations, or your own Groq / Gemini / Cerebras / OpenRouter / Ollama. This is what turns the project into a usable personal gateway instead of a reverse-engineering toy. |
| 2 | **Multi-channel failover** (`query()` now walks the ranked channel list) | Upstream died? The request silently retries on the next channel. Upstream of this patch, one dead channel = one failed request. |
| 3 | **Streaming "first chunk" probe** | Streams can't be replayed once bytes are sent, so the first chunk is fetched *before* the SSE response is committed. Failure there → next channel, client sees nothing. |
| 4 | **gpt4free adapter fixed** | Old code probed *every* provider including image/audio ones and died on `PollinationsImage 402`. Now: text-only, no-auth providers, per-provider model names, failover between providers. |
| 5 | **Hardening** | API keys masked in logs (`sk-abc12345...C417`) instead of logged in full; `random_ad` off; `/v1/models` now lists models from your enabled channels instead of 3 hardcoded ones. |

Verified end-to-end against a live instance: non-stream completion, SSE streaming, tool-call
plumbing, failover on both paths (a deliberately dead channel was inserted, requests still
returned 200 with `attempt=2/2` in the log), and the key masking.

## Quick start

```bash
git clone https://github.com/adminewacc/free-one-api.git
cd free-one-api
git checkout personal-gateway

docker compose up -d --build          # first build compiles the web UI, give it a few minutes
docker compose exec free-one-api python3 scripts/personal_setup.py
```

The setup script prints your base URL, an `sk-` key and the model list. **Change the admin
password** first if you're going to expose this at all:

```bash
# data/config.yaml -> router.token ; then restart
docker compose restart
```

Management UI: http://localhost:3000/ (channels, keys, request logs, per-channel test).

## Client setup

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/v1", api_key="sk-...")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hi"}],
)
print(resp.choices[0].message.content)
```

Streaming, tools, `/v1/models`, `/v1/responses` and `/v1/messages` (Anthropic) all work. In
LobeChat / NextChat / Cherry Studio just set the base URL to `http://<host>:3000/v1`, paste the
`sk-` key, and pick a model from `/v1/models`.

## Channels: what you can plug in

Run `curl http://localhost:3000/v1/models -H "Authorization: Bearer sk-..."` to see what's live.

### Keyless (works out of the box, `scripts/personal_setup.py` creates both)

| Channel | Adapter | Notes |
|---|---|---|
| `pollinations-free` | `openai/compatible` | `https://text.pollinations.ai/openai`, no key. Serves `gpt-oss-20b` under the alias `openai`. |
| `gpt4free-free` | `xtekky/gpt4free` | Pinned to `Gemini`, `HuggingSpace`, `CohereForAI_C4AI_Command` — the only no-auth providers that actually answered on 2026-10-15. Fails over between them. |

### With your own free-tier key (add via UI or the API)

| Provider | `base_url` | Model example | Free tier |
|---|---|---|---|
| Groq | `https://api.groq.com/openai` | `llama-3.3-70b-versatile` | generous, fast |
| Google AI Studio | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.5-flash` | generous |
| Cerebras | `https://api.cerebras.ai/v1` | `llama3.1-8b` | generous, very fast |
| OpenRouter | `https://openrouter.ai/api/v1` | `deepseek/deepseek-chat-v3:free` | `:free` models, rate-limited |
| Ollama (local) | `http://host.docker.internal:11434/v1` | your model | free, your hardware |

Add one with the admin API (get the md5 of `router.token` first — the admin token is
`md5(config.router.token)`, not the raw one):

```bash
MD5=$(python3 -c "import hashlib;print(hashlib.md5(b'YOUR_ROUTER_TOKEN').hexdigest())")

curl -X POST http://localhost:3000/api/channel/create \
  -H "Authorization: Bearer $MD5" -H "Content-Type: application/json" \
  -d '{"id":0,"name":"groq-free","adapter":{"type":"openai/compatible","config":{
        "base_url":"https://api.groq.com/openai","api_key":"gsk_...",
        "models":["llama-3.3-70b-versatile"],"default_model":"llama-3.3-70b-versatile",
        "pass_through_model":true,"timeout_secs":120}},"model_mapping":{},"enabled":true,"latency":0}'
```

Then click **test** on the channel. More providers = more headroom; the router load-balances
across them by recent success/latency and fails over on errors.

## unoRouter free models (optional, rotating model pool)

[UnoRouter](https://unorouter.com) is an OpenAI-compatible aggregator with ~93 `:free` models —
`gpt-4o:free`, `gemini-3.6-flash:free`, `claude-haiku-5-5:free`, `deepseek-v4.1-flash:free` — that
never charge. Its catch: **1 request per minute per free model per account**, so a burst on one
model gets `429` regardless of how much credit you have.

The `openai/compatible` adapter handles that: set `free_model_pool` and it rotates through the pool
whenever one model returns 429/503, instead of surfacing the error.

```json
{
  "base_url": "https://api.unorouter.com/v1",
  "api_key": "sk-...",
  "models": ["unorouter-free"],
  "pass_through_model": true,
  "free_model_pool": ["deepseek-v4.1-flash:free", "gpt-4o:free", "gemini-3.6-flash:free",
                      "glm-4.7-flash:free", "claude-haiku-5-5:free", "qwen3.6-plus:free"],
  "max_model_rotations": 4
}
```

Set `UNOROUTER_API_KEY` before running the setup script and it creates this channel for you —
the key is read from the environment and never written into the repo.

Verified behaviour: 5 parallel requests to `unorouter-free` all returned 200, with the log showing
3 rotations (`deepseek-v4.1-flash:free → next`, `gemini-3.6-flash:free → next` twice) caused by the
per-model cap.

**Paid models stay blocked until you add balance** — they return
`403 Your balance is empty (remaining: $0.000000)`. The only free ways to get balance:

| Action | Reward |
|---|---|
| Link Discord to your account, claim on the verify panel | $0.25 once |
| Boost their Discord server | $0.50 per boost, then every 30 days |
| Vote on Top.gg / Discords.com / Discadia / DiscordServers.com | $0.013 per vote, ~every 12h per site |
| Invite someone who joins | $0.025 each |
| Reach Discord activity levels by chatting | $0.03 → $12.50, **$21.96 total** for all 9 levels |
| Wear their server tag | $0.005 per full day |
| Bug bounty (approved findings) | up to $50 each |
| Affiliate commission on referred top-ups | up to $44 across levels |

Reward balance **expires if unused**; purchased balance does not. Subscriptions
($20/$50/$100 per month) give 2x credit and cut the waiting on free models.

## Real limits (read this)

There is no such thing as an unlimited free LLM API. What you get here:

- **Pollinations anonymous: 1 concurrent request per IP.** Hitting it twice at once returns
  `429 Queue full for IP ... (max: 1)`. Sequential personal use is fine; parallel batch jobs
  will collide. Registering at `enter.pollinations.ai` raises this. Their legacy text endpoint is
  deprecated *for authenticated users*; anonymous access still works today.
- **gpt4free is a moving target.** On 2026-10-15, of the 64 providers it ships, only `Gemini`,
  `HuggingSpace` and `CohereForAI_C4AI_Command` answered without auth. Several that used to be
  free (`Groq`, `GeminiPro`, `Airforce`, `GLM`) now return `402 No cake credits — bake
  proof-of-work cakes at g4f.dev/chat`. `Cloudflare`/`DeepInfra` want a real Chrome. This is why
  patch #4 exists and why pinning `providers` in the channel config is worth doing.
- **Free tiers are rate-limited, not unlimited.** Groq/Gemini/Cerebras have per-minute and
  per-day caps; pooling several is the closest thing to "no limits" you'll get.
- **The cookie-based adapters are a different risk class.** `acheong08/ChatGPT`,
  `KoushikNavuluri/Claude-API`, `xw5xr6/revTongYi`, `dsdanielpark/Bard-API`,
  `Zai-Kun/reverse-engineered-chatgpt` drive the *consumer web UI* of those products with your
  logged-in session. That breaks their terms of service and can get your account banned, and
  most of them haven't worked since 2024 (upstream issues #117, #104). They're left in place but
  off — decide for yourself whether you want them.
- **`ds-webagent/deepseek-web`** is the most capable reverse-engineered path (real browser,
  no cookies, tool calls supported) but needs a headful Chromium and handles **one request at a
  time**. See `docs/zh-CN/DSWebAgent.md` and `docker-compose.dswebagent.yaml`.

## Security

- The admin token defaults to `12345678` in the upstream project — **change `router.token` before
  you expose anything**, and note this branch no longer logs full Bearer keys.
- `docker-compose.yml` binds to `127.0.0.1:3000` on purpose. To reach it from elsewhere put it
  behind a reverse proxy with TLS and its own auth, rather than publishing the port.
- Everything stateful is in `data/` (`free_one_api.db` = channels, keys, logs; `config.yaml`).
  Back that directory up; it is the whole instance.

## Ops

```bash
docker compose logs -f --tail=100 free-one-api   # live requests, channel errors, failovers
docker compose restart                            # after editing data/config.yaml
cp -r data data.bak                               # backup
```

Channel heartbeat disables a channel after 3 consecutive failed checks (`watchdog.heartbeat`),
so a broken upstream drops out instead of poisoning every request. Re-enable it from the UI once
it's fixed.
