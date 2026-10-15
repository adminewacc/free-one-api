"""脱敏工具:管理接口可以回显渠道配置,但绝不能回显明文密钥。

背景:adapter.dump_adapter() 的结果同时被两处使用 ——
  1. WebAPIGroup 的 /api/channel/details 返回给管理页;
  2. database/sqlite.py 序列化后写回 channel 表。
如果在 adapter.get_config() 里做脱敏,第 2 条会把 "***" 存进数据库,
任何一次渠道编辑都会把真实 key 抹掉(实测已丢过 key)。所以脱敏必须
只发生在接口出口,并且更新时要能把回显的占位符还原回库里已有的值。
"""

SENTINEL = "***"

SECRET_HINTS = (
    "api_key", "api_keys", "apikey", "access_token", "refresh_token",
    "token", "secret", "password", "passwd", "cookie", "cookies",
    "authorization", "session", "session_id", "sid",
)


def is_secret(name) -> bool:
    n = str(name).lower()
    return any(h in n for h in SECRET_HINTS)


def redact(value, key=None):
    """递归复制一份配置,密钥字段替换成占位符;空值原样保留。"""
    if isinstance(value, dict):
        return {k: redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        if is_secret(key):
            return [SENTINEL for _ in value]
        return [redact(v, key) for v in value]
    if is_secret(key):
        return value if value in (None, "") else SENTINEL
    return value


def merge_redacted(old, new, key=None):
    """把"回显占位符"还原成库里已有的真实值,其余字段以 new 为准。

    这样管理页可以放心地"读取 -> 改 -> 保存",不会把 key 覆盖成 \"***\"。
    """
    if isinstance(new, dict):
        old_dict = old if isinstance(old, dict) else {}
        return {k: merge_redacted(old_dict.get(k), v, k) for k, v in new.items()}
    if isinstance(new, list):
        if is_secret(key) and new and all(x == SENTINEL for x in new) and isinstance(old, list):
            return old
        return [merge_redacted(old[i] if isinstance(old, list) and i < len(old) else None, v, key)
                for i, v in enumerate(new)]
    if is_secret(key) and new == SENTINEL:
        return old
    return new
