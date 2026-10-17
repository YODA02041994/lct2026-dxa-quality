"""Клиент DeepSeek для эволюции признаков (копия из lct2026-fsp-jobs). Ключ — из переменной DXA_LLM_API_KEY; если её нет и указан --key-from-vps,
ключ читается по ssh с сервера владельца (в выводе и файлах не появляется)."""
from __future__ import annotations
import json
import os
import subprocess
import time
import httpx

BASE = os.environ.get("DXA_LLM_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.environ.get("DXA_LLM_MODEL", "deepseek-flash")
_KEY = os.environ.get("DXA_LLM_API_KEY", "")
SPEND = {"calls": 0, "input": 0, "output": 0, "cache_hit": 0}


def key() -> str:
    global _KEY
    if not _KEY:
        _KEY = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "vps", "sed -n 's/^DEEPSEEK_REAL_API_KEY=//p' /home/knee_bot/keys.env | tr -d '\"' | head -1"],
                              capture_output=True, text=True, timeout=40).stdout.strip()
    if not _KEY:
        raise SystemExit("нет ключа модели: задайте DXA_LLM_API_KEY")
    return _KEY


def chat(messages: list[dict], *, json_mode: bool = False, max_tokens: int = 4000, temperature: float = 0.7, timeout: float = 180, retries: int = 3) -> str:
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "thinking": {"type": "disabled"}}     # рассуждения выключены: иначе они съедают лимит токенов
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    last = None
    for attempt in range(retries):
        try:
            r = httpx.post(BASE.rstrip("/") + "/chat/completions", json=body, timeout=timeout,
                           headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json"})
            r.raise_for_status()
            d = r.json()
            u = d.get("usage", {})
            SPEND["calls"] += 1; SPEND["input"] += u.get("prompt_tokens", 0); SPEND["output"] += u.get("completion_tokens", 0)
            SPEND["cache_hit"] += u.get("prompt_cache_hit_tokens", 0)
            return d["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"модель не ответила: {last}")


def cost_usd() -> float:
    return round((SPEND["input"] * 0.14 + SPEND["output"] * 0.28) / 1e6, 4)


def extract_code(text: str) -> str:
    """Вытащить первый блок ```python ... ``` или весь текст."""
    if "```" in text:
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i]
            if block.startswith("python"):
                block = block[len("python"):]
            return block.strip("\n")
    return text
