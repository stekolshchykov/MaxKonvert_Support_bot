#!/usr/bin/env python3
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assistant import Assistant
from config import Config

SMOKE_QUESTIONS = [
    "Привет, я хочу запускать трафик. С чего лучше начать?",
    "Какие услуги предлагает MaxKonvert?",
    "Какие есть способы вывода и когда выплаты?",
    "Можно ли припарковать домен и какие форматы лучше для RU трафика?",
    "У меня 10к кликов в сутки, что посоветуете для старта?",
    "Есть ли у вас API для интеграции?",
]


async def ask_once(assistant: Assistant, question: str) -> tuple[int, float, str, str]:
    t0 = time.time()
    result = await assistant.process_message(
        user_text=question,
        conversation_key="smoke:test",
        channel="smoke_test",
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    return elapsed_ms, result["best_score"], result["answer"], result["route"]


def status_text(answer: str) -> str:
    lowered = answer.lower()
    bad_markers = ["нет в документации", "не найдено", "я не знаю"]
    return "OK" if not any(marker in lowered for marker in bad_markers) else "BAD_STYLE"


async def main() -> int:
    assistant = Assistant()
    assistant.ensure_index()
    print(f"Provider: {assistant.provider.name}  Model: {assistant.provider.model_id}")
    worst = 0
    for i, q in enumerate(SMOKE_QUESTIONS, start=1):
        ms, score, answer, route = await ask_once(assistant, q)
        worst = max(worst, ms)
        print(f"[{i}] {q}")
        print(f"    ms={ms} score={score:.4f} route={route} style={status_text(answer)}")
        print(f"    answer={answer[:420].replace(chr(10), ' ')}")
    print(f"\nWorst latency: {worst} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
