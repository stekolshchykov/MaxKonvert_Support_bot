#!/usr/bin/env python3
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bot import (  # noqa: E402
    LOW_CONFIDENCE_MODEL_SCORE,
    TOP_K,
    SIMILARITY_THRESHOLD,
    ask_ollama,
    build_docs_text,
    build_prompt,
    build_sales_manager_fallback,
    contains_unbacked_claims,
    ensure_index,
    get_index,
    is_unknown_answer,
)

SMOKE_QUESTIONS = [
    "Привет, я хочу запускать трафик. С чего лучше начать?",
    "Какие услуги предлагает MaxKonvert?",
    "Какие есть способы вывода и когда выплаты?",
    "Можно ли припарковать домен и какие форматы лучше для RU трафика?",
    "У меня 10к кликов в сутки, что посоветуете для старта?",
    "Есть ли у вас API для интеграции?",
]


async def ask_once(question: str) -> tuple[int, float, str]:
    t0 = time.time()
    ensure_index()
    results = get_index().search(question, top_k=TOP_K)
    best_score = results[0][0] if results else 0.0
    if best_score < SIMILARITY_THRESHOLD:
        answer = build_sales_manager_fallback(question, results, low_match=True)
    elif best_score < LOW_CONFIDENCE_MODEL_SCORE:
        answer = build_sales_manager_fallback(question, results, low_match=True)
    else:
        docs_text = build_docs_text(results)
        prompt = build_prompt(question, "(история пуста)", docs_text)
        answer = await ask_ollama(prompt)
        if not answer or is_unknown_answer(answer) or contains_unbacked_claims(answer, docs_text):
            answer = build_sales_manager_fallback(
                question, results, low_match=best_score < 0.35
            )
    elapsed_ms = int((time.time() - t0) * 1000)
    return elapsed_ms, best_score, answer


def status_text(answer: str) -> str:
    lowered = answer.lower()
    bad_markers = ["нет в документации", "не найдено", "я не знаю"]
    return "OK" if not any(marker in lowered for marker in bad_markers) else "BAD_STYLE"


async def main() -> int:
    worst = 0
    for i, q in enumerate(SMOKE_QUESTIONS, start=1):
        ms, score, answer = await ask_once(q)
        worst = max(worst, ms)
        print(f"[{i}] {q}")
        print(f"    ms={ms} score={score:.4f} style={status_text(answer)}")
        print(f"    answer={answer[:420].replace(chr(10), ' ')}")
    print(f"\nWorst latency: {worst} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
