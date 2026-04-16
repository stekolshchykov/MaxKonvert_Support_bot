"""Shared assistant pipeline.

Entry point: Assistant.process_message(...)
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from actions import ActionExecutor, ActionRegistry, parse_action_requests
from config import Config
from indexer import DocIndex
from providers import get_provider, ModelProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistence helpers (moved from bot.py)
# ---------------------------------------------------------------------------


def read_json(path: str) -> dict[str, Any]:
    try:
        p = Path(path)
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8") or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def append_jsonl(path: str, payload: dict[str, Any]) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("Failed to write JSONL: %s", path)


# ---------------------------------------------------------------------------
# Assistant core
# ---------------------------------------------------------------------------


class Assistant:
    """Central processing unit for all input channels."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        actions_config_path: str | None = None,
    ):
        self.provider = provider or get_provider()
        self.index: DocIndex | None = None
        self.index_signature: tuple[int, int, int, int] | None = None
        self.conversation_memory = defaultdict(lambda: deque(maxlen=max(2, Config.CONTEXT_TURNS * 2)))
        self.question_state = read_json(Config.QUESTIONS_STATE_FILE)
        self.state_lock = threading.Lock()
        self.index_lock = threading.Lock()

        # Action layer
        registry = ActionRegistry(actions_config_path or Config.ACTIONS_CONFIG_PATH)
        self.executor = ActionExecutor(registry, timeout=30.0)

        # Ensure dirs/files exist
        Path(Config.LOG_PATH).mkdir(parents=True, exist_ok=True)
        Path(Config.QUESTIONS_DIR).mkdir(parents=True, exist_ok=True)
        Path(Config.QUESTIONS_LOG_FILE).touch(exist_ok=True)
        Path(Config.NEW_QUESTIONS_LOG_FILE).touch(exist_ok=True)
        Path(Config.UNANSWERED_LOG_FILE).touch(exist_ok=True)

    # ------------------------------------------------------------------
    # Index management (preserved from bot.py)
    # ------------------------------------------------------------------

    def _current_index_signature(self) -> tuple[int, int, int, int]:
        index_dir = Path(Config.INDEX_PATH)
        files = [index_dir / "index.faiss", index_dir / "meta.pkl"]
        values: list[int] = []
        for f in files:
            try:
                st = f.stat()
                values.extend([int(st.st_mtime_ns), int(st.st_size)])
            except FileNotFoundError:
                values.extend([0, 0])
        return (values[0], values[1], values[2], values[3])

    def _maybe_reload_index(self, force: bool = False) -> DocIndex:
        sig = self._current_index_signature()
        if force or self.index is None:
            logger.info("Initializing index...")
            self.index = DocIndex(Config.INDEX_PATH, Config.EMBEDDING_MODEL)
            self.index_signature = self._current_index_signature()
            return self.index
        if self.index_signature is not None and sig != self.index_signature and any(sig):
            logger.info("Detected index update on disk, reloading in-memory index")
            previous_chunks = len(self.index.chunks)
            candidate = DocIndex(Config.INDEX_PATH, Config.EMBEDDING_MODEL)
            if candidate.chunks:
                self.index = candidate
                self.index_signature = self._current_index_signature()
                logger.info(
                    "Index reloaded successfully (%s -> %s chunks)",
                    previous_chunks,
                    len(candidate.chunks),
                )
            else:
                logger.warning("Index reload produced empty chunks; keeping current in-memory index")
        return self.index  # type: ignore[return-value]

    def get_index(self) -> DocIndex:
        with self.index_lock:
            return self._maybe_reload_index(force=False)

    def ensure_index(self) -> DocIndex:
        with self.index_lock:
            idx = self._maybe_reload_index(force=False)
            if not idx.chunks:
                logger.info("Index empty, building...")
                idx.build(Config.DOCS_PATH)
                self.index_signature = self._current_index_signature()
            return idx

    # ------------------------------------------------------------------
    # Conversation memory (preserved from bot.py)
    # ------------------------------------------------------------------

    def add_dialog_turn(self, key: str, role: str, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.conversation_memory[key].append({"role": role, "text": text})

    def get_recent_user_context(self, key: str) -> str:
        history = self.conversation_memory.get(key)
        if not history:
            return ""
        user_turns = [t["text"] for t in history if t["role"] == "user"]
        if not user_turns:
            return ""
        return "\n".join(user_turns[-max(1, Config.RETRIEVAL_HISTORY_TURNS):])

    def get_dialog_history_text(self, key: str) -> str:
        history = self.conversation_memory.get(key)
        if not history:
            return "(история пуста)"
        lines = []
        for turn in history:
            prefix = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {turn['text']}")
        text = "\n".join(lines)
        return text[-Config.MAX_CONTEXT_CHARS:] if len(text) > Config.MAX_CONTEXT_CHARS else text

    # ------------------------------------------------------------------
    # Question logging (preserved from bot.py)
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_question(text: str) -> str:
        s = (text or "").lower().strip()
        s = s.replace("ё", "е")
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^a-zа-я0-9 ]", "", s)
        return s.strip()

    def build_question_event(
        self,
        user_text: str,
        status: str,
        best_score: float,
        key: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "conversation_key": key,
            "status": status,
            "best_score": round(best_score, 6),
            "question": user_text,
            "normalized_question": self.normalize_question(user_text),
        }
        if metadata:
            event["metadata"] = metadata
        return event

    def record_question_if_new(self, user_text: str, key: str, best_score: float) -> None:
        normalized = self.normalize_question(user_text)
        if not normalized:
            return
        with self.state_lock:
            count = int(self.question_state.get(normalized, 0)) + 1
            self.question_state[normalized] = count
            write_json_atomic(Config.QUESTIONS_STATE_FILE, self.question_state)
        if count == 1:
            event = self.build_question_event(user_text, "new_question", best_score, key)
            event["occurrence_count"] = count
            append_jsonl(Config.NEW_QUESTIONS_LOG_FILE, event)

    def log_received(self, user_text: str, key: str, best_score: float, metadata: dict | None = None) -> None:
        append_jsonl(
            Config.QUESTIONS_LOG_FILE,
            self.build_question_event(user_text, "received", best_score, key, metadata),
        )

    def log_status(self, status: str, user_text: str, key: str, best_score: float, metadata: dict | None = None) -> None:
        append_jsonl(
            Config.QUESTIONS_LOG_FILE,
            self.build_question_event(user_text, status, best_score, key, metadata),
        )
        if status.startswith("needs_kb_update") or status in ("model_timeout_or_error",):
            append_jsonl(
                Config.UNANSWERED_LOG_FILE,
                self.build_question_event(user_text, status, best_score, key, metadata),
            )

    # ------------------------------------------------------------------
    # Prompt building (preserved from bot.py)
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        user_text: str,
        history_text: str,
        docs_text: str,
        action_results_text: str = "",
    ) -> str:
        action_section = ""
        if self.executor.registry.actions:
            action_section = (
                "\nЕсли для ответа требуется вызвать внешний API, используй доступные действия.\n"
                "Доступные действия:\n"
                + "\n".join(self.executor.registry.list_descriptions())
                + "\nЧтобы вызвать действие, вставь в ответ строку:\n"
                'TOOL:{"name":"<имя_действия>","params":{...},"body":{...}}\n'
                "Затем я выполню действие и верну тебе результат для финального ответа.\n"
            )
        if action_results_text:
            action_section += f"\n[Результаты вызванных действий]\n{action_results_text}\n"

        return (
            f"{Config.BOT_ROLE}\n"
            f"{Config.BOT_EXTRA_RULES}\n\n"
            "Отвечай только по фрагментам базы знаний ниже.\n"
            "Не добавляй маркетинговые CTA и не придумывай факты.\n"
            "Если во фрагментах есть прямое определение, используй его почти дословно.\n"
            "Стиль ответа: кратко, по делу, 1-4 предложения.\n"
            f"{action_section}\n"
            "[Chat history with this user]\n"
            f"{history_text}\n\n"
            "[Knowledge base fragments]\n"
            f"{docs_text}\n\n"
            f"[Current question]\n{user_text}\n\n"
            "[Answer]"
        )

    def build_docs_text(self, results: list[tuple[float, dict[str, Any]]]) -> str:
        parts = []
        for _, payload in results[: max(1, Config.MAX_DOC_FRAGMENTS)]:
            file_name = payload.get("file", "unknown")
            text = (payload.get("text", "") or "")[: Config.MAX_DOC_SNIPPET_CHARS]
            parts.append(f"[{file_name}]\n{text}")
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Retrieval helpers (preserved from bot.py)
    # ------------------------------------------------------------------

    def rerank_results_by_query_tokens(
        self, user_text: str, results: list[tuple[float, dict[str, Any]]]
    ) -> list[tuple[float, dict[str, Any]]]:
        tokens = self.query_tokens(user_text)
        if not tokens or not results:
            return results
        rescored: list[tuple[float, int, float, dict[str, Any]]] = []
        for score, payload in results:
            text = (payload.get("text", "") or "").lower().replace("ё", "е")
            overlap = sum(1 for tok in tokens if tok in text)
            adjusted = float(score) + (0.08 * overlap)
            if overlap and len(tokens) == 1 and tokens[0] in text:
                adjusted += 0.12
            rescored.append((adjusted, overlap, float(score), payload))
        rescored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
        return [(row[2], row[3]) for row in rescored]

    def query_tokens(self, text: str) -> list[str]:
        parts = re.findall(r"[a-zа-я0-9]+", (text or "").lower().replace("ё", "е"))
        blacklist = {
            "что",
            "как",
            "какой",
            "какие",
            "есть",
            "это",
            "а",
            "и",
            "по",
            "в",
            "на",
            "ли",
            "за",
            "к",
            "у",
            "из",
            "про",
            "подскажите",
            "скажите",
            "пожалуйста",
        }
        return [p for p in parts if len(p) >= 3 and p not in blacklist]

    def is_followup_query(self, text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        followup_starts = (
            "а ",
            "и ",
            "это ",
            "а это",
            "а как",
            "а где",
            "а что",
            "а какой",
            "а какие",
            "а когда",
            "а сколько",
            "по этому",
            "по нему",
            "и это",
        )
        if lowered.startswith(followup_starts):
            return True
        pronouns = {"это", "этот", "эта", "эти", "они", "он", "она", "его", "ее", "её", "их"}
        tokens = [t for t in re.split(r"\W+", lowered) if t]
        return len(tokens) <= 7 and any(t in pronouns for t in tokens)

    def is_definition_query(self, text: str) -> bool:
        lowered = (text or "").strip().lower()
        patterns = (
            "что такое ",
            "кто такой ",
            "кто такая ",
            "что за ",
            "define ",
            "what is ",
        )
        return lowered.startswith(patterns)

    # ------------------------------------------------------------------
    # Quality checks (preserved from bot.py)
    # ------------------------------------------------------------------

    def is_unknown_answer(self, text: str) -> bool:
        lowered = (text or "").lower()
        markers = [
            "к сожалению",
            "нет информации",
            "не найден",
            "не указ",
            "not found in documentation",
            "no information in the documentation",
        ]
        return any(marker in lowered for marker in markers)

    def contains_unbacked_claims(self, answer: str, docs_text: str) -> bool:
        answer_tokens = {
            t
            for t in re.findall(r"[a-zа-я0-9]+", (answer or "").lower().replace("ё", "е"))
            if len(t) >= 4
        }
        docs_tokens = {
            t
            for t in re.findall(r"[a-zа-я0-9]+", (docs_text or "").lower().replace("ё", "е"))
            if len(t) >= 4
        }
        if not answer_tokens or not docs_tokens:
            return False
        ignore = {
            "maxkonvert",
            "можно",
            "нужно",
            "лучше",
            "вариант",
            "уточните",
            "подскажите",
            "подобрать",
            "запуск",
            "запуска",
            "трафик",
            "трафика",
            "условия",
            "формат",
            "форматы",
            "geo",
        }
        core = {t for t in answer_tokens if t not in ignore}
        if not core:
            return False
        unsupported = {t for t in core if t not in docs_tokens}
        ratio = len(unsupported) / max(1, len(core))
        return len(unsupported) >= 6 and ratio >= 0.55

    # ------------------------------------------------------------------
    # Fallback builders (preserved from bot.py)
    # ------------------------------------------------------------------

    def build_extractive_fallback(
        self,
        user_text: str,
        results: list[tuple[float, dict[str, Any]]],
        *,
        include_cta: bool = True,
        lead: str = "По вашему запросу вижу рабочие варианты:",
    ) -> str:
        if not results:
            return ""
        tokens = self.query_tokens(user_text)

        def collect_lines(require_tokens: bool) -> list[tuple[str, str]]:
            lines: list[tuple[str, str]] = []
            seen = set()
            for _, payload in results[: max(1, Config.MAX_DOC_FRAGMENTS)]:
                file_name = payload.get("file", "unknown")
                text = payload.get("text", "") or ""
                for raw in re.split(r"[\n\r]+", text):
                    line = raw.strip(" -*\t")
                    if len(line) < 18:
                        continue
                    low = line.lower().replace("ё", "е")
                    if require_tokens and tokens and not any(tok in low for tok in tokens):
                        continue
                    key = self.normalize_question(line)
                    if key in seen:
                        continue
                    seen.add(key)
                    lines.append((line, file_name))
                    if len(lines) >= 4:
                        break
                if len(lines) >= 4:
                    break
            return lines

        lines = collect_lines(require_tokens=True)
        if not lines:
            lines = collect_lines(require_tokens=False)
        if not lines:
            return ""
        snippets = [line for line, _ in lines[:3]]
        if not snippets:
            return ""
        body = " ".join(snippets)
        body = re.sub(r"\s+", " ", body).strip()
        if include_cta:
            return f"{lead} {body} {Config.SALES_DEFAULT_CTA}"
        return f"{lead} {body}"

    def build_sales_manager_fallback(
        self, user_text: str, results: list[tuple[float, dict[str, Any]]], *, low_match: bool = False
    ) -> str:
        extractive = self.build_extractive_fallback(user_text, results)
        if extractive:
            return extractive
        if low_match:
            return (
                "Спасибо за вопрос. "
                "Подберу для вас оптимальный сценарий запуска под ваш трафик и цели. "
                f"{Config.SALES_DEFAULT_CTA}"
            )
        return (
            "Могу быстро собрать для вас понятный план запуска и монетизации. "
            f"{Config.SALES_DEFAULT_CTA}"
        )

    def build_docs_grounded_fallback(
        self, user_text: str, results: list[tuple[float, dict[str, Any]]]
    ) -> str:
        extractive = self.build_extractive_fallback(
            user_text,
            results,
            include_cta=False,
            lead="По базе знаний:",
        )
        if extractive:
            return extractive
        return "В базе знаний пока нет точного ответа на этот вопрос."

    def build_direct_definition_answer(
        self, user_text: str, results: list[tuple[float, dict[str, Any]]]
    ) -> str:
        tokens = self.query_tokens(user_text)
        if not results or not tokens:
            return ""
        candidates: list[str] = []
        for _, payload in results:
            text = payload.get("text", "") or ""
            for raw in text.splitlines():
                line = raw.strip(" \t-*")
                if len(line) < 8:
                    continue
                low = line.lower().replace("ё", "е")
                if not any(tok in low for tok in tokens):
                    continue
                if (" это " in low) or ("- это" in low) or ("— это" in low):
                    candidates.append(line)
                    break
            if not candidates:
                compact = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
                if compact:
                    low = compact.lower().replace("ё", "е")
                    if any(tok in low for tok in tokens) and (
                        (" это " in low) or ("- это" in low) or ("— это" in low)
                    ):
                        candidates.append(compact)
            if candidates:
                break
        if not candidates:
            return ""
        line = re.sub(r"^[#]+\s*", "", candidates[0]).strip()
        line = re.sub(r"\s+", " ", line).strip()
        line = line.rstrip(" .") + "."
        return line

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------

    async def process_message(
        self,
        user_text: str,
        conversation_key: str,
        channel: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a single user message and return a structured result.

        Returns dict with keys:
            - answer: str
            - route: str
            - best_score: float
            - elapsed_ms: int
            - metadata: dict
        """
        started = datetime.now(timezone.utc)
        meta = {"channel": channel, **(metadata or {})}

        # Retrieval
        history_for_retrieval = self.get_recent_user_context(conversation_key)
        retrieval_query = user_text
        if history_for_retrieval and self.is_followup_query(user_text):
            retrieval_query = f"{history_for_retrieval}\n{user_text}"
        definition_mode = self.is_definition_query(user_text)

        idx = self.ensure_index()
        raw_results = idx.search(retrieval_query, top_k=max(Config.TOP_K, 8))
        best_score = raw_results[0][0] if raw_results else 0.0
        results = self.rerank_results_by_query_tokens(user_text, raw_results)[: Config.TOP_K]
        docs_text = self.build_docs_text(results)

        self.log_received(user_text, conversation_key, best_score, meta)
        self.record_question_if_new(user_text, conversation_key, best_score)

        # Direct definition shortcut
        direct_answer = self.build_direct_definition_answer(user_text, results)
        if direct_answer:
            self.log_status("direct_definition_from_docs", user_text, conversation_key, best_score, meta)
            self.add_dialog_turn(conversation_key, "user", user_text)
            self.add_dialog_turn(conversation_key, "assistant", direct_answer)
            elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
            return {
                "answer": direct_answer,
                "route": "direct_definition",
                "best_score": best_score,
                "elapsed_ms": elapsed_ms,
                "metadata": meta,
            }

        # Low match paths
        if best_score < Config.SIMILARITY_THRESHOLD:
            fallback = (
                self.build_docs_grounded_fallback(user_text, results)
                if definition_mode
                else self.build_sales_manager_fallback(user_text, results, low_match=True)
            )
            self.log_status("needs_kb_update_low_match", user_text, conversation_key, best_score, meta)
            self.add_dialog_turn(conversation_key, "user", user_text)
            self.add_dialog_turn(conversation_key, "assistant", fallback)
            elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
            return {
                "answer": fallback,
                "route": "low_match_fallback",
                "best_score": best_score,
                "elapsed_ms": elapsed_ms,
                "metadata": meta,
            }

        if best_score < Config.LOW_CONFIDENCE_MODEL_SCORE:
            fallback = (
                self.build_docs_grounded_fallback(user_text, results)
                if definition_mode
                else self.build_sales_manager_fallback(user_text, results, low_match=True)
            )
            self.log_status("needs_kb_update_low_confidence", user_text, conversation_key, best_score, meta)
            self.add_dialog_turn(conversation_key, "user", user_text)
            self.add_dialog_turn(conversation_key, "assistant", fallback)
            elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
            return {
                "answer": fallback,
                "route": "low_confidence_fallback",
                "best_score": best_score,
                "elapsed_ms": elapsed_ms,
                "metadata": meta,
            }

        # Model path with optional action loop
        history_text = self.get_dialog_history_text(conversation_key)
        action_results_text = ""
        answer = ""
        route = "model"

        # Up to 2 action turns to avoid infinite loops
        for _ in range(2):
            prompt = self.build_prompt(
                user_text=user_text,
                history_text=history_text,
                docs_text=docs_text,
                action_results_text=action_results_text,
            )
            system = Config.BOT_ROLE
            raw_answer = await self.provider.generate(prompt, system=system)
            if not raw_answer:
                break

            action_requests = parse_action_requests(raw_answer)
            if not action_requests:
                answer = raw_answer
                break

            # Execute the first valid action request and accumulate results
            executed = False
            for req in action_requests:
                if req.get("type") == "tool":
                    result = await self.executor.execute(
                        req["name"],
                        query_params=req.get("params"),
                        body=req.get("body"),
                    )
                    action_results_text += (
                        f"\nДействие '{req['name']}' результат: {json.dumps(result, ensure_ascii=False)}\n"
                    )
                    executed = True
                    break  # one action per loop iteration
                # Raw HTTP requests are ignored for security unless they map to a known action
            if not executed:
                answer = raw_answer
                break

        # Post-process model answer
        if not answer:
            answer = (
                self.build_docs_grounded_fallback(user_text, results)
                if definition_mode
                else self.build_sales_manager_fallback(user_text, results)
            )
            route = "fallback_after_model_error"
            if answer:
                self.log_status("sales_fallback_after_model_error", user_text, conversation_key, best_score, meta)
            self.log_status("model_timeout_or_error", user_text, conversation_key, best_score, meta)
        elif self.is_unknown_answer(answer):
            answer = (
                self.build_docs_grounded_fallback(user_text, results)
                if definition_mode
                else self.build_sales_manager_fallback(user_text, results, low_match=best_score < 0.35)
            )
            route = "fallback_unknown_answer"
            self.log_status("sales_override_model_no_answer", user_text, conversation_key, best_score, meta)
            self.log_status("needs_kb_update_model_no_answer", user_text, conversation_key, best_score, meta)
        elif self.contains_unbacked_claims(answer, docs_text):
            self.log_status("model_answer_with_unbacked_tokens", user_text, conversation_key, best_score, meta)

        self.add_dialog_turn(conversation_key, "user", user_text)
        self.add_dialog_turn(conversation_key, "assistant", answer)
        elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        logger.info(
            "Answered message key=%s score=%.4f elapsed_ms=%s model=%s route=%s provider=%s",
            conversation_key,
            best_score,
            elapsed_ms,
            self.provider.model_id,
            route,
            self.provider.name,
        )
        return {
            "answer": answer,
            "route": route,
            "best_score": best_score,
            "elapsed_ms": elapsed_ms,
            "metadata": meta,
        }
