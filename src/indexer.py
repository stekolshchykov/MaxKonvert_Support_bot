import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".csv", ".html"}
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128


def split_text(text: str, max_len: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraphs: List[str] = []
    i = 0
    while i < len(raw_paragraphs):
        para = raw_paragraphs[i]
        # Keep markdown headings together with the next paragraph/list block.
        if para.startswith("#") and i + 1 < len(raw_paragraphs):
            nxt = raw_paragraphs[i + 1]
            combined = f"{para}\n{nxt}"
            if len(combined) <= (max_len * 2):
                paragraphs.append(combined)
                i += 2
                continue
        paragraphs.append(para)
        i += 1
    chunks = []
    for para in paragraphs:
        if len(para) <= max_len:
            chunks.append(para)
            continue
        sentences = [s.strip() for s in para.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 2 <= max_len:
                current = f"{current}. {sent}" if current else sent
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current:
            chunks.append(current)
    final_chunks = []
    for c in chunks:
        if len(c) > max_len:
            for i in range(0, len(c), max_len - overlap):
                part = c[i:i + max_len].strip()
                if part:
                    final_chunks.append(part)
        else:
            final_chunks.append(c)
    return final_chunks


def load_docs(docs_path: str) -> List[Tuple[str, str]]:
    docs = []
    p = Path(docs_path)
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
            try:
                text = f.read_text(encoding="utf-8")
                docs.append((str(f.relative_to(p)), text))
            except Exception as e:
                logger.warning("Cannot read %s: %s", f, e)
    return docs


class DocIndex:
    def __init__(self, index_dir: str, model_name: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index_file = self.index_dir / "index.faiss"
        self.meta_file = self.index_dir / "meta.pkl"
        self.index = None
        self.chunks = []
        self._load()

    def _load(self):
        if self.index_file.exists() and self.meta_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.meta_file, "rb") as f:
                    self.chunks = pickle.load(f)
                logger.info("Loaded existing index with %s chunks", len(self.chunks))
                return
            except Exception as e:
                logger.warning("Failed to load index: %s", e)
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks = []

    def build(self, docs_path: str):
        docs = load_docs(docs_path)
        new_chunks = []
        for rel_path, text in docs:
            parts = split_text(text)
            for part in parts:
                new_chunks.append({"file": rel_path, "text": part})
        if not new_chunks:
            raise RuntimeError("No documents found to index")
        texts = [c["text"] for c in new_chunks]
        logger.info("Embedding %s chunks...", len(texts))
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.chunks = new_chunks
        self.save()
        logger.info("Index built with %s chunks", len(new_chunks))

    def save(self):
        faiss.write_index(self.index, str(self.index_file))
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.chunks, f)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, dict]]:
        if not self.chunks:
            return []
        emb = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(emb)
        scores, ids = self.index.search(emb, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if 0 <= idx < len(self.chunks):
                results.append((float(score), self.chunks[idx]))
        return results
