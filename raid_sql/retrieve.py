"""Structure-aware few-shot retrieval over Spider train."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Protocol

from config import Settings, get_settings
from raid_sql.metrics import StageMetrics, cost_usd, estimate_tokens
from raid_sql.spider_data import load_spider_split, sql_skeleton


def _hardness_tag(sql: str) -> str:
    s = (sql or "").upper()
    nested = any(
        k in s for k in (" INTERSECT ", " UNION ", " EXCEPT ", " IN ", " NOT IN ")
    )
    join = " JOIN " in s
    if nested:
        return "NESTED"
    if join:
        return "NON-NESTED"
    return "EASY"


class Embedder(Protocol):
    model_name: str

    def embed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> tuple[list[list[float]], int]:
        """Return (embeddings, estimated_or_actual input tokens)."""
        ...


class LocalEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> tuple[list[list[float]], int]:
        del task_type
        vectors = self._model.encode(texts, normalize_embeddings=True).tolist()
        tokens = sum(estimate_tokens(t) for t in texts)
        return vectors, tokens


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError("Set OPENAI_API_KEY for OpenAI embeddings")
        kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._client = OpenAI(**kwargs)
        self.model_name = settings.embedding_model
        self._rate = settings.openai_embedding_per_m

    def embed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> tuple[list[list[float]], int]:
        del task_type  # OpenAI embeddings API has no task_type
        # OpenAI allows batches; keep chunks modest
        all_vecs: list[list[float]] = []
        total_tokens = 0
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            resp = self._client.embeddings.create(model=self.model_name, input=chunk)
            # ensure order by index
            ordered = sorted(resp.data, key=lambda d: d.index)
            all_vecs.extend([list(d.embedding) for d in ordered])
            usage = getattr(resp, "usage", None)
            total_tokens += int(getattr(usage, "total_tokens", 0) or 0) or sum(
                estimate_tokens(t) for t in chunk
            )
        return all_vecs, total_tokens


class GeminiEmbedder:
    """Vertex / Gemini embeddings — pairs with gemini-2.5-flash chat.

    Recommended model: ``text-embedding-004`` (or ``gemini-embedding-001``).
    Uses the same auth as Gemini chat (API key or Vertex SA).
    """

    def __init__(self, settings: Settings) -> None:
        from raid_sql.llm import _build_gemini_client

        self.settings = settings
        self._client = _build_gemini_client(settings)
        self.model_name = settings.embedding_model or "text-embedding-004"
        self._rate = settings.gemini_embedding_per_m
        self._dim = settings.gemini_embedding_dims

    def _embed_once(
        self,
        texts: list[str],
        *,
        task_type: str,
    ) -> tuple[list[list[float]], int]:
        from google.genai import types

        all_vecs: list[list[float]] = []
        total_tokens = 0
        batch_size = 16
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            cfg_kwargs: dict[str, Any] = {"task_type": task_type}
            if self._dim:
                cfg_kwargs["output_dimensionality"] = self._dim
            resp = self._client.models.embed_content(
                model=self.model_name,
                contents=chunk,
                config=types.EmbedContentConfig(**cfg_kwargs),
            )
            embeddings = getattr(resp, "embeddings", None) or []
            for emb in embeddings:
                values = list(getattr(emb, "values", None) or [])
                all_vecs.append(_l2_normalize(values))
            usage = getattr(resp, "metadata", None) or getattr(resp, "usage_metadata", None)
            batch_tok = 0
            if usage is not None:
                batch_tok = int(
                    getattr(usage, "billable_character_count", 0)
                    or getattr(usage, "total_tokens", 0)
                    or getattr(usage, "prompt_token_count", 0)
                    or 0
                )
            if not batch_tok:
                batch_tok = sum(estimate_tokens(t) for t in chunk)
            total_tokens += batch_tok
        if len(all_vecs) != len(texts):
            if len(texts) > 1 and len(all_vecs) != len(texts):
                all_vecs, total_tokens = [], 0
                for t in texts:
                    cfg_kwargs = {"task_type": task_type}
                    if self._dim:
                        cfg_kwargs["output_dimensionality"] = self._dim
                    resp = self._client.models.embed_content(
                        model=self.model_name,
                        contents=[t],
                        config=types.EmbedContentConfig(**cfg_kwargs),
                    )
                    embeddings = getattr(resp, "embeddings", None) or []
                    if not embeddings:
                        raise RuntimeError(
                            f"Gemini embed returned no vectors for model={self.model_name}"
                        )
                    values = list(getattr(embeddings[0], "values", None) or [])
                    all_vecs.append(_l2_normalize(values))
                    total_tokens += estimate_tokens(t)
            elif len(all_vecs) != len(texts):
                raise RuntimeError(
                    f"Gemini embed count mismatch: got {len(all_vecs)} for {len(texts)} texts"
                )
        return all_vecs, total_tokens

    def embed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> tuple[list[list[float]], int]:
        import random
        import time

        from raid_sql.llm import (
            _build_gemini_client,
            _is_network_error,
            _is_retryable,
        )

        if not texts:
            return [], 0
        attempts = max(1, self.settings.llm_max_retries)
        last_exc: Optional[BaseException] = None
        for attempt in range(attempts):
            try:
                return self._embed_once(texts, task_type=task_type)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt + 1 >= attempts or not _is_retryable(exc):
                    raise
                if _is_network_error(exc):
                    try:
                        self._client = _build_gemini_client(self.settings)
                    except Exception:  # noqa: BLE001
                        pass
                sleep_s = self.settings.llm_retry_base_sec * (2**attempt) + random.uniform(
                    0, 1.0
                )
                if _is_network_error(exc):
                    sleep_s = max(sleep_s, 5.0)
                sleep_s = min(sleep_s, 90.0)
                print(
                    f"  [retry {attempt + 1}/{attempts} embed={self.model_name}] "
                    f"{exc} — sleep {sleep_s:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_s)
        assert last_exc is not None
        raise last_exc


def _l2_normalize(vec: list[float]) -> list[float]:
    import math

    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def create_embedder(settings: Settings) -> Embedder:
    provider = (settings.embedding_provider or "openai").lower().strip()
    if provider in {"openai", "gpt"}:
        return OpenAIEmbedder(settings)
    if provider in {"local", "sentence-transformers", "minilm"}:
        return LocalEmbedder(settings.embedding_model)
    if provider in {"gemini", "google", "vertex"}:
        return GeminiEmbedder(settings)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER={settings.embedding_provider!r}")


class FewShotRetriever:
    """Chroma-backed retrieval with optional structure boost."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._collection = None
        self._embedder: Optional[Embedder] = None

    def _ensure(self) -> None:
        if self._collection is not None:
            return
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._embedder = create_embedder(self.settings)
        client = chromadb.PersistentClient(
            path=str(self.settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = client.get_or_create_collection(
            name=self.settings.fewshot_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def build_index(self, data_dir: Path, *, reset: bool = False) -> int:
        self._ensure()
        assert self._collection is not None and self._embedder is not None
        if reset:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            client = chromadb.PersistentClient(
                path=str(self.settings.chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            try:
                client.delete_collection(self.settings.fewshot_collection)
            except Exception:  # noqa: BLE001
                pass
            self._collection = client.get_or_create_collection(
                name=self.settings.fewshot_collection,
                metadata={"hnsw:space": "cosine"},
            )

        examples = load_spider_split(data_dir, "train")
        batch_size = 64
        n = 0
        embed_tokens = 0
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            ids = [f"train-{start + i}" for i in range(len(batch))]
            docs = []
            metas = []
            for ex in batch:
                q = ex["question"]
                sql = ex.get("query") or ex.get("SQL") or ""
                sk = sql_skeleton(sql)
                docs.append(f"{q}\n{sk}")
                metas.append(
                    {
                        "question": q,
                        "sql": sql,
                        "db_id": ex["db_id"],
                        "hardness": _hardness_tag(sql),
                        "skeleton": sk[:500],
                    }
                )
            embeddings, tokens = self._embedder.embed(
                docs, task_type="RETRIEVAL_DOCUMENT"
            )
            embed_tokens += tokens
            self._collection.add(
                ids=ids, documents=docs, embeddings=embeddings, metadatas=metas
            )
            n += len(batch)
            print(f"  indexed {n}/{len(examples)}", flush=True)

        emb_cost = 0.0
        prov = (self.settings.embedding_provider or "").lower()
        if prov in {"openai", "gpt"}:
            emb_cost = (embed_tokens / 1_000_000.0) * self.settings.openai_embedding_per_m
        elif prov in {"gemini", "google", "vertex"}:
            emb_cost = (embed_tokens / 1_000_000.0) * self.settings.gemini_embedding_per_m

        meta_path = self.settings.chroma_dir / "index_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(
                {
                    "n": n,
                    "collection": self.settings.fewshot_collection,
                    "embedding_provider": self.settings.embedding_provider,
                    "embedding_model": self.settings.embedding_model,
                    "embed_tokens": embed_tokens,
                    "embed_cost_usd_est": emb_cost,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"  embed_tokens={embed_tokens} embed_cost_est=${emb_cost:.4f}",
            flush=True,
        )
        return n

    def retrieve(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        prefer_hardness: Optional[str] = None,
        prefer_skeleton: Optional[str] = None,
        est_latency_ms: float = 50.0,
    ) -> tuple[list[dict[str, Any]], StageMetrics]:
        metrics = StageMetrics(
            stage="retrieve",
            model=self.settings.embedding_model,
            est_latency_ms=est_latency_ms,
            est_cost_usd=0.0,
        )
        t0 = time.perf_counter()
        k = top_k or self.settings.fewshot_top_k
        try:
            self._ensure()
            assert self._collection is not None and self._embedder is not None
            fetch_k = max(k * 5, k)
            emb, tokens = self._embedder.embed(
                [question], task_type="RETRIEVAL_QUERY"
            )
            metrics.input_tokens = tokens
            metrics.est_input_tokens = tokens
            prov = (self.settings.embedding_provider or "").lower()
            if prov in {"openai", "gpt"}:
                metrics.cost_usd = cost_usd(
                    tokens, 0, self.settings.openai_embedding_per_m, 0.0
                )
                metrics.est_cost_usd = metrics.cost_usd
            elif prov in {"gemini", "google", "vertex"}:
                metrics.cost_usd = cost_usd(
                    tokens, 0, self.settings.gemini_embedding_per_m, 0.0
                )
                metrics.est_cost_usd = metrics.cost_usd
            result = self._collection.query(query_embeddings=emb, n_results=fetch_k)
            items: list[dict[str, Any]] = []
            metas = (result.get("metadatas") or [[]])[0]
            docs = (result.get("documents") or [[]])[0]
            dists = (result.get("distances") or [[]])[0]
            for meta, doc, dist in zip(metas, docs, dists):
                items.append(
                    {
                        "question": meta.get("question", ""),
                        "sql": meta.get("sql", ""),
                        "db_id": meta.get("db_id", ""),
                        "hardness": meta.get("hardness", ""),
                        "skeleton": meta.get("skeleton", ""),
                        "distance": float(dist) if dist is not None else 1.0,
                        "document": doc,
                    }
                )

            skel_needles = [
                p.strip().upper()
                for p in (prefer_skeleton or "").split("|")
                if p.strip()
            ]

            def _skel_hit(it: dict[str, Any]) -> int:
                if not skel_needles:
                    return 1
                blob = f" {it.get('sql', '')} {it.get('skeleton', '')} ".upper()
                return 0 if any(n in blob for n in skel_needles) else 1

            def _sort_key(it: dict[str, Any]) -> tuple:
                hard_pen = (
                    0
                    if prefer_hardness and it.get("hardness") == prefer_hardness
                    else (1 if prefer_hardness else 0)
                )
                return (_skel_hit(it), hard_pen, it.get("distance", 1.0))

            items.sort(key=_sort_key)
            items = items[:k]
            metrics.ok = True
            metrics.latency_ms = (time.perf_counter() - t0) * 1000.0
            return items, metrics
        except Exception as exc:  # noqa: BLE001
            metrics.ok = False
            metrics.error = str(exc)[:500]
            metrics.latency_ms = (time.perf_counter() - t0) * 1000.0
            return [], metrics


def format_retrieved_demos(
    items: list[dict[str, Any]],
    *,
    style: str = "easy",
) -> str:
    """Format RAG demos to match generate-prompt shape (RAID-SQL v2)."""
    if not items:
        return ""
    style_l = (style or "easy").lower()
    lines = ["# Retrieved few-shot demonstrations"]
    for it in items:
        q = it.get("question", "")
        sql = (it.get("sql") or "").replace("\n", " ").strip()
        if style_l in {"easy", "e"}:
            lines.append(f'Q: "{q}"')
            lines.append("Schema_links: []")
            lines.append(f"SQL: {sql}")
            lines.append("")
        elif style_l in {"medium", "non-nested", "non_nested"}:
            lines.append(f'Q: "{q}"')
            lines.append("Schema_links: []")
            lines.append(
                "A: Let’s think step by step. Use schema links and Foreign_keys for JOINs."
            )
            lines.append(f"SQL: {sql}")
            lines.append("")
        else:
            lines.append(f'Q: "{q}"')
            lines.append("schema_links: []")
            lines.append(
                'A: Let\'s think step by step. Solve via nested / set-op sub-questions.'
            )
            lines.append(f"SQL: {sql}")
            lines.append("")
    return "\n".join(lines).strip()
