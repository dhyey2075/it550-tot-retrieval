"""
LLM wrapper with JSONL caching and --dry_run (plan_v2/01 §6.4).

Contract:
- llm_call(task, query, prompt_version) -> dict
- Cache key = hash(provider, model_id, prompt_version, prompt_text_hash, temperature, query_text)
- Write to cache after every successful call
- Temperature 0 by default
- Never silently drop a query; log failures
- --dry_run counts uncached queries and estimates call count

Live API calls are only made when a provider is configured and dry_run is False.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

CACHE_TASK_FILES = {
    "decomposition": "decompositions.jsonl",
    "reformulation": "reformulations.jsonl",
}


@dataclass
class LLMConfig:
    provider: str = "stub"  # stub | openai | gemini
    model: str = "stub-model"
    temperature: float = 0.0
    max_output_tokens: int | None = 1024
    max_retries: int = 5
    backoff_base_sec: float = 1.0


@dataclass
class LLMResult:
    key: str
    qid: str
    task: str
    model: str
    provider: str
    prompt_version: str
    prompt_text_hash: str
    temperature: float
    query_text: str
    raw_output: str
    parsed: Any
    timestamp: str
    cached: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_cache_row(self) -> dict[str, Any]:
        row = asdict(self)
        return row


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_prompt(template: str, query: str) -> str:
    """Substitute only the `{query}` placeholder. Avoids str.format so JSON braces in prompts are safe."""
    if "{query}" not in template:
        raise ValueError("prompt template must contain the literal placeholder {query}")
    return template.replace("{query}", query)


def make_cache_key(
    provider: str,
    model_id: str,
    prompt_version: str,
    prompt_text: str,
    temperature: float,
    query_text: str,
) -> str:
    prompt_text_hash = _sha256(prompt_text)
    material = "|".join(
        [
            provider,
            model_id,
            prompt_version,
            prompt_text_hash,
            f"{temperature:.6f}",
            query_text,
        ]
    )
    return _sha256(material)


class LLMCache:
    """Append-only JSONL cache; loads into memory for lookups."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self._by_key: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping corrupt cache line %s:%s", self.path, line_no)
                    continue
                key = row.get("key")
                if key:
                    self._by_key[key] = row

    def get(self, key: str) -> dict[str, Any] | None:
        return self._by_key.get(key)

    def put(self, row: dict[str, Any]) -> None:
        key = row["key"]
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._by_key[key] = row

    def __len__(self) -> int:
        return len(self._by_key)


def _extract_stub_facets(text: str) -> dict[str, str]:
    import re
    q_clean = text.replace("\n", " ").strip()
    sentences = [s.strip() for s in re.split(r"[.!?]+", q_clean) if len(s.strip()) > 5]
    cat = "N/A"
    m_cat = re.search(r"\b(horror movie|comedy|action|drama|fantasy movie|film|movie|documentary|sci-fi|thriller|western)\b", text, re.I)
    if m_cat:
        cat = m_cat.group(0).lower()
    time_p = "N/A"
    m_time = re.search(r"\b(early 2000s|late 90s|early 90s|late 80s|early 80s|19\d\d|20\d\d|[56789]0\s*s)\b", text, re.I)
    if m_time:
        time_p = m_time.group(0)
    place = "N/A"
    m_place = re.search(r"\b(apartment|house|island|city|desert|room|hospital|school|beach|farmhouse|lighthouse|junkyard|hotel)\b", text, re.I)
    if m_place:
        place = m_place.group(0)
    people = "N/A"
    m_people = re.findall(r"\b(woman|man|guy|girl|kid|boy|father|mother|couple|dads|police|gangster|trucker|aliens|robot|strangers)\b", text, re.I)
    if m_people:
        people = ", ".join(sorted(set(p.lower() for p in m_people[:4])))
    desc = sentences[0] if sentences else text[:100]
    distinctive = sentences[1] if len(sentences) > 1 else "N/A"
    return {
        "category": cat,
        "description": desc,
        "people": people,
        "place_setting": place,
        "time_period": time_p,
        "distinctive_details": distinctive,
        "name_fragments": "N/A",
    }


def _stub_complete(prompt_text: str, query_text: str, task: str) -> str:
    """Deterministic stub for offline / dry plumbing tests. Never invents answer titles."""
    import re
    effective_query = query_text.strip()
    if not effective_query:
        m = re.findall(r"Description:\s*([^\n]+(?:\n[^\n]+)*)", prompt_text)
        if m:
            effective_query = m[-1].strip()
        else:
            effective_query = prompt_text[:120].replace("\n", " ").strip()

    if task == "decomposition":
        # Facet schema (plan 05). Excerpt-only; never invent answer titles.
        return json.dumps(_extract_stub_facets(effective_query))
    if task == "reformulation":
        return f"[stub reformulation] {effective_query[:200]}"
    return f"[stub {task}] {effective_query[:200]}"


def _openai_complete(cfg: LLMConfig, prompt_text: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": cfg.temperature,
    }
    if cfg.max_output_tokens is not None:
        kwargs["max_tokens"] = cfg.max_output_tokens
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _gemini_complete(cfg: LLMConfig, prompt_text: str) -> str:
    import google.generativeai as genai

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(cfg.model)
    generation_config = {"temperature": cfg.temperature}
    if cfg.max_output_tokens is not None:
        generation_config["max_output_tokens"] = cfg.max_output_tokens
    resp = model.generate_content(prompt_text, generation_config=generation_config)
    return getattr(resp, "text", None) or str(resp)


PROVIDERS: dict[str, Callable[[LLMConfig, str], str]] = {
    "stub": lambda cfg, prompt: _stub_complete(prompt, "", "generic"),
    "openai": _openai_complete,
    "gemini": _gemini_complete,
}


def default_parse(task: str, raw: str) -> Any:
    raw = raw.strip()
    if task == "decomposition":
        try:
            from tot.decomposition import parse_facets

            return parse_facets(raw, predictive=False)
        except Exception:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"parse_error": True, "raw": raw}
    return raw


class LLMClient:
    def __init__(
        self,
        config: LLMConfig | None = None,
        cache_dir: str | Path = "cache",
        parse_fn: Callable[[str, str], Any] | None = None,
    ):
        self.config = config or LLMConfig()
        self.cache_dir = Path(cache_dir)
        self.parse_fn = parse_fn or default_parse
        self._caches: dict[str, LLMCache] = {}

    def _cache_for(self, task: str) -> LLMCache:
        filename = CACHE_TASK_FILES.get(task, f"{task}.jsonl")
        if task not in self._caches:
            self._caches[task] = LLMCache(self.cache_dir / filename)
        return self._caches[task]

    def _complete_with_retries(self, prompt_text: str, task: str, query: str = "") -> str:
        provider = self.config.provider.lower()
        if provider == "stub":
            return _stub_complete(prompt_text, query, task)

        if provider not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider '{provider}'. Use stub|openai|gemini.")

        complete = PROVIDERS[provider]
        last_err: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                if provider == "stub":
                    return _stub_complete(prompt_text, query, task)
                return complete(self.config, prompt_text)
            except Exception as exc:  # noqa: BLE001 — retry then surface
                last_err = exc
                sleep_for = self.config.backoff_base_sec * (2**attempt)
                logger.warning(
                    "LLM call failed (attempt %s/%s): %s; sleeping %.1fs",
                    attempt + 1,
                    self.config.max_retries,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)
        assert last_err is not None
        raise last_err

    def llm_call(
        self,
        task: str,
        query: str,
        prompt_version: str,
        *,
        qid: str = "",
        prompt_text: str | None = None,
        prompt_template: str | None = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> LLMResult | dict[str, Any]:
        """
        Call the LLM (or return cache hit). If dry_run=True, do not call the API;
        return a status dict describing whether the query would be a cache hit.
        """
        if prompt_text is None:
            if prompt_template is None:
                raise ValueError("Provide prompt_text or prompt_template")
            prompt_text = render_prompt(prompt_template, query)

        key = make_cache_key(
            self.config.provider,
            self.config.model,
            prompt_version,
            prompt_text,
            float(self.config.temperature),
            query,
        )
        prompt_text_hash = _sha256(prompt_text)
        cache = self._cache_for(task)

        if dry_run:
            hit = cache.get(key) is not None and not force
            return {
                "qid": qid,
                "task": task,
                "key": key,
                "cached": hit,
                "would_call": not hit,
            }

        if not force:
            cached = cache.get(key)
            if cached is not None:
                result = LLMResult(
                    key=cached["key"],
                    qid=cached.get("qid", qid),
                    task=cached.get("task", task),
                    model=cached.get("model", self.config.model),
                    provider=cached.get("provider", self.config.provider),
                    prompt_version=cached.get("prompt_version", prompt_version),
                    prompt_text_hash=cached.get("prompt_text_hash", prompt_text_hash),
                    temperature=float(cached.get("temperature", self.config.temperature)),
                    query_text=cached.get("query_text", query),
                    raw_output=cached.get("raw_output", ""),
                    parsed=cached.get("parsed"),
                    timestamp=cached.get("timestamp", ""),
                    cached=True,
                    meta=cached.get("meta", {}),
                )
                return result

        raw = self._complete_with_retries(prompt_text, task, query)
        parsed = self.parse_fn(task, raw)
        result = LLMResult(
            key=key,
            qid=qid,
            task=task,
            model=self.config.model,
            provider=self.config.provider,
            prompt_version=prompt_version,
            prompt_text_hash=prompt_text_hash,
            temperature=float(self.config.temperature),
            query_text=query,
            raw_output=raw,
            parsed=parsed,
            timestamp=datetime.now(timezone.utc).isoformat(),
            cached=False,
        )
        cache.put(result.to_cache_row())
        return result


def dry_run_estimate(
    client: LLMClient,
    task: str,
    items: list[dict[str, str]],
    prompt_version: str,
    prompt_template: str,
) -> dict[str, Any]:
    """
    items: list of {qid, query}
    Returns counts of cached vs uncached and estimated API calls.
    """
    cached = 0
    uncached = 0
    details = []
    for item in items:
        status = client.llm_call(
            task=task,
            query=item["query"],
            prompt_version=prompt_version,
            qid=item.get("qid", ""),
            prompt_template=prompt_template,
            dry_run=True,
        )
        assert isinstance(status, dict)
        details.append(status)
        if status["cached"]:
            cached += 1
        else:
            uncached += 1
    return {
        "task": task,
        "n_queries": len(items),
        "cached": cached,
        "uncached": uncached,
        "estimated_api_calls": uncached,
        "details": details,
    }


def _load_queries_jsonl(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    """Best-effort loader; field names confirmed in plan 02. Tries common id/text keys."""
    items: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # TREC-ToT 2025 uses query_id + query (confirmed in data_report.md).
            qid = str(obj.get("query_id") or obj.get("id") or obj.get("qid") or "")
            query = obj.get("query") or obj.get("text") or obj.get("request") or ""
            if not isinstance(query, str):
                query = str(query)
            items.append({"qid": qid, "query": query})
            if limit is not None and len(items) >= limit:
                break
    return items


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM wrapper with cache / dry-run")
    p.add_argument("--task", required=True, choices=sorted(set(CACHE_TASK_FILES) | {"custom"}))
    p.add_argument("--prompt-version", required=True)
    p.add_argument("--prompt-file", type=Path, help="Prompt template file; use {query} placeholder")
    p.add_argument("--queries", type=Path, help="JSONL queries file")
    p.add_argument("--qid", default="", help="Single-query mode qid")
    p.add_argument("--query", default="", help="Single-query mode text")
    p.add_argument("--cache-dir", type=Path, default=Path("cache"))
    p.add_argument("--provider", default=os.environ.get("TOT_LLM_PROVIDER", "stub"))
    p.add_argument("--model", default=os.environ.get("TOT_LLM_MODEL", "stub-model"))
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true", help="Ignore cache and re-call")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_argparser().parse_args(argv)

    if not args.prompt_file and not (args.query or args.queries):
        raise SystemExit("Need --prompt-file with --queries or --query")

    prompt_template = None
    if args.prompt_file:
        prompt_template = args.prompt_file.read_text(encoding="utf-8")
    else:
        prompt_template = "Description: {query}\n"

    cfg = LLMConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )
    client = LLMClient(config=cfg, cache_dir=args.cache_dir)

    if args.queries:
        items = _load_queries_jsonl(args.queries, limit=args.limit)
    else:
        items = [{"qid": args.qid, "query": args.query}]

    if args.dry_run:
        summary = dry_run_estimate(
            client,
            task=args.task if args.task != "custom" else "custom",
            items=items,
            prompt_version=args.prompt_version,
            prompt_template=prompt_template,
        )
        print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2))
        print(f"estimated_api_calls={summary['estimated_api_calls']}")
        return 0

    failures = 0
    for item in items:
        try:
            result = client.llm_call(
                task=args.task if args.task != "custom" else "custom",
                query=item["query"],
                prompt_version=args.prompt_version,
                qid=item["qid"],
                prompt_template=prompt_template,
                force=args.force,
            )
            assert isinstance(result, LLMResult)
            print(
                json.dumps(
                    {
                        "qid": result.qid,
                        "cached": result.cached,
                        "key": result.key[:16] + "...",
                        "raw_preview": result.raw_output[:120],
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            logger.error("Failed qid=%s: %s", item.get("qid"), exc)

    if failures:
        logger.error("%s queries failed", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
