"""
LLM adapters and generation backends.

Two layers:

* **LLM adapters** turn a provider SDK into a single ``complete(prompt) -> str``
  call.  :class:`OpenAILLM`, :class:`AnthropicLLM`, and :class:`CallableLLM`
  ship here; anything else with a ``complete`` method (or any plain callable)
  works too.

* **Backends** implement the four LLM-driven steps of Algorithm 1 — chain of
  thought, answer, judge, query refinement.  :class:`PromptedBackend` drives
  them with the paper's prompts through any LLM adapter.
  :class:`HeuristicBackend` implements them with string matching and no model
  at all, so the pipeline can be exercised offline, in tests, and in the demo
  without an API key.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .prompts import DEFAULT_PROMPTS

__all__ = [
    "LLM",
    "OpenAILLM",
    "AnthropicLLM",
    "CallableLLM",
    "coerce_llm",
    "extract_json",
    "Backend",
    "PromptedBackend",
    "HeuristicBackend",
    "Judgement",
]


# ════════════════════════════════════════════════════════════════════════════
#  LLM adapters
# ════════════════════════════════════════════════════════════════════════════

class LLM:
    """Minimal LLM interface: a prompt in, text out."""

    def complete(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, prompt: str) -> str:
        return self.complete(prompt)


class OpenAILLM(LLM):
    """OpenAI chat models via the official ``openai`` SDK.

    >>> llm = OpenAILLM("gpt-4o-mini")            # doctest: +SKIP
    >>> llm.complete("Say hi in JSON.")           # doctest: +SKIP
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **client_kwargs: Any,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "OpenAILLM needs the 'openai' package: pip install 'vendirag[openai]'"
            ) from exc
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, **client_kwargs) if api_key else OpenAI(**client_kwargs)

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class AnthropicLLM(LLM):
    """Claude models via the official ``anthropic`` SDK.

    >>> llm = AnthropicLLM("claude-sonnet-5")     # doctest: +SKIP
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **client_kwargs: Any,
    ):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "AnthropicLLM needs the 'anthropic' package: "
                "pip install 'vendirag[anthropic]'"
            ) from exc
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = (
            anthropic.Anthropic(api_key=api_key, **client_kwargs)
            if api_key else anthropic.Anthropic(**client_kwargs)
        )

    def complete(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class CallableLLM(LLM):
    """Wrap any ``str -> str`` function (a local model, a LangChain chain, a mock)."""

    def __init__(self, fn: Callable[[str], str]):
        self.fn = fn

    def complete(self, prompt: str) -> str:
        return str(self.fn(prompt))


def coerce_llm(llm: Any) -> LLM:
    """Accept an adapter, a bare callable, or a LangChain-style object."""
    if llm is None:
        raise ValueError("an llm is required")
    if isinstance(llm, LLM):
        return llm
    if hasattr(llm, "complete"):
        return CallableLLM(llm.complete)
    if hasattr(llm, "invoke"):  # LangChain runnables
        def _invoke(prompt: str) -> str:
            out = llm.invoke(prompt)
            return getattr(out, "content", out)
        return CallableLLM(_invoke)
    if callable(llm):
        return CallableLLM(llm)
    raise TypeError(f"cannot use {type(llm).__name__} as an LLM")


# ════════════════════════════════════════════════════════════════════════════
#  JSON extraction
# ════════════════════════════════════════════════════════════════════════════

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON object extraction from an LLM response.

    Handles bare JSON, fenced code blocks, and JSON embedded in prose.
    Returns ``{}`` when nothing parses, letting callers fall back rather than
    crash a multi-iteration loop on one malformed response.
    """
    if not text:
        return {}
    text = text.strip()
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _json_candidates(text: str):
    yield text
    for match in _FENCE_RE.findall(text):
        yield match.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        yield text[start:end + 1]


# ════════════════════════════════════════════════════════════════════════════
#  Generation backends
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Judgement:
    """Judge output.  ``quality`` is ``mean(C, R, QA) / 10`` in [0, 1]."""

    coherence: float
    relevance: float
    query_alignment: float

    @property
    def quality(self) -> float:
        return (self.coherence + self.relevance + self.query_alignment) / 30.0


class Backend:
    """The four LLM-driven steps of Algorithm 1."""

    def cot(self, question: str, documents: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def answer(self, question: str, documents: str, reasoning: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def judge(self, question: str, documents: str, answer: str) -> Judgement:  # pragma: no cover
        raise NotImplementedError

    def refine(self, question: str, answer: str, reasoning: str) -> str:  # pragma: no cover
        raise NotImplementedError


class PromptedBackend(Backend):
    """Runs the paper's prompts against an LLM adapter.

    Parameters
    ----------
    llm : LLM or callable
        Used for chain of thought, answer, and query refinement.
    judge_llm : LLM or callable, optional
        Separate judge model.  The paper's target is *relative* quality, which
        is why a smaller or different judge works (Appendix B.3).
    prompts : dict, optional
        Overrides for any of ``cot``, ``answer``, ``judge``, ``refine``.
    on_error : {"fallback", "raise"}
        ``"fallback"`` keeps a long loop alive through a single bad response.
    """

    def __init__(
        self,
        llm: Any,
        judge_llm: Any = None,
        prompts: Optional[Dict[str, str]] = None,
        on_error: str = "fallback",
    ):
        self.llm = coerce_llm(llm)
        self.judge_llm = coerce_llm(judge_llm) if judge_llm is not None else self.llm
        self.prompts = {**DEFAULT_PROMPTS, **(prompts or {})}
        self.on_error = on_error
        #: Every raw prompt/response pair, for debugging and cost accounting.
        self.calls: List[dict] = []

    def _run(self, llm: LLM, name: str, **kwargs) -> Dict[str, Any]:
        prompt = self.prompts[name].format(**kwargs)
        try:
            raw = llm.complete(prompt)
        except Exception:
            if self.on_error == "raise":
                raise
            return {}
        self.calls.append({"step": name, "prompt": prompt, "response": raw})
        return extract_json(raw)

    def cot(self, question: str, documents: str) -> str:
        out = self._run(self.llm, "cot", question=question, documents=documents)
        return str(out.get("reasoning", ""))

    def answer(self, question: str, documents: str, reasoning: str) -> str:
        out = self._run(
            self.llm, "answer",
            question=question, documents=documents, reasoning=reasoning,
        )
        return str(out.get("answer", ""))

    def judge(self, question: str, documents: str, answer: str) -> Judgement:
        out = self._run(
            self.judge_llm, "judge",
            question=question, documents=documents, answer=answer,
        )
        if not out:
            # Neutral score: neither triggers early stopping nor collapses s.
            return Judgement(5.0, 5.0, 5.0)
        return Judgement(
            coherence=_as_score(out.get("coherence")),
            relevance=_as_score(out.get("relevance")),
            query_alignment=_as_score(out.get("query_alignment", out.get("alignment"))),
        )

    def refine(self, question: str, answer: str, reasoning: str) -> str:
        out = self._run(
            self.llm, "refine",
            question=question, answer=answer, reasoning=reasoning,
        )
        return str(out.get("refined_question", "")) or question


def _as_score(value: Any, default: float = 5.0) -> float:
    try:
        return float(min(10.0, max(1.0, float(value))))
    except (TypeError, ValueError):
        return default


class HeuristicBackend(Backend):
    """A deterministic, model-free stand-in for the four LLM steps.

    It does not understand language.  It extracts proper-noun entities, walks
    the bridge chain outward from the entities named in the question, and
    answers with the entity it reaches whose supporting sentence matches the
    question's head noun ("which *instrument*", "in which *town*").  It grades
    an answer by whether that chain closed, not by whether the answer is
    correct.

    That is enough to exercise the *control flow* of Algorithm 1 — the judge
    signal, the EMA on ``s``, early stopping, query refinement — with no API
    key, no network, and byte-identical output on every run, which is what
    makes the demo and the test suite reproducible.  It works because the toy
    corpus is a clean entity graph; on prose it will fail, and it is not a
    substitute for an LLM.
    """

    _STOPWORDS = frozenset(
        "a an the of in on at to for by with and or is are was were which who "
        "whom whose what when where why how did does do that this these those "
        "from as it its their his her they he she be been being also had has "
        "have not no than then there here about into over under between".split()
    )

    #: Head nouns the toy questions ask for, in the order they are looked up.
    TYPE_WORDS = ("instrument", "observatory", "town", "city", "leader",
                  "researcher", "specialist")

    _ABBREV_RE = re.compile(
        r"\b(Dr|Mr|Mrs|Ms|Prof|St|Mt|Jr|Sr|vs|etc|e\.g|i\.e)\.", re.IGNORECASE
    )
    # Spans of capitalized words.  A sentence-ending period is never part of a
    # span; only the shielded dots of abbreviations (\x00) are.
    _ENTITY_RE = re.compile("(?:[A-Z][\\w'\\x00-]*)(?:\\s+[A-Z][\\w'\\x00-]*)*")
    #: Capitalized words that start sentences rather than name things.
    _NOT_ENTITIES = frozenset(
        "the a an it he she they records according contemporary archival "
        "official reports funding weather photographs documentary "
        "which what who where when whose why how at in on of and or is was "
        "step steps from".split()
    )

    def __init__(self, quality_floor: float = 0.3):
        self.quality_floor = quality_floor

    # -- text utilities -------------------------------------------------------

    @classmethod
    def keywords(cls, text: str) -> set:
        tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
        return {t for t in tokens if t not in cls._STOPWORDS and len(t) > 2}

    @classmethod
    def sentences(cls, text: str) -> List[str]:
        # Shield common abbreviations so "Dr. Kovac" is not two sentences.
        shielded = cls._ABBREV_RE.sub(
            lambda m: m.group(0)[:-1] + "\x00", text.replace("\n", " ")
        )
        parts = re.split(r"(?<=[.!?])\s+", shielded)
        return [p.replace("\x00", ".").strip() for p in parts if p.strip()]

    @classmethod
    def entities(cls, text: str) -> set:
        """Proper-noun spans, with sentence-initial filler words trimmed off."""
        found = set()
        # Shield abbreviation dots first so a span cannot run across a real
        # sentence boundary ("...Institute. Dr. Kovac" is two entities).
        shielded = cls._ABBREV_RE.sub(
            lambda m: m.group(0)[:-1] + "\x00", text
        )
        for span in cls._ENTITY_RE.findall(shielded):
            span = span.replace("\x00", ".")
            words = span.strip().strip(".,;:").split()
            while words and words[0].lower().rstrip(".") in cls._NOT_ENTITIES:
                words.pop(0)
            if not words:
                continue
            name = " ".join(words).strip(".,;:")
            if len(name) > 2:
                found.add(name)
        return found

    @classmethod
    def head_type(cls, question: str) -> Optional[str]:
        """The noun the question asks for — the earliest type word it contains."""
        lowered = question.lower()
        hits = [(lowered.find(w), w) for w in cls.TYPE_WORDS if w in lowered]
        return min(hits)[1] if hits else None

    def _walk(self, question: str, documents: str):
        """Breadth-first walk of the entity graph out from the question.

        Returns ``(discoveries, steps)`` where ``discoveries`` maps each newly
        reached entity to ``(depth, supporting sentence)``.
        """
        sentences = self.sentences(documents)
        sent_entities = [(s, self.entities(s)) for s in sentences]
        seed = self.entities(question)
        known = set(seed)
        frontier = set(seed)
        discoveries: Dict[str, tuple] = {}
        steps: List[str] = []
        depth = 1

        while frontier and depth <= 6:
            found: Dict[str, tuple] = {}
            for sentence, ents in sent_entities:
                if not (ents & frontier):
                    continue
                for entity in ents - known:
                    found.setdefault(entity, (depth, sentence))
            if not found:
                break
            for entity, (d, sentence) in found.items():
                discoveries[entity] = (d, sentence)
            # Quote the supporting sentences verbatim: the chain-of-thought is
            # what carries evidence forward between iterations, so it has to
            # preserve the appositive glosses the answer step matches on.
            for sentence in dict.fromkeys(sent for _, sent in found.values()):
                steps.append(f"Step {len(steps) + 1}: {sentence}")
            known |= set(found)
            frontier = set(found)
            depth += 1

        return discoveries, steps

    def _resolve(self, question: str, documents: str):
        """Pick the answer entity: deepest match for the question's head noun."""
        discoveries, steps = self._walk(question, documents)
        if not discoveries:
            return None, steps
        head = self.head_type(question)
        if not head:
            return None, steps
        # An entity whose own name carries the head noun ("Rowan Observatory")
        # is a direct match; otherwise fall back to the appositive gloss in the
        # sentence that introduced it ("..., the instrument that the team...").
        # Shallowest wins in both cases: a later sentence naming the head noun
        # is describing an entity the walk has already passed.
        by_name = [(d, e) for e, (d, _) in discoveries.items() if head in e.lower()]
        if by_name:
            return min(by_name)[1], steps
        by_gloss = [(d, e) for e, (d, sent) in discoveries.items() if head in sent.lower()]
        if by_gloss:
            return min(by_gloss)[1], steps
        return None, steps

    # -- Backend interface ----------------------------------------------------

    def cot(self, question: str, documents: str) -> str:
        _, steps = self._walk(question, documents)
        if not steps:
            return "No document in the retrieved evidence mentions the entities in the question."
        return " ".join(steps)

    def answer(self, question: str, documents: str, reasoning: str) -> str:
        # Evidence from earlier iterations survives only in the accumulated
        # reasoning, so the answer is resolved over both.
        entity, _ = self._resolve(question, f"{reasoning}\n{documents}")
        if entity:
            return entity
        return "Insufficient evidence in the retrieved documents."

    def judge(self, question: str, documents: str, answer: str) -> Judgement:
        if not answer or answer.startswith("Insufficient"):
            return Judgement(2.0, 1.0, 1.0)
        head = self.head_type(question)
        sentence = next(
            (s for s in self.sentences(documents) if answer in s), ""
        )
        q_words = self.keywords(question)
        evidence = (
            len(q_words & self.keywords(documents)) / len(q_words) if q_words else 0.0
        )
        chain_closed = bool(sentence)
        aligned = bool(head and sentence and (head in sentence.lower() or head in answer.lower()))

        def to_ten(x: float) -> float:
            return round(1.0 + 9.0 * (self.quality_floor + (1 - self.quality_floor) * x), 1)

        return Judgement(
            coherence=to_ten(1.0 if chain_closed else 0.2),
            relevance=to_ten(evidence),
            query_alignment=to_ten(1.0 if aligned else 0.2),
        )

    def refine(self, question: str, answer: str, reasoning: str) -> str:
        """Chase the next hop by naming the bridge entities just reached.

        This is the crude version of what a real refiner does: it drops the
        original question's entity — already covered — and asks about the
        frontier instead, so the next retrieval matches on the bridge rather
        than on the thicket of documents about the question's head entity.
        """
        discoveries, _ = self._walk(question, reasoning)
        if not discoveries:
            return question
        # Only the deepest entities are the live frontier; anything shallower
        # has already been retrieved for.
        deepest = max(d for d, _ in discoveries.values())
        frontier = sorted(e for e, (d, _) in discoveries.items() if d == deepest)
        return f"What is known about {', '.join(frontier[:2])}?"
