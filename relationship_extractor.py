"""
Relationship Extractor using OpenAI API.
Improvements over v1:
- LLM-based pronoun resolution (more accurate than spaCy en_core_web_sm)
- Structured JSON output via response_format=json_object (no regex fallback)
- Few-shot examples in extraction prompts
- Tiktoken-based token budgeting for Pass 2 (replaces arbitrary [:40] cap)
- Case-insensitive deduplication
"""

import json
import re
import tiktoken
import pandas as pd
from typing import List, Dict
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# System prompts — define the JSON schema once; reused across all calls
# ---------------------------------------------------------------------------
_SYSTEM_EXTRACT = """\
You are a precise relationship extraction engine. Always return valid JSON in exactly this schema:
{"relationships": [{"subject": "string", "predicate": "string", "object": "string", "negated": boolean, "source_quote": "string"}]}
Rules:
- subject and object must never be empty strings.
- source_quote must be a verbatim excerpt from the provided text.
- negated is true only when the relationship is explicitly denied in the text.
- Return {"relationships": []} if no relationships are found.
"""

_SYSTEM_CROSS = """\
You are a relationship reasoning engine. Always return valid JSON in exactly this schema:
{"relationships": [{"subject": "string", "predicate": "string", "object": "string", "negated": boolean, "source_quote": "string"}]}
Rules:
- Only emit NEW relationships derivable by combining two or more of the provided relationships.
- Do NOT re-emit any input relationship.
- Do NOT use external knowledge.
- source_quote must be "derived from existing relationships".
- Return {"relationships": []} if no new relationships can be derived.
"""

_SYSTEM_IMPLICIT = """\
You are an implicit relationship extraction engine. Always return valid JSON in exactly this schema:
{"relationships": [{"subject": "string", "predicate": "string", "object": "string", "negated": boolean, "source_quote": "string"}]}
Rules:
- Only extract relationships that are strongly implied but not literally stated.
- source_quote must be the verbatim text fragment that implies the relationship.
- Do NOT use external knowledge.
- Return {"relationships": []} if none found.
"""


class RelationshipExtractor:
    """Extract relationships from text using OpenAI with a multi-pass approach."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._enc = tiktoken.get_encoding("cl100k_base")
        self._chunk_size = 1200
        self._chunk_overlap = 150

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks on natural boundaries."""
        separators = ["\n\n", "\n", ". ", " ", ""]
        size, overlap = self._chunk_size, self._chunk_overlap

        def _split(t: str, seps: list[str]) -> list[str]:
            if not seps or len(t) <= size:
                return [t] if t.strip() else []
            sep = seps[0]
            parts = re.split(re.escape(sep), t) if sep else list(t)
            chunks, current = [], ""
            for part in parts:
                candidate = current + (sep if current else "") + part
                if len(candidate) > size and current:
                    chunks.extend(_split(current, seps[1:]))
                    current = part
                else:
                    current = candidate
            if current.strip():
                chunks.extend(_split(current, seps[1:]))
            return chunks

        raw = _split(text, separators)
        if not raw:
            return []

        # Apply overlap: prepend tail of previous chunk to each chunk
        result = [raw[0]]
        for chunk in raw[1:]:
            tail = result[-1][-overlap:] if len(result[-1]) > overlap else result[-1]
            result.append(tail + chunk)
        return result

    # ------------------------------------------------------------------
    # Low-level API helpers
    # ------------------------------------------------------------------

    def _chat(self, prompt: str, max_tokens: int = 1024) -> str:
        """Free-text completion (used for pronoun resolution)."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def _chat_json(self, system: str, user: str, max_tokens: int = 2048) -> List[Dict]:
        """Structured JSON completion; returns the relationships list."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        try:
            data = json.loads(raw)
            rels = data.get("relationships", data.get("results", []))
            return rels if isinstance(rels, list) else []
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Pronoun resolution (LLM-based — more accurate than spaCy sm model)
    # ------------------------------------------------------------------

    def _resolve_pronouns_llm(self, text: str) -> str:
        """Rewrite text replacing all pronouns with explicit antecedents."""
        prompt = (
            "Rewrite the following text replacing every pronoun with its explicit antecedent. "
            "Preserve all other wording exactly. Return ONLY the rewritten text, no commentary.\n\n"
            f"{text}"
        )
        return self._chat(prompt, max_tokens=max(1024, len(text) // 3))

    # ------------------------------------------------------------------
    # Token-budget-aware slicing for Pass 2
    # ------------------------------------------------------------------

    def _token_budget_slice(self, items: List[Dict], max_tokens: int = 3000) -> List[Dict]:
        """Return as many items as fit within the token budget."""
        total = 0
        result = []
        for item in items:
            line = f"- {item.get('subject', '')} {item.get('predicate', '')} {item.get('object', '')}\n"
            n = len(self._enc.encode(line))
            if total + n > max_tokens:
                break
            total += n
            result.append(item)
        return result

    # ------------------------------------------------------------------
    # Per-pass extraction
    # ------------------------------------------------------------------

    def _extract_from_chunk(self, payload: str, pass_number: int = 1) -> List[Dict]:
        """
        - Pass 1: LLM pronoun resolution then explicit extraction.
        - Pass 2: Cross-chunk reasoning from a relationship summary.
        - Pass 3: Implicit-only extraction from raw text.
        """
        if pass_number == 1:
            payload = self._resolve_pronouns_llm(payload)
            user = self._build_local_extraction_prompt(payload)
            return self._chat_json(_SYSTEM_EXTRACT, user)
        elif pass_number == 2:
            user = self._build_cross_chunk_prompt(payload)
            return self._chat_json(_SYSTEM_CROSS, user)
        else:
            user = self._build_implicit_extraction_prompt(payload)
            return self._chat_json(_SYSTEM_IMPLICIT, user)

    # ------------------------------------------------------------------
    # Prompt builders (few-shot examples included)
    # ------------------------------------------------------------------

    def _build_local_extraction_prompt(self, chunk: str) -> str:
        return f"""Extract ALL explicitly stated relationships from the text below.

RULES:
1. Only extract relationships EXPLICITLY stated — no inferences or external knowledge.
2. Set "negated": true when the relationship is explicitly denied.
3. "source_quote" must be a verbatim excerpt from the passage.
4. Subject and object must be non-empty.

EXAMPLES:

Text: "Tesla was founded by Elon Musk in 2003. He did not invent the electric car."
{{"relationships": [
  {{"subject": "Tesla", "predicate": "was founded by", "object": "Elon Musk", "negated": false, "source_quote": "Tesla was founded by Elon Musk in 2003"}},
  {{"subject": "Tesla", "predicate": "was founded in", "object": "2003", "negated": false, "source_quote": "Tesla was founded by Elon Musk in 2003"}},
  {{"subject": "Elon Musk", "predicate": "invented", "object": "electric car", "negated": true, "source_quote": "He did not invent the electric car"}}
]}}

Text: "The treaty was signed by France and Germany. It did not cover military matters."
{{"relationships": [
  {{"subject": "France", "predicate": "signed", "object": "the treaty", "negated": false, "source_quote": "The treaty was signed by France and Germany"}},
  {{"subject": "Germany", "predicate": "signed", "object": "the treaty", "negated": false, "source_quote": "The treaty was signed by France and Germany"}},
  {{"subject": "the treaty", "predicate": "covers", "object": "military matters", "negated": true, "source_quote": "It did not cover military matters"}}
]}}

NOW EXTRACT from this text:
{chunk}"""

    def _build_cross_chunk_prompt(self, relationships_str: str) -> str:
        return f"""Below are relationships already extracted from different parts of a document.
Identify NEW relationships supported ONLY by combining two or more of these — no external knowledge.
Do NOT re-emit any existing relationship.

EXAMPLE:
Input:
- John is the CEO of Acme Corp
- Acme Corp is headquartered in Berlin
New: {{"relationships": [{{"subject": "John", "predicate": "is based in", "object": "Berlin", "negated": false, "source_quote": "derived from existing relationships"}}]}}

NOW PROCESS:
{relationships_str}"""

    def _build_implicit_extraction_prompt(self, chunk: str) -> str:
        return f"""From this text, extract ONLY relationships that are strongly implied but not literally stated.
Use only evidence from the text itself — no external knowledge.

Text:
{chunk}"""

    def extract(self, text: str, include_implicit: bool = False) -> pd.DataFrame:
        """
        Multi-pass extraction:
        - Pass 1: LLM pronoun resolution + explicit extraction per chunk
        - Pass 2: Cross-chunk reasoning (tiktoken-budgeted, not hard-capped)
        - Pass 3 (optional): Implicit relationship extraction
        """
        print("Chunking text...")
        chunks = self.chunk_text(text)
        print(f"Split into {len(chunks)} chunks")

        # PASS 1
        print("\nPass 1: Extracting explicit relationships from chunks...")
        all_relationships: List[Dict] = []
        for chunk in tqdm(chunks, desc="Processing chunks"):
            rels = self._extract_from_chunk(chunk, pass_number=1)
            all_relationships.extend(rels)
        print(f"Extracted {len(all_relationships)} relationships from chunks")

        explicit_relationships = [r for r in all_relationships if not r.get("negated", False)]
        print(f"After filtering negations: {len(explicit_relationships)} non-negated relationships")

        # PASS 2 — seed only with non-negated; result added to the full set
        print("\nPass 2: Cross-chunk reasoning...")
        budgeted = self._token_budget_slice(explicit_relationships, max_tokens=3000)
        lines = [
            f"- {r.get('subject', 'Unknown')} {r.get('predicate', 'related to')} {r.get('object', 'Unknown')}"
            for r in budgeted
        ]
        relationships_summary = "\n".join(lines)
        if relationships_summary.strip():
            cross_chunk_new = self._extract_from_chunk(relationships_summary, pass_number=2)
            print(f"Found {len(cross_chunk_new)} cross-chunk relationships")
            explicit_relationships.extend(cross_chunk_new)

        # PASS 3
        if include_implicit:
            print("\nPass 3: Extracting implicit relationships...")
            window = 5000
            implicit_found: List[Dict] = []
            for start in range(0, len(text), window):
                snippet = text[start: start + window]
                implicit_found.extend(self._extract_from_chunk(snippet, pass_number=3))
                if len(implicit_found) >= 100:
                    break
            print(f"Found {len(implicit_found)} implicit relationships")
            explicit_relationships.extend(implicit_found)

        # Deduplicate (case-insensitive key to catch surface-form variants)
        # Include negated from Pass 1 in the final output; they were only excluded from seeding Pass 2
        negated_from_pass1 = [r for r in all_relationships if r.get("negated", False)]
        final_set = explicit_relationships + negated_from_pass1

        print("\nDeduplicating relationships...")
        unique: Dict[tuple, dict] = {}
        for rel in final_set:
            s = (rel.get("subject") or "").strip()
            p = (rel.get("predicate") or "").strip()
            o = (rel.get("object") or "").strip()
            if s and p and o:
                key = (s.lower(), p.lower(), o.lower())
                if key not in unique:
                    unique[key] = {
                        "subject": s,
                        "predicate": p,
                        "object": o,
                        "negated": bool(rel.get("negated", False)),
                        "source_quote": (rel.get("source_quote") or "").strip(),
                    }
        print(f"After deduplication: {len(unique)} unique relationships")

        if not unique:
            return pd.DataFrame(columns=["subject", "predicate", "object", "negated", "source_quote"])

        df = pd.DataFrame(list(unique.values()))
        # Ensure column order
        expected = ["subject", "predicate", "object", "negated", "source_quote"]
        for col in expected:
            if col not in df.columns:
                df[col] = None
        return df[expected]

    def extract_batch(self, texts: List[str], include_implicit: bool = False) -> dict:
        results = {}
        for i, t in enumerate(texts):
            print(f"\n{'='*60}\nProcessing text {i+1}/{len(texts)}\n{'='*60}")
            results[f"text_{i+1}"] = self.extract(t, include_implicit)
        return results