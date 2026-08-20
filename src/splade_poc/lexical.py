from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .corpus import Document


TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class RankedHit:
    document_id: str
    score: float


def _push_unique(values: list[str], seen: set[str], candidate: str) -> None:
    if candidate and candidate not in seen:
        values.append(candidate)
        seen.add(candidate)


def _split_identifier(token: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for part in filter(None, re.split(r"[_-]", token)):
        current = ""
        for index, character in enumerate(part):
            previous_is_lower_or_digit = index > 0 and (
                part[index - 1].islower() or part[index - 1].isdigit()
            )
            if character.isupper() and previous_is_lower_or_digit and current:
                _push_unique(result, seen, current.lower())
                current = ""
            current += character
        _push_unique(result, seen, current.lower())
    return result or [token.lower()]


def tokenize_with_identifiers(text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        if len(token) < 2:
            continue
        _push_unique(terms, seen, token.lower())
        for part in _split_identifier(token):
            _push_unique(terms, seen, part)
    return terms


class Bm25Index:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.document_terms = {
            document.id: tokenize_with_identifiers(document.text) for document in documents
        }
        self.frequencies = {
            document_id: Counter(terms) for document_id, terms in self.document_terms.items()
        }
        self.document_frequency = Counter(
            term for terms in self.document_terms.values() for term in set(terms)
        )
        self.average_length = (
            sum(map(len, self.document_terms.values())) / len(documents) if documents else 0.0
        )

    def search(self, query: str, limit: int = 24) -> list[RankedHit]:
        query_terms = tokenize_with_identifiers(query)
        if not query_terms or not self.documents:
            return []
        hits: list[RankedHit] = []
        for document in self.documents:
            score = 0.0
            tf_by_term = self.frequencies[document.id]
            document_length = len(self.document_terms[document.id])
            for term in query_terms:
                tf = tf_by_term.get(term, 0)
                if not tf:
                    continue
                df = self.document_frequency[term]
                count = len(self.documents)
                inverse_document_frequency = math.log((count - df + 0.5) / (df + 0.5) + 1.0)
                numerator = tf * 2.4
                denominator = tf + 1.4 * (
                    1.0 - 0.75 + 0.75 * (document_length / max(self.average_length, 1.0))
                )
                score += inverse_document_frequency * numerator / max(denominator, 1e-5)
            if query.lower() in document.text.lower():
                score += 0.15
            if score > 0:
                hits.append(RankedHit(document.id, score))
        hits.sort(key=lambda hit: (-hit.score, hit.document_id))
        return hits[:limit]


def rank_bm25(documents: list[Document], query: str, limit: int = 24) -> list[RankedHit]:
    return Bm25Index(documents).search(query, limit)
