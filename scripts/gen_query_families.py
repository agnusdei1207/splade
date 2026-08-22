"""Generate several query families so one synthetic style cannot decide the outcome.

Heading-derived queries are keyword-shaped and share vocabulary with their source
document, which flatters a lexical scorer. These families vary along the two axes
that actually matter here: how much wording the query shares with the document,
and which language the user typed in.

  heading    section titles, keyword shaped, held out of the body
  title      document titles, held out of the body
  sentence   a body sentence (inverse cloze), held out of the body
  identifier backticked code spans and paths, exact-match shaped
  question   a body sentence rewritten into an interrogative shell

Every family holds its query text out of the indexed document, so no family can be
answered by substring matching alone.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_support import (  # noqa: E402
    HEADING_RE,
    clean_heading,
    markdown_files,
    parse_frontmatter,
    usable_heading,
)

HANGUL_RE = re.compile(r"[가-힣]")
CODE_SPAN_RE = re.compile(r"`([^`\n]{3,60})`")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+|\n")
NOISE_PREFIX_RE = re.compile(r"^[\s\d.)\-–—•*#]+")

KOREAN_QUESTION_SHELLS = ["{} 어떻게 하나요?", "{} 무엇인가요?", "{} 알려줘"]
ENGLISH_QUESTION_SHELLS = ["how do I {}?", "what is {}?", "explain {}"]


def language_of(text: str) -> str:
    hangul = len(HANGUL_RE.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if hangul + latin == 0:
        return "other"
    return "ko" if hangul / (hangul + latin) > 0.35 else "en"


def clean_line(text: str) -> str:
    text = NOISE_PREFIX_RE.sub("", text).strip()
    text = re.sub(r"[`*_\[\]]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def content_words(text: str) -> int:
    return len([w for w in text.split() if len(w) >= 2])


def usable_sentence(text: str) -> bool:
    if not 5 <= content_words(text) <= 30:
        return False
    if text.startswith(("|", ">", "```", "http")):
        return False
    return not re.match(r"^[-=]{3,}$", text)


def usable_identifier(text: str) -> bool:
    text = text.strip()
    if not 3 <= len(text) <= 60 or " " in text and len(text.split()) > 3:
        return False
    return bool(re.search(r"[A-Za-z]", text)) and not HANGUL_RE.search(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=Path("/corpus"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-family-per-document", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    documents: list[dict] = []
    families: dict[str, dict[str, list[str]]] = {
        name: {} for name in ("heading", "title", "sentence", "identifier", "question")
    }

    for path in markdown_files(args.corpus_root):
        raw = path.read_text(encoding="utf-8", errors="replace")
        meta, body = parse_frontmatter(raw)
        document_id = path.relative_to(args.corpus_root).as_posix()

        headings: list[str] = []
        kept: list[str] = []
        for line in body.splitlines():
            match = HEADING_RE.match(line)
            if match:
                heading = clean_heading(match.group(2))
                if usable_heading(heading):
                    headings.append(heading)
                    continue
            kept.append(line)

        title_match = re.search(r"^#[ \t]+(.+?)[ \t]*$", body, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else ""
        stripped = "\n".join(line for line in kept if not line.startswith("# "))

        identifiers = [
            span for span in CODE_SPAN_RE.findall(stripped) if usable_identifier(span)
        ]
        sentences = [
            cleaned
            for raw_sentence in SENTENCE_SPLIT_RE.split(stripped)
            if usable_sentence(cleaned := clean_line(raw_sentence))
        ]

        picked_sentences = rng.sample(
            sentences, min(args.per_family_per_document, len(sentences))
        )
        picked_questions = rng.sample(
            sentences, min(2, len(sentences))
        )
        picked_identifiers = rng.sample(
            list(dict.fromkeys(identifiers)),
            min(args.per_family_per_document, len(set(identifiers))),
        )

        held_out = set(headings) | set(picked_sentences) | set(picked_questions)
        if title:
            held_out.add(title)
        body_lines = [line for line in stripped.splitlines() if clean_line(line) not in held_out]

        documents.append(
            {
                "id": document_id,
                "title": title or path.stem,
                "text": "{}\n{}\n{}\n{}".format(
                    "",  # title held out for the title family
                    " ".join(meta.get("aliases", [])),
                    " ".join(meta.get("tags", [])),
                    "\n".join(body_lines),
                ),
                "bytes": len(raw.encode("utf-8")),
                "language": language_of(raw),
            }
        )

        for heading in set(headings):
            families["heading"].setdefault(heading, []).append(document_id)
        if title and content_words(title) >= 2:
            families["title"].setdefault(title, []).append(document_id)
        for sentence in picked_sentences:
            families["sentence"].setdefault(sentence, []).append(document_id)
        for identifier in picked_identifiers:
            families["identifier"].setdefault(identifier, []).append(document_id)
        for sentence in picked_questions:
            shells = (
                KOREAN_QUESTION_SHELLS
                if language_of(sentence) == "ko"
                else ENGLISH_QUESTION_SHELLS
            )
            core = sentence.rstrip(".!?。")
            families["question"].setdefault(rng.choice(shells).format(core), []).append(document_id)

    queries: list[dict] = []
    for family, mapping in families.items():
        for index, (text, document_ids) in enumerate(sorted(mapping.items())):
            queries.append(
                {
                    "id": f"{family[:3]}{index:05d}",
                    "family": family,
                    "query": text,
                    "language": language_of(text),
                    "relevance": sorted(set(document_ids)),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "documents.json").write_text(
        json.dumps(documents, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "queries.json").write_text(
        json.dumps(queries, ensure_ascii=False), encoding="utf-8"
    )

    breakdown: dict[str, dict[str, int]] = {}
    for query in queries:
        row = breakdown.setdefault(query["family"], {"total": 0, "ko": 0, "en": 0, "other": 0})
        row["total"] += 1
        row[query["language"]] += 1
    summary = {
        "documents": len(documents),
        "document_languages": {
            language: sum(1 for d in documents if d["language"] == language)
            for language in ("ko", "en", "other")
        },
        "queries": len(queries),
        "families": breakdown,
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
