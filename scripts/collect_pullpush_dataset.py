#!/usr/bin/env python3
"""Collect and draft-label public r/nba comments for TakeMeter.

The labels are intentionally draft labels. Review the output CSV before treating
the dataset as final course submission data.
"""

from __future__ import annotations

import csv
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


API_URL = "https://api.pullpush.io/reddit/search/comment/"
SUBREDDIT = "nba"
TARGET_PER_LABEL = 70
OUTPUT_PATH = Path("data/takemeter_nba_labeled.csv")
RAW_PATH = Path("data/raw_pullpush_comments.jsonl")
CACHE_DIR = Path("data/pullpush")


SEARCH_QUERIES = {
    "general": [
        "",
        "playoffs",
        "finals",
        "Lakers",
        "Celtics",
        "Nuggets",
        "Thunder",
        "Timberwolves",
        "Knicks",
        "Mavericks",
    ],
    "analysis": [
        "because",
        "defense",
        "offense",
        "spacing",
        "lineup",
        "matchup",
        "efficiency",
        "usage",
        "rotation",
        "transition",
        "pick and roll",
        "rebounding",
        "possessions",
        "salary cap",
        "adjustments",
    ],
    "hot_take": [
        "overrated",
        "washed",
        "fraud",
        "exposed",
        "trash",
        "fake contender",
        "not winning",
        "best player",
        "worst",
        "choke",
        "no chance",
        "top 5",
        "MVP",
    ],
    "reaction": [
        "lol",
        "lmao",
        "wow",
        "insane",
        "crazy",
        "that dunk",
        "refs",
        "thread",
        "hilarious",
        "nasty",
        "beautiful",
        "what a game",
    ],
}


ANALYSIS_TERMS = {
    "because",
    "since",
    "if",
    "when",
    "why",
    "means",
    "reason",
    "adjustment",
    "adjustments",
    "scheme",
    "coverage",
    "spacing",
    "lineup",
    "lineups",
    "rotation",
    "rotations",
    "matchup",
    "matchups",
    "switch",
    "switching",
    "drop",
    "zone",
    "transition",
    "half court",
    "pick and roll",
    "screen",
    "screens",
    "rim",
    "paint",
    "corner",
    "weak side",
    "rebounding",
    "usage",
    "efficiency",
    "possessions",
    "turnovers",
    "assist",
    "defense",
    "offense",
    "shooting",
    "salary",
    "cap",
    "contract",
}

HOT_TAKE_TERMS = {
    "overrated",
    "washed",
    "fraud",
    "frauds",
    "exposed",
    "trash",
    "garbage",
    "terrible",
    "awful",
    "fake",
    "contender",
    "cooked",
    "choke",
    "choked",
    "choker",
    "never",
    "always",
    "best",
    "worst",
    "top",
    "mvp",
    "goat",
    "no chance",
    "not winning",
    "can't win",
    "cannot win",
    "not good",
    "unserious",
}

REACTION_TERMS = {
    "lol",
    "lmao",
    "lmfao",
    "haha",
    "wow",
    "damn",
    "insane",
    "crazy",
    "wild",
    "nasty",
    "disgusting",
    "beautiful",
    "smooth",
    "hilarious",
    "thread",
    "refs",
    "ref",
    "foul",
    "whistle",
    "what a game",
    "no way",
    "bruh",
    "bro",
}

BOT_MARKERS = {
    "i am a bot",
    "beep boop",
    "automoderator",
    "your post has been removed",
    "message the moderators",
}

OFF_TOPIC_SPORT_TERMS = {
    "qb",
    "quarterback",
    "touchdown",
    "interception",
    "interceptions",
    "jameis",
    "devito",
    "super bowl",
    "nfl",
}

BASKETBALL_CONTEXT_TERMS = {
    "nba",
    "basketball",
    "player",
    "players",
    "team",
    "teams",
    "coach",
    "lineup",
    "defense",
    "offense",
    "shooting",
    "spacing",
    "rim",
    "paint",
    "rebound",
    "rebounding",
    "assist",
    "turnover",
    "three",
    "3pt",
    "playoffs",
    "finals",
    "lakers",
    "celtics",
    "nuggets",
    "thunder",
    "wolves",
    "knicks",
    "mavs",
    "warriors",
    "bucks",
    "sixers",
    "suns",
    "heat",
}


@dataclass(frozen=True)
class Candidate:
    comment_id: str
    text: str
    label: str
    notes: str
    source_permalink: str
    created_utc: int
    score: int
    confidence_score: int


def fetch_comments(query: str, size: int = 100) -> list[dict]:
    params = {
        "subreddit": SUBREDDIT,
        "size": str(size),
        "sort": "desc",
        "sort_type": "created_utc",
    }
    if query:
        params["q"] = query
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "takemeter-course-project/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("data", [])


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_usable(text: str) -> bool:
    lower = text.lower()
    if lower in {"[deleted]", "[removed]"}:
        return False
    if any(marker in lower for marker in BOT_MARKERS):
        return False
    if "http://" in lower or "https://" in lower:
        return False
    off_topic_hits = term_hits(lower, OFF_TOPIC_SPORT_TERMS)
    basketball_hits = term_hits(lower, BASKETBALL_CONTEXT_TERMS)
    if off_topic_hits >= 2 and basketball_hits == 0:
        return False
    words = re.findall(r"[A-Za-z0-9']+", text)
    if len(words) < 5:
        return False
    if len(words) > 120:
        return False
    alpha = sum(ch.isalpha() for ch in text)
    if alpha < max(12, len(text) * 0.45):
        return False
    return True


def term_hits(lower: str, terms: set[str]) -> int:
    return sum(1 for term in terms if term in lower)


def label_comment(text: str) -> tuple[str, str, int]:
    lower = text.lower()
    words = re.findall(r"[A-Za-z0-9']+", lower)
    word_count = len(words)
    sentence_count = max(1, len(re.findall(r"[.!?]+", text)))
    numeric_evidence = len(re.findall(r"\b\d+(\.\d+)?\s*(%|ppg|rpg|apg|mpg|seed|seeds|wins|losses)?\b", lower))

    analysis_score = 0
    analysis_score += term_hits(lower, ANALYSIS_TERMS)
    analysis_score += 2 if numeric_evidence else 0
    analysis_score += 2 if word_count >= 28 else 0
    analysis_score += 1 if sentence_count >= 2 else 0
    analysis_score += 1 if any(marker in lower for marker in ["but", "however", "whereas", "compared to"]) else 0

    hot_take_score = 0
    hot_take_score += term_hits(lower, HOT_TAKE_TERMS)
    hot_take_score += 2 if any(marker in lower for marker in ["no chance", "not winning", "can't win", "cannot win"]) else 0
    hot_take_score += 1 if "!" in text else 0
    hot_take_score += 1 if word_count <= 35 else 0
    hot_take_score += 1 if any(marker in lower for marker in ["isn't", "is not", "are not", "ain't"]) else 0

    reaction_score = 0
    reaction_score += term_hits(lower, REACTION_TERMS)
    reaction_score += 2 if word_count <= 18 else 0
    reaction_score += 1 if re.search(r"\b(l+o+l+|lmao+|lmfao+|haha+)\b", lower) else 0
    reaction_score += 1 if "?" in text or "!" in text else 0
    reaction_score += 1 if any(ch in text for ch in ["😂", "😭", "💀"]) else 0

    scores = {
        "analysis": analysis_score,
        "hot_take": hot_take_score,
        "reaction": reaction_score,
    }

    # Prefer substantive reasoning over inflammatory wording when both appear.
    if analysis_score >= 4 and analysis_score >= hot_take_score:
        label = "analysis"
    elif hot_take_score >= 3 and hot_take_score >= reaction_score:
        label = "hot_take"
    elif reaction_score >= 2:
        label = "reaction"
    else:
        label = max(scores, key=scores.get)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    confidence_score = ordered[0][1] - ordered[1][1]
    notes = ""
    if confidence_score <= 1:
        pair = "/".join(sorted([ordered[0][0], ordered[1][0]]))
        notes = f"Borderline {pair}; draft label chosen by taxonomy decision rule."
    elif label == "analysis" and hot_take_score >= 3:
        notes = "Strong claim with supporting basketball reasoning; labeled analysis."
    elif label == "hot_take" and analysis_score >= 3:
        notes = "Claim uses basketball terms but evidence is too thin; labeled hot_take."
    elif label == "reaction" and hot_take_score >= 3:
        notes = "Emotional immediate response despite strong wording; labeled reaction."

    return label, notes, max(scores.values())


def make_candidate(raw: dict) -> Candidate | None:
    text = normalize_text(str(raw.get("body", "")))
    if not is_usable(text):
        return None
    comment_id = str(raw.get("id", "")).strip()
    if not comment_id:
        return None
    label, notes, confidence_score = label_comment(text)
    permalink = str(raw.get("permalink", "")).strip()
    source_permalink = f"https://www.reddit.com{permalink}" if permalink.startswith("/") else permalink
    created_utc = int(float(raw.get("created_utc", 0) or 0))
    score = int(raw.get("score", 0) or 0)
    return Candidate(
        comment_id=comment_id,
        text=text,
        label=label,
        notes=notes,
        source_permalink=source_permalink,
        created_utc=created_utc,
        score=score,
        confidence_score=confidence_score,
    )


def collect_raw_comments() -> dict[str, dict]:
    seen: dict[str, dict] = {}
    for group, queries in SEARCH_QUERIES.items():
        for query in queries:
            try:
                rows = fetch_comments(query)
            except Exception as exc:
                print(f"Fetch failed for {group}:{query!r}: {exc}")
                continue
            for row in rows:
                comment_id = str(row.get("id", "")).strip()
                if comment_id:
                    seen[comment_id] = row
            print(f"{group}:{query or '<general>'} -> {len(rows)} rows, {len(seen)} unique")
            time.sleep(0.25)
    return seen


def collect_cached_comments() -> dict[str, dict]:
    seen: dict[str, dict] = {}
    for path in sorted(CACHE_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue
        rows = payload.get("data", [])
        for row in rows:
            comment_id = str(row.get("id", "")).strip()
            if comment_id:
                seen[comment_id] = row
        print(f"{path.name} -> {len(rows)} rows, {len(seen)} unique")
    return seen


def choose_balanced(candidates: list[Candidate]) -> list[Candidate]:
    buckets = {"analysis": [], "hot_take": [], "reaction": []}
    for candidate in candidates:
        buckets[candidate.label].append(candidate)

    selected: list[Candidate] = []
    for label, rows in buckets.items():
        rows = sorted(
            rows,
            key=lambda item: (
                item.confidence_score,
                1 if item.notes else 0,
                item.score,
                item.created_utc,
            ),
            reverse=True,
        )
        take = rows[:TARGET_PER_LABEL]
        selected.extend(take)
        print(f"{label}: selected {len(take)} of {len(rows)} candidates")

    selected = sorted(selected, key=lambda item: (item.label, item.created_utc, item.comment_id))
    return selected


def write_outputs(raw_comments: dict[str, dict], selected: list[Candidate]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_PATH.open("w", encoding="utf-8") as handle:
        for row in raw_comments.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text", "label", "notes", "source_permalink", "comment_id"],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "text": row.text,
                    "label": row.label,
                    "notes": row.notes,
                    "source_permalink": row.source_permalink,
                    "comment_id": row.comment_id,
                }
            )


def main() -> None:
    raw_comments = collect_cached_comments() if "--from-cache" in sys.argv else collect_raw_comments()
    candidates = [candidate for row in raw_comments.values() if (candidate := make_candidate(row))]
    selected = choose_balanced(candidates)
    write_outputs(raw_comments, selected)
    print(f"Wrote {len(selected)} labeled examples to {OUTPUT_PATH}")
    if len(selected) < TARGET_PER_LABEL * 3:
        raise SystemExit("Not enough examples were collected for a balanced dataset.")


if __name__ == "__main__":
    main()
