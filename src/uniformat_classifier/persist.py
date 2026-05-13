"""JSONL persistence for user-confirmed labels.

Format (one JSON object per line):
    {"text": "...", "code": "A1010", "source": "user", "ts": "2026-04-28T12:34:56", "file": "...", "row": 42}

`source`:
    - "ingest": loaded from a historical Excel file (Niveau columns already filled)
    - "user":   confirmed/typed by the user during an active-learning session

We never deduplicate on disk so the file remains an append-only audit trail.
Deduplication happens in-memory at load time (last-write-wins per (text, source) pair).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .ingest import LabeledExample


@dataclass
class TrainingRecord:
    text: str
    code: str
    source: str           # "ingest" | "user"
    ts: str               # ISO timestamp
    file: str | None = None
    row: int | None = None

    @classmethod
    def from_example(cls, ex: LabeledExample) -> "TrainingRecord":
        return cls(
            text=ex.text,
            code=ex.code,
            source="ingest",
            ts=datetime.utcnow().isoformat(timespec="seconds"),
            file=ex.source_file,
            row=ex.source_row,
        )

    @classmethod
    def from_user(cls, text: str, code: str) -> "TrainingRecord":
        return cls(
            text=text,
            code=code,
            source="user",
            ts=datetime.utcnow().isoformat(timespec="seconds"),
        )


class TrainingStore:
    """Append-only JSONL store, with an in-memory deduplicated view."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[TrainingRecord] = []
        if path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self._records.append(TrainingRecord(**obj))
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"  ! skipping malformed line in {self.path.name}: {e}")

    def append(self, rec: TrainingRecord) -> None:
        self._records.append(rec)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def append_many(self, recs: list[TrainingRecord]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for rec in recs:
                self._records.append(rec)
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def deduped(self) -> list[TrainingRecord]:
        """Return one record per text. User-source records always win over ingest-source."""
        # First pass: keep latest per (text, source).
        latest: dict[tuple[str, str], TrainingRecord] = {}
        for rec in self._records:
            latest[(rec.text, rec.source)] = rec
        # Second pass: prefer 'user' over 'ingest' for the same text.
        merged: dict[str, TrainingRecord] = {}
        for (text, source), rec in latest.items():
            existing = merged.get(text)
            if existing is None:
                merged[text] = rec
            elif source == "user" and existing.source != "user":
                merged[text] = rec
        return list(merged.values())

    def __len__(self) -> int:
        return len(self._records)

    @property
    def all(self) -> list[TrainingRecord]:
        return list(self._records)


def seed_from_ingest(
    store: TrainingStore,
    examples: list[LabeledExample],
    *,
    only_if_empty: bool = True,
) -> int:
    """Append ingest-source records for any (text, code) not already present.

    Returns the number of new records written.
    """
    if only_if_empty and any(r.source == "ingest" for r in store._records):
        return 0
    existing_texts: set[str] = {r.text for r in store._records if r.source == "ingest"}
    new = [
        TrainingRecord.from_example(ex)
        for ex in examples
        if ex.text not in existing_texts
    ]
    if new:
        store.append_many(new)
    return len(new)
