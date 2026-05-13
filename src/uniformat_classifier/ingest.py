"""Ingest historical tenderlist Excel files into labeled training examples.

A row is usable as training data if its `Ydelse` text plus at least one of `Niveau 1/2/3`
can be resolved to a Uniformat code via the taxonomy.

Resolution priority (most-specific first): Niveau 3 -> Niveau 2 -> Niveau 1.
Returned codes therefore vary in length (1, 3, or 5 chars), but most rows resolve to
Niveau 1 or Niveau 3 in practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .excel_io import read_rows
from .taxonomy import Taxonomy


@dataclass(frozen=True)
class LabeledExample:
    text: str           # Ydelse
    code: str           # Uniformat code (most specific available)
    source_file: str    # filename for traceability
    source_row: int     # 1-indexed Excel row


def ingest_folder(
    folder: Path,
    taxonomy: Taxonomy,
    *,
    skip_files: set[str] | None = None,
) -> list[LabeledExample]:
    skip_files = skip_files or set()
    out: list[LabeledExample] = []
    for xlsx in sorted(folder.glob("*.xlsx")):
        if xlsx.name in skip_files or xlsx.name.startswith("~$"):
            continue
        try:
            rows = read_rows(xlsx)
        except Exception as e:
            # One bad file should not kill ingest; surface and move on.
            print(f"  ! skipping {xlsx.name}: {e}")
            continue
        for r in rows:
            code = _best_code(r.niveau3, r.niveau2, r.niveau1, taxonomy)
            if code is None:
                continue
            out.append(
                LabeledExample(
                    text=r.ydelse,
                    code=code,
                    source_file=xlsx.name,
                    source_row=r.sheet_row_ix,
                )
            )
    return out


def _best_code(
    niveau3: str | None,
    niveau2: str | None,
    niveau1: str | None,
    taxonomy: Taxonomy,
) -> str | None:
    for label in (niveau3, niveau2, niveau1):
        if not label:
            continue
        code = taxonomy.code_from_label(label)
        if code is not None:
            return code
    return None
