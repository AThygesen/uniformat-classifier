"""End-to-end smoke test for the categorize pipeline (non-interactive).

Drives the same code paths the CLI uses, but auto-accepts the top-1 prediction
instead of prompting via questionary. Picks a small sample file, makes a
disposable copy, blanks out the Niveau columns, runs the pipeline, and reports
how many rows got classified plus a few sample predictions.

This exists only to prove the wiring is correct; not a unit test.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

# Ensure the local source dir is on the path even when run with `python smoketest.py`
sys.path.insert(0, str(Path(__file__).parent / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from uniformat_classifier.classifier import Classifier, fit_from_records  # noqa: E402
from uniformat_classifier.excel_io import (  # noqa: E402
    BENCHMARKING_COL,
    NIVEAU_1_COL,
    NIVEAU_2_COL,
    NIVEAU_3_COL,
    read_rows,
    write_back,
)
from uniformat_classifier.persist import TrainingStore  # noqa: E402
from uniformat_classifier.taxonomy import Taxonomy  # noqa: E402

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "Indskrivning_template.xlsx"
HISTORY = ROOT.parent / "Cost_Database_Projekter"
DATA = ROOT / "data"


def main() -> None:
    sample = HISTORY / "Bulkhal - Landdybet 14.xlsx"
    if not sample.exists():
        print(f"sample not found: {sample}")
        return

    workdir = Path(__file__).parent / "data" / "smoketest"
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / sample.name
    shutil.copy2(sample, target)

    # 1. Load taxonomy + training store
    print("Loading taxonomy + training store...")
    tax = Taxonomy.from_template(TEMPLATE)
    store = TrainingStore(DATA / "training_data.jsonl")
    print(f"  {len(store.deduped())} deduped training records")

    # 2. Fit classifier from the training store
    # Use a 5k-record subset to keep the smoke test fast (~1 min on CPU).
    SUBSET = 5000
    records = store.deduped()[:SUBSET]
    print(f"Encoding {len(records)} training records (subset of {len(store.deduped())})...")
    t0 = time.time()
    clf = Classifier(cache_dir=DATA / "embed_cache")
    fit_from_records(clf, records, show_progress=False)
    train_texts = [r.text for r in records]
    train_embs = clf.all_training_embeddings(train_texts)
    print(f"  took {time.time() - t0:.1f}s, {len(clf.known_codes())} prototypes")

    # 3. Read target file and blank out Niveau columns to simulate "new file"
    print(f"Reading sample file: {target.name}")
    rows = read_rows(target)
    print(f"  {len(rows)} data rows")

    # Blank Niveau cols for testing
    blank_updates = {r.sheet_row_ix: {NIVEAU_1_COL: "", NIVEAU_2_COL: "", NIVEAU_3_COL: ""}
                     for r in rows}
    write_back(target, blank_updates, backup=False)
    rows = read_rows(target)  # reload
    n_unlabeled = sum(1 for r in rows if not r.is_labeled)
    print(f"  blanked Niveau cols; {n_unlabeled}/{len(rows)} now unlabeled")

    # 4. Predict for each row, count actions, capture samples
    print()
    print("Predicting (auto-accepting top-1 — no user prompts)...")
    from uniformat_classifier.activelearn import Action, Policy
    policy = Policy()
    updates: dict[int, dict[str, str]] = {}
    counts = {Action.AUTO: 0, Action.CONFIRM: 0, Action.ASK: 0}
    samples = []
    for r in rows:
        if r.is_labeled:
            continue
        verdict = clf.predict(r.ydelse, top_k=3, all_training_embs=train_embs)
        action = policy.decide(verdict)
        counts[action] = counts.get(action, 0) + 1
        if not verdict.top:
            continue
        chosen = verdict.top[0].code
        n1, n2, n3 = tax.labels_for(chosen)
        bench = tax.get(chosen).benchmarking if tax.get(chosen) else ""
        fields: dict[str, str] = {}
        if n1: fields[NIVEAU_1_COL] = n1
        if n2: fields[NIVEAU_2_COL] = n2
        if n3: fields[NIVEAU_3_COL] = n3
        if bench: fields[BENCHMARKING_COL] = bench
        updates[r.sheet_row_ix] = fields
        if len(samples) < 8:
            samples.append((r.ydelse, chosen, n3 or n2 or n1, action, verdict.top[0].score))

    print(f"Action breakdown: AUTO={counts[Action.AUTO]}  "
          f"CONFIRM={counts[Action.CONFIRM]}  ASK={counts[Action.ASK]}")
    print()
    print("Sample predictions (Ydelse → predicted label):")
    for text, code, label, action, score in samples:
        print(f"  [{action.value:7s}] score={score:+.2f}  {text[:50]!r:55s} -> {code}  {label}")

    # 5. Write back
    print()
    print(f"Writing {len(updates)} rows back to {target.name}")
    write_back(target, updates, backup=False)

    # 6. Re-read and verify
    rows2 = read_rows(target)
    labeled_now = sum(1 for r in rows2 if r.is_labeled)
    print(f"After write-back: {labeled_now}/{len(rows2)} rows labeled")

    print()
    print(f"OK — output left at {target}")


if __name__ == "__main__":
    main()
