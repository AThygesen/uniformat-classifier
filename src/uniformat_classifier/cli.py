"""CLI entry point.

Commands:
    uniformat init       — first-time setup: ingest historical files, train, cache embeddings
    uniformat info       — show what's loaded and ready
    uniformat categorize <file.xlsx>  — classify a tenderlist with active learning
    uniformat eval       — stratified 80/20 holdout evaluation (1-of-K)
    uniformat reseed     — re-ingest historical files (overwriting old ingest records)
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

# Force UTF-8 on Windows consoles that default to cp1252.
# (Affects every CLI command, harmless on UTF-8 terminals.)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import questionary
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .activelearn import Action, Policy
from .classifier import Classifier, evaluate_holdout, fit_from_records
from .excel_io import (
    BENCHMARKING_COL,
    NIVEAU_1_COL,
    NIVEAU_2_COL,
    NIVEAU_3_COL,
    read_rows,
    write_back,
)
from .ingest import ingest_folder
from .persist import TrainingRecord, TrainingStore, seed_from_ingest
from .taxonomy import Taxonomy

app = typer.Typer(help="Active-learning Uniformat classifier", no_args_is_help=True)
console = Console()


# ---- paths ----------------------------------------------------------------

def _project_root() -> Path:
    """The 'Cost database dev' folder (one level above the uniformat_classifier package)."""
    here = Path(__file__).resolve()
    # src/uniformat_classifier/cli.py → up to uniformat_classifier (parent: 'Cost database dev')
    return here.parents[3]


def _data_dir() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2] / "data"


def _default_template() -> Path:
    return _project_root() / "Indskrivning_template.xlsx"


def _default_history_folder() -> Path:
    return _project_root() / "Cost_Database_Projekter"


def _training_jsonl() -> Path:
    return _data_dir() / "training_data.jsonl"


def _embed_cache_dir() -> Path:
    return _data_dir() / "embed_cache"


# ---- shared loaders -------------------------------------------------------

def _load_taxonomy() -> Taxonomy:
    template = _default_template()
    if not template.exists():
        console.print(f"[red]Template not found:[/red] {template}")
        raise typer.Exit(2)
    return Taxonomy.from_template(template)


def _load_store() -> TrainingStore:
    return TrainingStore(_training_jsonl())


def _load_and_fit_classifier(
    store: TrainingStore,
    *,
    show_progress: bool = True,
) -> tuple[Classifier, list[str]]:
    """Build classifier and return (classifier, training_texts) ready for prediction."""
    clf = Classifier(cache_dir=_embed_cache_dir())
    records = store.deduped()
    fit_from_records(clf, records, show_progress=show_progress)
    train_texts = [r.text for r in records]
    return clf, train_texts


# =========================================================================
# init
# =========================================================================

@app.command()
def init(
    history: Path = typer.Option(
        None,
        help="Folder of historical tenderlist .xlsx files (default: ./Cost_Database_Projekter)",
    ),
    encode_now: bool = typer.Option(
        True,
        "--encode-now/--no-encode-now",
        help="Encode all training texts now (caches embeddings; ~1 hr first time on CPU). "
             "If --no-encode-now, embeddings will be computed on first use.",
    ),
) -> None:
    """One-time setup: ingest historical files into the JSONL training store."""
    history = history or _default_history_folder()
    if not history.exists():
        console.print(f"[red]History folder not found:[/red] {history}")
        raise typer.Exit(2)

    tax = _load_taxonomy()
    console.print(f"Loaded taxonomy: [bold]{len(tax.all())}[/bold] entries "
                  f"({len(tax.niveau3())} Niveau-3 classes)")

    store = _load_store()
    console.print(f"Training store: [bold]{_training_jsonl()}[/bold] ({len(store)} records)")

    console.print(f"Ingesting from [bold]{history}[/bold]...")
    examples = ingest_folder(history, tax)
    console.print(f"  → {len(examples)} labeled examples extracted")

    n_added = seed_from_ingest(store, examples, only_if_empty=True)
    if n_added:
        console.print(f"  → wrote {n_added} new ingest records to {_training_jsonl()}")
    else:
        console.print("  → store already has ingest records (skipped reseed); "
                      "use [bold]reseed[/bold] to refresh")

    if encode_now:
        console.print()
        console.print("Encoding all training texts (this may take a while)...")
        clf, _ = _load_and_fit_classifier(store, show_progress=True)
        console.print(f"  → {len(clf.known_codes())} prototypes built. Embeddings cached.")
    console.print("[green]Done.[/green] Run [bold]uniformat info[/bold] to see status.")


# =========================================================================
# reseed
# =========================================================================

@app.command()
def reseed(
    history: Path = typer.Option(
        None,
        help="Folder of historical tenderlist .xlsx files",
    ),
) -> None:
    """Re-ingest historical files. Drops existing ingest-source records, keeps user records."""
    history = history or _default_history_folder()
    tax = _load_taxonomy()

    p = _training_jsonl()
    if p.exists():
        # Filter out 'ingest' records, keep only 'user' records
        kept = []
        with p.open("r", encoding="utf-8") as f:
            import json
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("source") == "user":
                    kept.append(line)
        p.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        console.print(f"Kept {len(kept)} user records, dropped ingest records.")

    examples = ingest_folder(history, tax)
    store = _load_store()
    n_added = seed_from_ingest(store, examples, only_if_empty=False)
    console.print(f"Wrote {n_added} fresh ingest records.")


# =========================================================================
# info
# =========================================================================

@app.command()
def info() -> None:
    """Show training store stats and class coverage."""
    tax = _load_taxonomy()
    store = _load_store()
    deduped = store.deduped()

    console.print(Panel.fit(
        f"[bold]Uniformat classifier status[/bold]\n\n"
        f"Template:       {_default_template()}\n"
        f"History folder: {_default_history_folder()}\n"
        f"Training JSONL: {_training_jsonl()}\n"
        f"Embedding cache: {_embed_cache_dir()}",
        border_style="cyan",
    ))

    table = Table(title="Training records")
    table.add_column("Source", style="cyan")
    table.add_column("Count", justify="right")
    by_source = Counter(r.source for r in store.all)
    for src in ("ingest", "user"):
        table.add_row(src, str(by_source.get(src, 0)))
    table.add_row("[bold]deduped[/bold]", f"[bold]{len(deduped)}[/bold]")
    console.print(table)

    by_level = Counter(len(r.code) for r in deduped)
    by_n1 = Counter(r.code[0] for r in deduped)
    n3_codes = {r.code for r in deduped if len(r.code) == 5}

    table2 = Table(title="Class coverage")
    table2.add_column("Level", style="cyan")
    table2.add_column("In data", justify="right")
    table2.add_column("In taxonomy", justify="right")
    table2.add_column("Coverage", justify="right")
    n1_in_data = len({r.code[0] for r in deduped})
    table2.add_row("Niveau 1 (A–H)", str(n1_in_data), str(len(tax.at_level(1))),
                   f"{n1_in_data / max(1, len(tax.at_level(1))) * 100:.0f}%")
    table2.add_row("Niveau 3 (5-char)", str(len(n3_codes)), str(len(tax.at_level(3))),
                   f"{len(n3_codes) / max(1, len(tax.at_level(3))) * 100:.0f}%")
    console.print(table2)

    table3 = Table(title="Records by Niveau-1 letter")
    table3.add_column("Letter")
    table3.add_column("Records", justify="right")
    table3.add_column("Description")
    for letter in sorted(by_n1):
        entry = tax.get(letter)
        desc = entry.label if entry else "?"
        table3.add_row(letter, str(by_n1[letter]), desc)
    console.print(table3)


# =========================================================================
# eval
# =========================================================================

@app.command()
def eval(
    test_size: float = typer.Option(0.2, help="Fraction of data held out for testing"),
    seed: int = typer.Option(42, help="Random seed"),
    only_n3: bool = typer.Option(
        True,
        "--only-n3/--all-levels",
        help="Restrict eval to records with full N3 (5-char) codes (the prediction target)",
    ),
) -> None:
    """Stratified 80/20 holdout evaluation (the '1-of-K' test)."""
    tax = _load_taxonomy()
    store = _load_store()
    records = store.deduped()
    if only_n3:
        records = [r for r in records if len(r.code) == 5]

    if not records:
        console.print("[red]No training records — run `uniformat init` first.[/red]")
        raise typer.Exit(1)

    console.print(f"Eval set: {len(records)} records, {len(set(r.code for r in records))} classes")
    console.print(f"Holdout fraction: {test_size}, seed={seed}")

    clf = Classifier(cache_dir=_embed_cache_dir())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating", total=int(len(records) * test_size))

        def _cb(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        result = evaluate_holdout(
            clf,
            [r.text for r in records],
            [r.code for r in records],
            test_size=test_size,
            seed=seed,
            progress=_cb,
        )

    console.print()
    table = Table(title="Holdout results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Test rows", str(result.n))
    table.add_row("Top-1 accuracy", f"{result.top1_acc * 100:.1f}%")
    table.add_row("Top-3 accuracy", f"{result.top3_acc * 100:.1f}%")
    table.add_row("Niveau-1 accuracy", f"{result.n1_acc * 100:.1f}%")
    console.print(table)

    if result.confusions:
        console.print()
        ct = Table(title="Top confusions (true → predicted)")
        ct.add_column("True"); ct.add_column("True label")
        ct.add_column("Predicted"); ct.add_column("Predicted label")
        ct.add_column("Count", justify="right")
        for tc, pc, n in result.confusions:
            tl = (tax.get(tc).label if tax.get(tc) else "?")[:48]
            pl = (tax.get(pc).label if tax.get(pc) else "?")[:48]
            ct.add_row(tc, tl, pc, pl, str(n))
        console.print(ct)


# =========================================================================
# categorize
# =========================================================================

@app.command()
def categorize(
    file: Path = typer.Argument(..., help="The tenderlist .xlsx file to categorize"),
    only_unlabeled: bool = typer.Option(
        True,
        "--only-unlabeled/--all-rows",
        help="Skip rows with Niveau 1 already filled; auto-fill N1/N2 when N3 exists",
    ),
    backup: bool = typer.Option(True, help="Save a .bak copy before writing"),
    ingest_complete: bool = typer.Option(
        False,
        "--ingest-complete/--no-ingest",
        help="Ingest fully-labeled rows (N1+N2+N3 filled) into training data for future models",
    ),
    non_interactive: bool = typer.Option(
        False,
        help="Non-interactive mode: auto-accept all top-1 predictions (no prompts)",
    ),
) -> None:
    """Classify rows in a tenderlist file with active learning."""
    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(2)

    tax = _load_taxonomy()
    store = _load_store()

    if len(store) == 0:
        console.print("[red]Training store is empty — run `uniformat init` first.[/red]")
        raise typer.Exit(1)

    console.print("Loading classifier...")
    clf, train_texts = _load_and_fit_classifier(store, show_progress=False)
    train_embs = clf.all_training_embeddings(train_texts) if train_texts else None

    rows = read_rows(file)

    # Split rows into three buckets:
    # 1. Fully labeled (N1+N2+N3): skip or ingest
    # 2. Partially labeled (N3 but missing N1/N2): auto-fill parents, no user input needed
    # 3. Unlabeled (no N3): need prediction
    fully_labeled = []
    partial_n3_only = []
    needs_prediction = []

    for r in rows:
        if r.is_labeled and r.has_niveau3:
            fully_labeled.append(r)
        elif r.has_niveau3 and not r.is_labeled:
            partial_n3_only.append(r)
        elif not r.is_labeled:
            needs_prediction.append(r)

    if only_unlabeled:
        targets = needs_prediction
    else:
        targets = needs_prediction + partial_n3_only

    console.print(f"File: [bold]{file.name}[/bold]  rows={len(rows)}")
    console.print(f"  Fully labeled (skip): {len(fully_labeled)}")
    console.print(f"  Partial (N3→N1/N2 auto-fill): {len(partial_n3_only)}")
    console.print(f"  Needs prediction: {len(needs_prediction)}")
    console.print(f"  To process: {len(targets)}")

    if not targets:
        console.print("[green]Nothing to do.[/green]")
        return

    policy = Policy()
    updates: dict[int, dict[str, str]] = {}
    new_user_records: list[TrainingRecord] = []

    auto_count = confirm_count = ask_count = skip_count = 0

    # 1. Auto-fill partial rows (N3 → N1/N2)
    console.print()
    console.print(f"Auto-filling {len(partial_n3_only)} partial rows (N3 → N1/N2)...")
    for r in partial_n3_only:
        code = tax.code_from_label(r.niveau3) if r.niveau3 else None
        if code is None:
            console.print(f"  ! row {r.sheet_row_ix}: could not parse Niveau 3 '{r.niveau3}'")
            continue
        n1, n2, n3 = tax.labels_for(code)
        bench = tax.get(code).benchmarking if tax.get(code) else ""
        fields: dict[str, str] = {}
        if n1: fields[NIVEAU_1_COL] = n1
        if n2: fields[NIVEAU_2_COL] = n2
        if bench: fields[BENCHMARKING_COL] = bench
        updates[r.sheet_row_ix] = fields
    if partial_n3_only:
        console.print(f"  → {len(partial_n3_only)} rows updated")

    # 2. Predict for rows that need it
    console.print()
    console.print(f"Predicting for {len(needs_prediction)} unlabeled rows...")
    for i, row in enumerate(needs_prediction, start=1):
        console.rule(f"[bold]{i}/{len(needs_prediction)}[/bold]  row {row.sheet_row_ix}")
        console.print(f"  [cyan]Ydelse:[/cyan] {row.ydelse}")
        verdict = clf.predict(row.ydelse, top_k=3, all_training_embs=train_embs)
        action = policy.decide(verdict)

        if non_interactive or not _is_interactive():
            # Non-interactive mode: auto-accept best prediction
            if not verdict.top:
                skip_count += 1
                console.print("  [yellow]No predictions; skipping[/yellow]")
                continue
            chosen = verdict.top[0].code
            auto_count += 1
            n1, n2, n3 = tax.labels_for(chosen)
            console.print(f"  [green]AUTO[/green] (non-interactive) -> {chosen}  {n3 or n2 or n1}")
        elif not verdict.top:
            chosen = _ask_user_pick(row.ydelse, [], tax)
            if chosen is None:
                skip_count += 1
                continue
            ask_count += 1
        elif action == Action.AUTO:
            chosen = verdict.top[0].code
            auto_count += 1
            n1, n2, n3 = tax.labels_for(chosen)
            console.print(f"  [green]AUTO[/green] -> {chosen}  {n3 or n2 or n1}")
        elif action == Action.CONFIRM:
            chosen = _ask_user_confirm(row.ydelse, verdict, tax)
            if chosen is None:
                skip_count += 1
                continue
            confirm_count += 1
        else:  # ASK
            chosen = _ask_user_pick(row.ydelse, verdict.top, tax)
            if chosen is None:
                skip_count += 1
                continue
            ask_count += 1

        # Update model online so subsequent rows benefit immediately
        if action != Action.AUTO:
            clf.add_example(row.ydelse, chosen)
            new_user_records.append(TrainingRecord.from_user(row.ydelse, chosen))

        n1, n2, n3 = tax.labels_for(chosen)
        bench = tax.get(chosen).benchmarking if tax.get(chosen) else ""
        fields: dict[str, str] = {}
        if n1: fields[NIVEAU_1_COL] = n1
        if n2: fields[NIVEAU_2_COL] = n2
        if n3: fields[NIVEAU_3_COL] = n3
        if bench: fields[BENCHMARKING_COL] = bench
        updates[row.sheet_row_ix] = fields

    # 3. Optionally ingest fully-labeled rows as training data
    ingested_count = 0
    if ingest_complete and fully_labeled:
        console.print()
        console.print(f"Ingesting {len(fully_labeled)} fully-labeled rows into training data...")
        for r in fully_labeled:
            code = tax.code_from_label(r.niveau3) if r.niveau3 else None
            if code is None:
                continue
            ingested_count += 1
            new_user_records.append(TrainingRecord.from_user(r.ydelse, code))
        console.print(f"  → {ingested_count} rows ingested")

    # Persist user feedback
    if new_user_records:
        store.append_many(new_user_records)
        console.print(f"\n[green]Saved {len(new_user_records)} new user labels to {_training_jsonl()}[/green]")

    # Write back
    if updates:
        console.print(f"\nWriting {len(updates)} updated rows back to [bold]{file.name}[/bold]"
                      + (" (with .bak)" if backup else ""))
        write_back(file, updates, backup=backup)

    console.print()
    console.print(Panel.fit(
        f"Partial auto-filled (N3→N1/N2): {len(partial_n3_only)}\n"
        f"Predictions - AUTO accepted: {auto_count}\n"
        f"Predictions - User confirmed: {confirm_count}\n"
        f"Predictions - User picked/typed: {ask_count}\n"
        f"Predictions - Skipped: {skip_count}\n"
        f"Fully-labeled ingested to training: {ingested_count}\n"
        f"Total rows updated: {len(updates)}",
        title="Summary",
        border_style="green",
    ))


# ---- interactive prompts -------------------------------------------------

def _is_interactive() -> bool:
    """Check if stdout is connected to a terminal (not piped)."""
    return sys.stdout.isatty() and sys.stdin.isatty()


def _ask_user_confirm(text: str, verdict, tax: Taxonomy) -> str | None:
    """Show top-1 prediction; let user accept, pick alternative, type a code, or skip."""
    if not _is_interactive():
        console.print("  [yellow]Non-interactive mode:[/yellow] auto-accepting top-1")
        return verdict.top[0].code if verdict.top else None

    top = verdict.top[0]
    n1, n2, n3 = tax.labels_for(top.code)
    label = n3 or n2 or n1 or top.code
    console.print(f"  [yellow]Suggested[/yellow] (score={top.score:.2f}, "
                  f"margin={top.margin:.2f}): [bold]{top.code}[/bold]  {label}")
    choices = [
        questionary.Choice(title=f"✓ Accept: {top.code}  {label}", value=("accept", top.code)),
        *[
            questionary.Choice(
                title=f"  Pick alt: {p.code}  "
                      f"{(tax.labels_for(p.code)[2] or tax.labels_for(p.code)[1] or tax.labels_for(p.code)[0] or '?')}"
                      f"  (score={p.score:.2f})",
                value=("alt", p.code),
            )
            for p in verdict.top[1:]
        ],
        questionary.Choice(title="✎ Type a different code...", value=("type", None)),
        questionary.Choice(title="↷ Skip this row", value=("skip", None)),
    ]
    answer = questionary.select("How to label this row?", choices=choices).ask()
    if answer is None or answer[0] == "skip":
        return None
    if answer[0] in ("accept", "alt"):
        return answer[1]
    # type
    return _prompt_for_code(tax)


def _ask_user_pick(text: str, suggestions, tax: Taxonomy) -> str | None:
    """Low-confidence path — present the top-3 plus typing/skip options."""
    if not _is_interactive():
        if suggestions:
            console.print(f"  [yellow]Non-interactive mode:[/yellow] auto-accepting top-1")
            return suggestions[0].code
        else:
            console.print("  [yellow]Non-interactive mode:[/yellow] no suggestions, skipping row")
            return None

    if suggestions:
        console.print("  [yellow]Low-confidence suggestions:[/yellow]")
        for p in suggestions:
            n1, n2, n3 = tax.labels_for(p.code)
            console.print(f"    {p.code:6s}  score={p.score:+.2f}   {n3 or n2 or n1 or '?'}")
    else:
        console.print("  [yellow]No prior examples are similar enough to suggest.[/yellow]")

    choices = [
        questionary.Choice(
            title=f"  Pick: {p.code}  "
                  f"{(tax.labels_for(p.code)[2] or tax.labels_for(p.code)[1] or tax.labels_for(p.code)[0] or '?')}"
                  f"  (score={p.score:.2f})",
            value=("pick", p.code),
        )
        for p in suggestions
    ]
    choices.append(questionary.Choice(title="✎ Type a code (e.g. A1010)...", value=("type", None)))
    choices.append(questionary.Choice(title="↷ Skip this row", value=("skip", None)))

    answer = questionary.select("How to label this row?", choices=choices).ask()
    if answer is None or answer[0] == "skip":
        return None
    if answer[0] == "pick":
        return answer[1]
    return _prompt_for_code(tax)


def _prompt_for_code(tax: Taxonomy) -> str | None:
    """Free-text code entry with autocomplete from the taxonomy."""
    all_codes = [e.code for e in tax.all()]
    code = questionary.autocomplete(
        "Enter Uniformat code (Tab to autocomplete):",
        choices=all_codes,
    ).ask()
    if not code:
        return None
    code = code.strip()
    entry = tax.get(code)
    if entry is None:
        console.print(f"  [red]Code '{code}' not in taxonomy.[/red] Skipping.")
        return None
    n1, n2, n3 = tax.labels_for(code)
    console.print(f"  [green]✓[/green] {code}  {n3 or n2 or n1}")
    return code


if __name__ == "__main__":
    app()
