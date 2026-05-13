"""Streamlit UI for the Uniformat classifier — simplified version.

Run with: uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Force UTF-8 on Windows
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import streamlit as st
from streamlit_option_menu import option_menu

from uniformat_classifier.activelearn import Policy
from uniformat_classifier.classifier import Classifier, fit_from_records
from uniformat_classifier.excel_io import (
    BENCHMARKING_COL,
    NIVEAU_1_COL,
    NIVEAU_2_COL,
    NIVEAU_3_COL,
    read_rows,
    write_back,
)
from uniformat_classifier.persist import TrainingRecord, TrainingStore
from uniformat_classifier.taxonomy import Taxonomy

# =========================================================================
# Config
# =========================================================================

PROJECT_ROOT = Path(__file__).parent
TEMPLATE = PROJECT_ROOT / "Indskrivning_template.xlsx"
DATA = Path(__file__).parent / "data"

st.set_page_config(
    page_title="Uniformat Classifier",
    page_icon="🏗️",
    layout="wide",
)

# =========================================================================
# Cached loaders
# =========================================================================

@st.cache_resource
def load_taxonomy() -> Taxonomy:
    return Taxonomy.from_template(TEMPLATE)


@st.cache_resource
def load_store() -> TrainingStore:
    return TrainingStore(DATA / "training_data.jsonl")


@st.cache_resource
def load_classifier_and_texts() -> tuple[Classifier, list[str]]:
    """Load and fit classifier from training store."""
    store = load_store()
    clf = Classifier(cache_dir=DATA / "embed_cache")
    records = store.deduped()
    fit_from_records(clf, records, show_progress=False)
    train_texts = [r.text for r in records]
    return clf, train_texts


# =========================================================================
# Pages
# =========================================================================

def page_home():
    """Dashboard."""
    st.title("🏗️ Uniformat Classifier")
    st.markdown("Active-learning classification of Danish construction tenderlist services.")

    tax = load_taxonomy()
    store = load_store()
    deduped = store.deduped()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Records", len(deduped))
    with col2:
        n3_codes = len({r.code for r in deduped if len(r.code) == 5})
        st.metric("Niveau-3 Classes", f"{n3_codes}/85")
    with col3:
        n1_count = len({r.code[0] for r in deduped})
        st.metric("Niveau-1 Coverage", f"{n1_count}/8")
    with col4:
        cache_exists = (DATA / "embed_cache").exists()
        st.metric("Embedding Cache", "Ready" if cache_exists else "Not found")

    st.divider()
    st.subheader("How to use")
    st.markdown("""
    1. **Categorize** — Upload an Excel file to categorize rows
    2. **View Data** — Browse training records
    3. **Check Model** — See statistics and accuracy
    """)


def page_categorize():
    """Interactive categorization page."""
    st.title("📋 Categorize Tenderlist")

    tax = load_taxonomy()
    store = load_store()
    clf, train_texts = load_classifier_and_texts()
    train_embs = clf.all_training_embeddings(train_texts) if train_texts else None

    uploaded_file = st.file_uploader("Upload Excel tenderlist", type=["xlsx"])
    if not uploaded_file:
        st.info("Upload an Excel file to begin")
        return

    # Read file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        rows = read_rows(tmp_path)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return

    # Split rows
    fully_labeled = [r for r in rows if r.is_labeled and r.has_niveau3]
    partial_n3 = [r for r in rows if r.has_niveau3 and not r.is_labeled]
    needs_pred = [r for r in rows if not r.is_labeled]

    st.subheader("File Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Rows", len(rows))
    with c2:
        st.metric("Fully Labeled (skip)", len(fully_labeled))
    with c3:
        st.metric("Partial N3 (auto-fill)", len(partial_n3))
    with c4:
        st.metric("Need Prediction", len(needs_pred))

    updates: dict[int, dict[str, str]] = {}
    new_records: list[TrainingRecord] = []

    # Auto-fill partial rows
    if partial_n3:
        st.subheader("Auto-filling Partial Rows")
        count = 0
        for r in partial_n3:
            code = tax.code_from_label(r.niveau3) if r.niveau3 else None
            if code:
                n1, n2, n3 = tax.labels_for(code)
                bench = tax.get(code).benchmarking if tax.get(code) else ""
                fields: dict[str, str] = {}
                if n1: fields[NIVEAU_1_COL] = n1
                if n2: fields[NIVEAU_2_COL] = n2
                if bench: fields[BENCHMARKING_COL] = bench
                updates[r.sheet_row_ix] = fields
                count += 1
        st.success(f"Auto-filled {count} rows (N3 -> N1/N2)")

    # Predict
    if needs_pred:
        st.subheader(f"Predictions for {len(needs_pred)} Unlabeled Rows")
        policy = Policy()

        # Show predictions
        for i, row in enumerate(needs_pred[:50]):  # Limit display to 50 rows
            verdict = clf.predict(row.ydelse, top_k=3, all_training_embs=train_embs)

            with st.expander(f"Row {i+1}: {row.ydelse[:70]}"):
                if verdict.top:
                    # Show top 3 predictions
                    col1, col2, col3 = st.columns(3)
                    pred_choice = None
                    for j, pred in enumerate(verdict.top[:3]):
                        n1, n2, n3 = tax.labels_for(pred.code)
                        label = n3 or n2 or n1 or pred.code
                        with [col1, col2, col3][j]:
                            # Color based on score
                            if pred.score > 0.75:
                                color = "🟢"
                            elif pred.score > 0.6:
                                color = "🟡"
                            else:
                                color = "🔴"
                            if st.button(f"{color} {pred.code}\n{label}\nScore: {pred.score:.2f}", key=f"pred_{row.sheet_row_ix}_{j}"):
                                pred_choice = pred.code

                    # Custom code input
                    if st.checkbox("Enter custom code", key=f"custom_{row.sheet_row_ix}"):
                        pred_choice = st.selectbox("Select code:", [e.code for e in tax.all()], key=f"code_{row.sheet_row_ix}")

                    if pred_choice:
                        n1, n2, n3 = tax.labels_for(pred_choice)
                        bench = tax.get(pred_choice).benchmarking if tax.get(pred_choice) else ""
                        fields: dict[str, str] = {}
                        if n1: fields[NIVEAU_1_COL] = n1
                        if n2: fields[NIVEAU_2_COL] = n2
                        if n3: fields[NIVEAU_3_COL] = n3
                        if bench: fields[BENCHMARKING_COL] = bench
                        updates[row.sheet_row_ix] = fields
                        new_records.append(TrainingRecord.from_user(row.ydelse, pred_choice))
                        st.success(f"Selected: {pred_choice}")

                else:
                    st.warning("No predictions")

        if len(needs_pred) > 50:
            st.info(f"Showing first 50 of {len(needs_pred)} rows. Download and re-run to process remaining.")

    # Write back
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Changes to Excel"):
            if updates:
                write_back(tmp_path, updates, backup=True)
                if new_records:
                    store.append_many(new_records)
                    st.success(f"Saved {len(new_records)} new labels")
                st.balloons()
                with open(tmp_path, "rb") as f:
                    st.download_button(
                        label="Download Categorized File",
                        data=f.read(),
                        file_name=uploaded_file.name.replace(".xlsx", "_categorized.xlsx"),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            else:
                st.info("No changes to save")
    with col2:
        st.info(f"{len(updates)} rows ready to save")


def page_data():
    """Training data browser."""
    st.title("📊 Training Data")

    store = load_store()
    tax = load_taxonomy()
    deduped = store.deduped()

    from collections import Counter

    st.subheader("Statistics")
    c1, c2, c3 = st.columns(3)
    by_source = Counter(r.source for r in deduped)
    n3_codes = {r.code for r in deduped if len(r.code) == 5}
    with c1:
        st.metric("Ingest Records", by_source.get("ingest", 0))
    with c2:
        st.metric("User Records", by_source.get("user", 0))
    with c3:
        st.metric("Distinct N3 Codes", len(n3_codes))

    st.subheader("Browse Records")
    search = st.text_input("Search Ydelse:")
    filtered = [r for r in deduped if not search or search.lower() in r.text.lower()]

    st.info(f"Showing {len(filtered)}/{len(deduped)} records")

    for r in filtered[:100]:
        n1, n2, n3 = tax.labels_for(r.code)
        with st.expander(f"{r.code} — {r.text[:60]}"):
            st.markdown(f"""
            **Text:** {r.text}

            **Classification:**
            - N1: {n1}
            - N2: {n2}
            - N3: {n3}

            **Source:** {r.source} | **File:** {r.file}
            """)


def page_model():
    """Model stats."""
    st.title("🧠 Model Information")

    tax = load_taxonomy()
    store = load_store()
    clf, _ = load_classifier_and_texts()
    deduped = store.deduped()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Training Records", len(deduped))
    with c2:
        st.metric("Prototypes", len(clf.known_codes()))
    with c3:
        n3_codes = {r.code for r in deduped if len(r.code) == 5}
        coverage = len(n3_codes) / len(tax.at_level(3)) * 100
        st.metric("N3 Coverage", f"{coverage:.0f}%")

    st.divider()
    st.subheader("Class Coverage by Niveau-1")
    from collections import Counter
    by_n1 = Counter(r.code[0] for r in deduped)
    for letter in sorted(by_n1.keys()):
        entry = tax.get(letter)
        label = entry.label.split(" - ")[0] if entry else "?"
        st.progress(by_n1[letter] / 500, text=f"{letter}: {by_n1[letter]} records - {label}")


# =========================================================================
# Main
# =========================================================================

def main():
    with st.sidebar:
        st.title("Uniformat")
        page = option_menu(
            "Menu",
            ["Home", "Categorize", "Training Data", "Model Info"],
            icons=["house", "pencil", "book", "bar-chart"],
            menu_icon="menu-button-wide",
            default_index=0,
        )

    if page == "Home":
        page_home()
    elif page == "Categorize":
        page_categorize()
    elif page == "Training Data":
        page_data()
    elif page == "Model Info":
        page_model()


if __name__ == "__main__":
    main()
