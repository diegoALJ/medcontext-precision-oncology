from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from src.clinical_fact_compiler import ClinicalFactCompiler
from src.medcontext_packet import MedContextPacketBuilder


# ============================================================
# CONFIG
# ============================================================

VOCAB_PATH = (
    ROOT
    / "data"
    / "vocabulary"
    / "oncology_vocabulary.yaml"
)

MODEL_DIR = (
    ROOT
    / "results"
    / "models"
)


EXAMPLES = {
    "Uncertain IDH1 variant": (
        "IDH1 p.R132H cannot be excluded."
    ),
    "Family TP53 finding": (
        "The patient's father carried TP53 p.R175H."
    ),
    "Conflicting PML::RARA finding": (
        "PML::RARA was previously detected, "
        "but current testing was negative for PML::RARA."
    ),
    "Mixed oncology report": (
        "IDH1 p.R132H cannot be excluded. "
        "The patient's father carried TP53 p.R175H. "
        "PML::RARA was previously detected, "
        "but current testing was negative for PML::RARA."
    ),
}


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_pipeline():
    """
    Load BioLORD, XGBoost models, vocabulary, and all pipeline
    components once per Streamlit server process.
    """
    compiler = ClinicalFactCompiler(
        vocabulary_path=VOCAB_PATH,
        model_dir=MODEL_DIR,
    )

    packet_builder = MedContextPacketBuilder(
        compiler
    )

    return compiler, packet_builder


# ============================================================
# PRESENTATION HELPERS
# ============================================================

def facts_to_dataframe(
    facts: list[dict],
) -> pd.DataFrame:
    """
    Convert compiled Clinical Facts into a compact table
    for presentation in Streamlit.
    """
    rows = []

    for fact in facts:
        rows.append(
            {
                "Entity": fact["entity"],
                "Type": fact["entity_type"],
                "Assertion": fact["assertion"],
                "Experiencer": fact["experiencer"],
                "Temporality": fact["temporality"],
                "Certainty": fact["certainty"],
                "Confidence": fact["confidence"],
                "Source": fact["source_text"],
            }
        )

    return pd.DataFrame(
        rows
    )


def render_fact(
    fact: dict,
    index: int,
) -> None:
    """
    Render one detailed Clinical Fact inside an expandable
    Streamlit component.
    """
    title = (
        f"{index + 1}. "
        f"{fact['normalized_entity']} "
        f"— {fact['assertion']} / "
        f"{fact['temporality']}"
    )

    with st.expander(
        title,
        expanded=False,
    ):
        col1, col2 = st.columns(
            2
        )

        # ----------------------------------------------------
        # GENERAL CONTEXT
        # ----------------------------------------------------

        with col1:

            st.write(
                "**Entity type:**",
                fact["entity_type"],
            )

            st.write(
                "**Experiencer:**",
                fact["experiencer"],
            )

            st.write(
                "**Certainty:**",
                fact["certainty"],
            )

            st.write(
                "**Overall confidence:**",
                fact["confidence"],
            )

        # ----------------------------------------------------
        # MODEL DETAILS
        # ----------------------------------------------------

        with col2:

            st.write(
                "**Span:**",
                (
                    f"{fact['start_char']}:"
                    f"{fact['end_char']}"
                ),
            )

            st.write(
                "**Assertion confidence:**",
                fact[
                    "target_confidence"
                ]["assertion"],
            )

            st.write(
                "**Experiencer confidence:**",
                fact[
                    "target_confidence"
                ]["experiencer"],
            )

            st.write(
                "**Temporality confidence:**",
                fact[
                    "target_confidence"
                ]["temporality"],
            )

            st.write(
                "**Certainty confidence:**",
                fact[
                    "target_confidence"
                ]["certainty"],
            )

        # ----------------------------------------------------
        # SOURCE CONTEXT
        # ----------------------------------------------------

        st.write(
            "**Source context**"
        )

        st.code(
            fact["source_text"],
            language=None,
        )

        # ----------------------------------------------------
        # MOLECULAR REPRESENTATION
        # ----------------------------------------------------

        st.write(
            "**Molecular representation**"
        )

        st.json(
            fact["molecular"]
        )


# ============================================================
# APP
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # PAGE CONFIGURATION
    # --------------------------------------------------------

    st.set_page_config(
        page_title=(
            "MedContext Precision Oncology"
        ),
        page_icon="🧬",
        layout="wide",
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title(
        "MedContext Precision Oncology"
    )

    st.caption(
        "Neuro-symbolic clinical fact compilation "
        "for precision-oncology reports."
    )

    st.markdown(
        """
        Paste a precision-oncology report, load an example,
        or upload a `.txt` file.

        MedContext detects biomedical entities, parses their
        molecular structure, contextualizes each finding,
        and transforms the report into structured Clinical Facts.
        """
    )

    compiler, packet_builder = load_pipeline()

    # ========================================================
    # 1. INPUT REPORT
    # ========================================================

    st.subheader(
        "1. Input report"
    )

    example_name = st.selectbox(
        "Load example",
        options=[
            "Custom report",
            *EXAMPLES.keys(),
        ],
    )

    default_text = ""

    if example_name != "Custom report":

        default_text = EXAMPLES[
            example_name
        ]

    uploaded_file = st.file_uploader(
        "Optional: upload a .txt report",
        type=[
            "txt"
        ],
    )

    if uploaded_file is not None:

        default_text = (
            uploaded_file
            .read()
            .decode(
                "utf-8"
            )
        )

    report_text = st.text_area(
        "Precision-oncology report",
        value=default_text,
        height=220,
        placeholder=(
            "Example: EGFR p.L858R was detected. "
            "No ALK rearrangement was identified."
        ),
    )

    report_id = st.text_input(
        "Optional report ID",
        value="DEMO_001",
    )

    compile_clicked = st.button(
        "Compile Clinical Facts",
        type="primary",
    )

    # ========================================================
    # INFERENCE
    # ========================================================

    if not compile_clicked:
        return

    if not report_text.strip():

        st.warning(
            "Enter a report before running MedContext."
        )

        return

    with st.spinner(
        "Compiling clinical facts..."
    ):

        # ----------------------------------------------------
        # Run the full MedContext pipeline only once
        # ----------------------------------------------------

        facts = compiler.compile(
            report_text.strip()
        )

        # ----------------------------------------------------
        # Full Audit Packet
        # ----------------------------------------------------

        packet = packet_builder.build_packet(
            report_text=report_text.strip(),
            facts=facts,
            report_id=(
                report_id.strip()
                or None
            ),
        )

        # ----------------------------------------------------
        # Compact AI Packet
        # ----------------------------------------------------

        compact_packet = (
            packet_builder
            .build_compact_packet(
                facts=facts,
                report_id=(
                    report_id.strip()
                    or None
                ),
            )
        )

    summary = packet[
        "summary"
    ]

    # --------------------------------------------------------
    # NO ENTITIES
    # --------------------------------------------------------

    if not facts:

        st.info(
            "No supported biomedical entities "
            "were detected."
        )

        return

    # ========================================================
    # 2. REPORT SUMMARY
    # ========================================================

    st.subheader(
        "2. Report summary"
    )

    col1, col2, col3, col4 = (
        st.columns(
            4
        )
    )

    col1.metric(
        "Clinical facts",
        summary[
            "fact_count"
        ],
    )

    col2.metric(
        "Unique entities",
        summary[
            "unique_entity_count"
        ],
    )

    col3.metric(
        "Present findings",
        summary[
            "assertion_counts"
        ].get(
            "PRESENT",
            0,
        ),
    )

    col4.metric(
        "Current findings",
        summary[
            "temporality_counts"
        ].get(
            "CURRENT",
            0,
        ),
    )

    # ========================================================
    # 3. CLINICAL FACTS TABLE
    # ========================================================

    st.subheader(
        "3. Clinical Facts"
    )

    facts_df = facts_to_dataframe(
        facts
    )

    st.dataframe(
        facts_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # 4. DETAILED INTERPRETATION
    # ========================================================

    st.subheader(
        "4. Detailed interpretation"
    )

    for index, fact in enumerate(
        facts
    ):

        render_fact(
            fact,
            index,
        )

    # ========================================================
    # 5. MEDCONTEXT PACKETS
    # ========================================================

    st.subheader(
        "5. MedContext Packets"
    )

    full_tab, compact_tab = st.tabs(
        [
            "Full Audit Packet",
            "Compact AI Packet",
        ]
    )

    # --------------------------------------------------------
    # FULL AUDIT PACKET
    # --------------------------------------------------------

    with full_tab:

        st.caption(
            "Complete traceable representation "
            "for auditing, debugging, UI display, "
            "and human inspection."
        )

        st.json(
            packet
        )

        full_packet_json = json.dumps(
            packet,
            indent=2,
            ensure_ascii=False,
        )

        st.download_button(
            "Download Full Packet",
            data=full_packet_json,
            file_name=(
                f"{report_id or 'medcontext'}"
                "_full_packet.json"
            ),
            mime="application/json",
            key="download_full_packet",
        )

    # --------------------------------------------------------
    # COMPACT AI PACKET
    # --------------------------------------------------------

    with compact_tab:

        st.caption(
            "Compact representation designed for "
            "LLMs, RAG systems, agents, APIs, "
            "and other downstream AI workflows."
        )

        st.json(
            compact_packet
        )

        compact_packet_json = json.dumps(
            compact_packet,
            indent=2,
            ensure_ascii=False,
        )

        st.download_button(
            "Download Compact AI Packet",
            data=compact_packet_json,
            file_name=(
                f"{report_id or 'medcontext'}"
                "_compact_packet.json"
            ),
            mime="application/json",
            key="download_compact_packet",
        )

    # ========================================================
    # FOOTER
    # ========================================================

    st.divider()

    st.caption(
        "Research/demo prototype. Outputs are not "
        "intended for clinical decision-making."
    )


if __name__ == "__main__":
    main()