# MedContext Precision Oncology — Label Schema

## Purpose

This document defines the ground-truth annotation schema used by the synthetic precision-oncology dataset.

Each row in `annotations.csv` represents one contextualized biomedical entity, referred to as a **Biomedical Fact**.

The MVP focuses on preserving the clinical context surrounding genes, variants, fusions, biomarkers, drugs, and diseases.

---

## 1. Entity Types

Supported entity types:

- `GENE`
- `VARIANT`
- `FUSION`
- `BIOMARKER`
- `DRUG`
- `DISEASE`

`DISEASE` may be detected and stored, but the main contextual-classification targets are `GENE`, `VARIANT`, `FUSION`, `BIOMARKER`, and `DRUG`.

---

## 2. Assertion

Describes whether the biomedical finding is stated as present, absent, or possible.

| Label | Meaning | Example |
|---|---|---|
| `PRESENT` | The finding is reported as present. | `FLT3-ITD was detected.` |
| `ABSENT` | The finding is explicitly reported as absent. | `No FLT3-ITD was identified.` |
| `POSSIBLE` | The finding may be present but is not confirmed. | `FLT3-ITD cannot be excluded.` |

### Important

Assertion and certainty are separate dimensions.

Example:

`An IDH1 mutation is suspected.`

- `assertion = POSSIBLE`
- `certainty = SUSPECTED`

---

## 3. Experiencer

Identifies who the biomedical finding refers to.

| Label | Meaning | Example |
|---|---|---|
| `PATIENT` | The finding applies to the patient. | `The patient has an NPM1 mutation.` |
| `FAMILY` | The finding applies to a family member. | `The patient's father carried a TP53 mutation.` |
| `OTHER` | The finding applies to another individual or external subject. | `The donor had no FLT3 mutation.` |

---

## 4. Temporality

Describes when the finding applies.

| Label | Meaning | Example |
|---|---|---|
| `CURRENT` | Current or most recent status. | `Current sequencing confirms FLT3-ITD.` |
| `HISTORICAL` | A past finding or previous state. | `FLT3-ITD was detected at diagnosis.` |
| `FUTURE` | Planned, pending, conditional, or future evaluation. | `Repeat FLT3 testing is planned.` |
| `UNKNOWN` | Timing cannot be reliably inferred. | `FLT3 status is mentioned in the record.` |

---

## 5. Certainty

Describes how strongly the report presents the finding.

| Label | Meaning | Example |
|---|---|---|
| `CONFIRMED` | The finding is presented as established or explicitly negative. | `Sequencing confirmed IDH1 p.R132H.` |
| `SUSPECTED` | The finding is explicitly suspected or considered possible. | `An IDH1 mutation is suspected.` |
| `UNKNOWN` | Certainty cannot be inferred. | `IDH1 mutation status was discussed.` |

For explicit negative findings such as:

`No ALK rearrangement was detected.`

use:

- `assertion = ABSENT`
- `certainty = CONFIRMED`

because the negative result itself is presented as confirmed.

---

## 6. Case Types

Each synthetic annotation may include a `case_type` field for analysis.

Supported values:

- `positive`
- `negative`
- `uncertain`
- `historical`
- `family`
- `conflicting`
- `hypothetical`
- `mixed`

### Examples

**positive**

`FLT3-ITD was detected.`

**negative**

`No FLT3-ITD was identified.`

**uncertain**

`FLT3-ITD cannot be excluded.`

**historical**

`FLT3-ITD was detected at diagnosis.`

**family**

`The patient's father carried a TP53 mutation.`

**conflicting**

`HER2 was previously positive, but current testing is negative.`

**hypothetical**

`If an EGFR mutation is detected, osimertinib may be considered.`

**mixed**

`EGFR p.L858R was detected, whereas ALK and ROS1 rearrangements were absent.`

`case_type` is metadata for evaluation and must not be used as a predictive feature.

---

## 7. Difficulty

Difficulty reflects linguistic complexity, not molecular complexity.

### `EASY`

One entity with an explicit context cue.

Example:

`No ALK rearrangement was detected.`

### `MEDIUM`

Requires interpretation of family context, uncertainty, or temporality.

Example:

`Previous sequencing identified FLT3-ITD.`

### `HARD`

Contains multiple entities, contrast, indirect language, or conflicting temporal states.

Example:

`Although FLT3-ITD had been reported at diagnosis, repeat sequencing did not detect the alteration.`

Difficulty is evaluation metadata and must not be used as a predictive feature.

---

## 8. annotations.csv Schema

Each row represents one contextualized biomedical entity.

| Field | Description |
|---|---|
| `annotation_id` | Unique annotation identifier. |
| `report_id` | Identifier of the source report. |
| `entity_text` | Exact surface form found in the report. |
| `normalized_entity` | Canonical entity from `oncology_vocabulary.yaml`. |
| `entity_type` | `GENE`, `VARIANT`, `FUSION`, `BIOMARKER`, `DRUG`, or `DISEASE`. |
| `assertion` | `PRESENT`, `ABSENT`, or `POSSIBLE`. |
| `experiencer` | `PATIENT`, `FAMILY`, or `OTHER`. |
| `temporality` | `CURRENT`, `HISTORICAL`, `FUTURE`, or `UNKNOWN`. |
| `certainty` | `CONFIRMED`, `SUSPECTED`, or `UNKNOWN`. |
| `value` | Optional biomarker or measurement value. |
| `unit` | Optional unit. |
| `start_char` | Character offset where the entity starts. |
| `end_char` | Character offset where the entity ends. |
| `source_sentence` | Sentence containing the entity. |
| `case_type` | Synthetic scenario category. |
| `difficulty` | `EASY`, `MEDIUM`, or `HARD`. |
| `template_family` | Linguistic template family used during synthetic generation. |

---

## 9. reports.csv Schema

Each row represents one synthetic precision-oncology report.

| Field | Description |
|---|---|
| `report_id` | Unique report identifier. |
| `disease` | Main oncology domain: `AML`, `APL`, `B_ALL`, or `NSCLC`. |
| `report_text` | Complete synthetic report text. |
| `num_entities` | Expected number of annotated biomedical entities. |

A `split` column (`train`, `validation`, `test`) will be added after dataset generation.

---

## 10. Annotation Principles

### One entity mention can create one Biomedical Fact

Example:

`FLT3-ITD was detected.`

produces one annotation.

### One sentence can create multiple Biomedical Facts

Example:

`FLT3-ITD was detected, while NPM1 mutation was not identified.`

produces:

1. `FLT3-ITD -> PRESENT`
2. `NPM1 mutation -> ABSENT`

### The same normalized entity may appear multiple times

Example:

`HER2 was positive on the previous biopsy, but current testing is negative.`

produces two facts:

1. `HER2 -> PRESENT / HISTORICAL`
2. `HER2 -> ABSENT / CURRENT`

### Family findings must not be assigned to the patient

Example:

`The patient's mother carried a pathogenic TP53 variant.`

produces:

- `experiencer = FAMILY`

### Drug-target metadata must not create inferred molecular facts

Example:

The presence of `gilteritinib` must not automatically imply that `FLT3` is mutated.

MedContext extracts facts explicitly supported by the source text.

---

## 11. Synthetic Dataset Evaluation Metadata

The fields below are used only for dataset analysis and evaluation:

- `case_type`
- `difficulty`
- `template_family`

They must never be provided to the contextual classifier as model features.

This prevents synthetic-data leakage and allows evaluation by scenario type and linguistic difficulty.

---

## 12. Dataset Split Principle

Synthetic rows must not be randomly split without considering linguistic templates.

Instead, reports should be split by `template_family` so that closely related sentence templates do not appear across both training and test sets.

Bad example:

**Train**

`No EGFR mutation was identified.`

**Test**

`No ALK mutation was identified.`

These sentences are structurally almost identical and can artificially inflate evaluation performance.

The final split should therefore prioritize **template-family separation** to provide a more realistic estimate of generalization.

---

## 13. Core MedContext Target

The central mapping learned by the MVP is:

```text
Biomedical entity
+
local clinical context
        ↓
Assertion
Experiencer
Temporality
Certainty
```

The output will later be compiled into the MedContext Clinical Fact Intermediate Representation (Clinical Fact IR).
