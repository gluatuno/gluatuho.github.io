"""Parse HIV sample metadata out of sample/file IDs.

Naming scheme (confirmed with the data owner)::

    [r]<patient_id>[_<cohort_digit><week:03d>]_<cell_type>_<well>_<replicate>

- Optional leading ``r`` marks a resequencing of the same sample.
- ``patient_id`` identifies the patient (e.g. 3695, 1599).
- The optional 4-digit code before the cell type packs two fields: its
  first digit is a cohort marker (``2`` = HIV patient in this dataset) and
  the remaining 3 digits are the week of sampling since detection, e.g.
  ``2003`` -> week 3. A sample with no such code (e.g. ``1548_APC_3_33``)
  has no known week and is treated as an older/undated sample.
- ``cell_type`` is the sorted cell population, e.g. CD4, APC.
- The trailing numeric tokens (well/replicate) are kept verbatim as
  ``replicate`` since their exact meaning wasn't specified.

Examples::

    r1599_CD4_3_86          -> patient=1599, week=None, cell=CD4, resequenced=True
    3695_2003_APC_1_23      -> patient=3695, cohort=2, week=3, cell=APC
    1664_CD4_2_2            -> patient=1664, week=None, cell=CD4
"""

from dataclasses import dataclass

import pandas as pd

KNOWN_SUFFIXES = ("_extracted",)


@dataclass
class SampleMeta:
    sample_id: str
    patient_id: str
    resequenced: bool
    cohort_code: "int | None"
    week: "int | None"
    cell_type: str
    replicate: str


def parse_sample_id(sample_id: str) -> SampleMeta:
    stem = sample_id
    for suffix in KNOWN_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    tokens = stem.split("_")
    if not tokens or not tokens[0]:
        raise ValueError(f"Cannot parse sample id: {sample_id!r}")

    first = tokens[0]
    resequenced = first[:1].lower() == "r" and first[1:].isdigit()
    patient_id = first[1:] if resequenced else first
    if not patient_id.isdigit():
        raise ValueError(f"Cannot parse patient id out of {sample_id!r}")

    rest = tokens[1:]
    if not rest:
        raise ValueError(f"Missing cell type in sample id: {sample_id!r}")

    cohort_code = None
    week = None
    if rest[0].isdigit() and len(rest[0]) == 4:
        cohort_code = int(rest[0][0])
        week = int(rest[0][1:])
        rest = rest[1:]

    if not rest:
        raise ValueError(f"Missing cell type in sample id: {sample_id!r}")
    cell_type = rest[0]
    replicate = "_".join(rest[1:])

    return SampleMeta(
        sample_id=sample_id,
        patient_id=patient_id,
        resequenced=resequenced,
        cohort_code=cohort_code,
        week=week,
        cell_type=cell_type,
        replicate=replicate,
    )


def build_metadata_table(sample_ids) -> pd.DataFrame:
    rows = [parse_sample_id(s).__dict__ for s in sample_ids]
    return pd.DataFrame(rows).set_index("sample_id")
