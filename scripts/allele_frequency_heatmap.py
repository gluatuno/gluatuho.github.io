#!/usr/bin/env python3
"""Plot an allele-frequency heatmap for one gene or the whole genome.

Expects a variant table (.tsv/.csv/.xlsx) with annotation columns
``variant, chrom, pos, ref, alt, gene, effect`` followed by one column per
sample holding that sample's allele frequency for each variant (blank/NaN
where the sample has no value). Rows are variants; columns are samples.

Sample columns are matched against sample_metadata.parse_sample_id, so the
--patient/--cell/--week/--resequenced filters below can narrow the heatmap
to a within-patient, between-patient, between-cell-type, or cross-week
comparison.

Usage:
    python allele_frequency_heatmap.py variant_matrix_af.tsv --gene gag
    python allele_frequency_heatmap.py variant_matrix_af.tsv --gene whole_genome
    python allele_frequency_heatmap.py variant_matrix_af.tsv --patient 3695           # one patient, all its samples
    python allele_frequency_heatmap.py variant_matrix_af.tsv --patient 3695,1599      # between two patients
    python allele_frequency_heatmap.py variant_matrix_af.tsv --patient 3695 --cell CD4,APC   # between cell types
    python allele_frequency_heatmap.py variant_matrix_af.tsv --patient 3695 --cell CD4 --week 1,2,3   # across weeks
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sample_metadata import parse_sample_id

ANNOTATION_COLS = ["variant", "chrom", "pos", "ref", "alt", "gene", "effect"]


def load_variant_table(path: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep="\t")
    missing = [c for c in ANNOTATION_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing expected annotation column(s): {missing}")
    return df


def filter_sample_columns(sample_cols, patients=None, cells=None, weeks=None, resequenced=None):
    """Narrow sample columns to those matching the given metadata filters.

    Each of patients/cells is a set of strings compared case-insensitively;
    weeks is a set of ints. A sample whose id can't be parsed by
    sample_metadata.parse_sample_id is dropped with a warning rather than
    crashing the whole run. None for a filter means "don't restrict on it".
    """
    kept = []
    for sample in sample_cols:
        try:
            meta = parse_sample_id(sample)
        except ValueError as exc:
            print(f"Warning: skipping sample {sample!r} ({exc})")
            continue
        if patients is not None and meta.patient_id not in patients:
            continue
        if cells is not None and meta.cell_type.lower() not in cells:
            continue
        if weeks is not None and meta.week not in weeks:
            continue
        if resequenced is not None and meta.resequenced != resequenced:
            continue
        kept.append(sample)
    return kept


def select_matrix(df: pd.DataFrame, gene: str, patients=None, cells=None, weeks=None, resequenced=None):
    """Return (matrix, table) for the selected gene/whole-genome + sample filters.

    gene="whole_genome" keeps every variant; any other value filters rows
    to that gene (case-insensitive). patients/cells/weeks/resequenced (see
    filter_sample_columns) narrow which sample columns are considered.
    Samples with no frequency value anywhere in the selected rows are
    dropped from the result.

    ``matrix`` is a variants (indexed by pos) x samples numeric frame, for
    plotting. ``table`` is the same data with the 7 annotation columns
    (variant, chrom, pos, ref, alt, gene, effect) followed by the kept
    sample columns, for writing out alongside the heatmap.
    """
    if gene.lower() != "whole_genome":
        available = sorted(df["gene"].dropna().unique())
        match = [g for g in available if g.lower() == gene.lower()]
        if not match:
            raise ValueError(f"Gene {gene!r} not found. Available genes: {available}")
        subset = df[df["gene"] == match[0]]
    else:
        subset = df

    sample_cols = [c for c in df.columns if c not in ANNOTATION_COLS]
    sample_cols = filter_sample_columns(sample_cols, patients=patients, cells=cells, weeks=weeks, resequenced=resequenced)
    if not sample_cols:
        raise ValueError("No sample columns matched the given --patient/--cell/--week/--resequenced filters")

    numeric = subset[sample_cols].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")  # drop samples with no values for this selection
    kept_sample_cols = list(numeric.columns)

    table = pd.concat([subset[ANNOTATION_COLS], numeric], axis=1)
    matrix = table.set_index("pos")[kept_sample_cols]
    return matrix, table


def plot_heatmap(matrix: pd.DataFrame, title: str, output: str, show_variant_labels=None) -> None:
    n_variants, n_samples = matrix.shape
    if show_variant_labels is None:
        show_variant_labels = n_variants <= 120

    # Samples on the y-axis, variant positions on the x-axis.
    plot_matrix = matrix.T

    fig_w = min(max(8, 0.18 * n_variants), 40)
    fig_h = min(max(6, 0.28 * n_samples), 40)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        plot_matrix,
        cmap="RdBu_r",  # blue = low allele frequency, red = high allele frequency
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Allele frequency"},
        xticklabels=show_variant_labels,
        yticklabels=True,
        mask=plot_matrix.isna(),
        ax=ax,
    )
    ax.set_facecolor("#dddddd")
    ax.set_xlabel("Variant position")
    ax.set_ylabel("Sample")
    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved heatmap ({n_variants} variants x {n_samples} samples) to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="Path to the variant allele-frequency table (.tsv/.csv/.xlsx)")
    parser.add_argument(
        "--gene",
        default="whole_genome",
        help='Gene to plot (e.g. "gag"), or "whole_genome" for all variants (default: whole_genome)',
    )
    parser.add_argument("--output", default=None, help="Output image path (default: heatmap_<gene>[_<filters>].png)")
    parser.add_argument(
        "--table",
        default=None,
        help="Output TSV path for the table behind the heatmap (default: same name as --output with .tsv)",
    )
    parser.add_argument(
        "--show-labels",
        dest="show_labels",
        action="store_true",
        default=None,
        help="Force variant position labels on even for large heatmaps",
    )
    parser.add_argument("--patient", default=None, help="Comma-separated patient id(s) to include, e.g. 3695 or 3695,1599")
    parser.add_argument("--cell", default=None, help="Comma-separated cell type(s) to include, e.g. CD4 or CD4,APC")
    parser.add_argument("--week", default=None, help="Comma-separated week number(s) to include, e.g. 1,2,3")
    resequenced_group = parser.add_mutually_exclusive_group()
    resequenced_group.add_argument(
        "--resequenced-only", action="store_true", help="Only include resequenced (r-prefixed) samples"
    )
    resequenced_group.add_argument(
        "--exclude-resequenced", action="store_true", help="Exclude resequenced (r-prefixed) samples"
    )
    args = parser.parse_args()

    patients = set(args.patient.split(",")) if args.patient else None
    cells = {c.lower() for c in args.cell.split(",")} if args.cell else None
    weeks = {int(w) for w in args.week.split(",")} if args.week else None
    resequenced = True if args.resequenced_only else (False if args.exclude_resequenced else None)

    df = load_variant_table(args.file)
    matrix, table = select_matrix(df, args.gene, patients=patients, cells=cells, weeks=weeks, resequenced=resequenced)

    if matrix.shape[1] == 0:
        raise SystemExit(f"No samples have allele-frequency values for gene={args.gene!r} with the given filters")

    label_parts = [args.gene if args.gene.lower() != "whole_genome" else "whole genome"]
    if args.patient:
        label_parts.append(f"patient {args.patient}")
    if args.cell:
        label_parts.append(f"cell {args.cell}")
    if args.week:
        label_parts.append(f"week {args.week}")
    title = "Allele frequency — " + ", ".join(label_parts)

    if args.output:
        output = args.output
    else:
        suffix_parts = [args.gene.lower()]
        if args.patient:
            suffix_parts.append(f"p{args.patient.replace(',', '-')}")
        if args.cell:
            suffix_parts.append(f"c{args.cell.replace(',', '-')}")
        if args.week:
            suffix_parts.append(f"w{args.week.replace(',', '-')}")
        output = "heatmap_" + "_".join(suffix_parts) + ".png"

    plot_heatmap(matrix, title, output, show_variant_labels=args.show_labels)

    table_output = args.table or str(Path(output).with_suffix(".tsv"))
    table.to_csv(table_output, sep="\t", index=False)
    print(f"Saved table ({table.shape[0]} variants x {table.shape[1] - len(ANNOTATION_COLS)} samples) to {table_output}")


if __name__ == "__main__":
    main()
