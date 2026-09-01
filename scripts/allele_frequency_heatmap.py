#!/usr/bin/env python3
"""Plot an allele-frequency heatmap for one gene or the whole genome.

Expects a variant table (.tsv/.csv/.xlsx) with annotation columns
``variant, chrom, pos, ref, alt, gene, effect`` followed by one column per
sample holding that sample's allele frequency for each variant (blank/NaN
where the sample has no value). Rows are variants; columns are samples.

Usage:
    python allele_frequency_heatmap.py variant_matrix_af.tsv --gene gag
    python allele_frequency_heatmap.py variant_matrix_af.tsv --gene whole_genome
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


def select_matrix(df: pd.DataFrame, gene: str) -> pd.DataFrame:
    """Return a variants x samples allele-frequency matrix.

    gene="whole_genome" keeps every variant; any other value filters rows
    to that gene (case-insensitive). Samples with no frequency value
    anywhere in the selected rows are dropped from the result.
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
    matrix = subset.set_index("variant")[sample_cols].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.dropna(axis=1, how="all")  # drop samples with no values for this selection
    return matrix


def plot_heatmap(matrix: pd.DataFrame, title: str, output: str, show_variant_labels=None) -> None:
    n_variants, n_samples = matrix.shape
    if show_variant_labels is None:
        show_variant_labels = n_variants <= 120

    fig_w = min(max(8, 0.28 * n_samples), 40)
    fig_h = min(max(6, 0.18 * n_variants), 40)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        matrix,
        cmap="viridis",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Allele frequency"},
        yticklabels=show_variant_labels,
        xticklabels=True,
        mask=matrix.isna(),
        ax=ax,
    )
    ax.set_facecolor("#dddddd")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Variant")
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
    parser.add_argument("--output", default=None, help="Output image path (default: heatmap_<gene>.png)")
    parser.add_argument(
        "--show-labels",
        dest="show_labels",
        action="store_true",
        default=None,
        help="Force variant row labels on even for large heatmaps",
    )
    args = parser.parse_args()

    df = load_variant_table(args.file)
    matrix = select_matrix(df, args.gene)

    if matrix.shape[1] == 0:
        raise SystemExit(f"No samples have allele-frequency values for gene={args.gene!r}")

    label = args.gene if args.gene.lower() != "whole_genome" else "whole genome"
    output = args.output or f"heatmap_{args.gene.lower()}.png"
    plot_heatmap(matrix, f"Allele frequency — {label}", output, show_variant_labels=args.show_labels)


if __name__ == "__main__":
    main()
