#!/usr/bin/env python3
"""Build a variant x sample allele-frequency matrix from per-sample VCF files.

Scans a directory of lofreq/SnpEff-style VCFs (one file per sample), each
holding one row per variant with INFO containing AF=<float> and, for
SnpEff-annotated files, ANN=<snpeff annotation string>. Not every file needs
annotation: gene/effect for a variant are filled in from whichever file
happens to carry an ANN for that position. Produces:

- a wide matrix TSV: variant, chrom, pos, ref, alt, gene, effect, <sample columns...>
  (same shape consumed by allele_frequency_heatmap.py)
- a sample metadata TSV (patient, week, cell type, resequenced, replicate)
  parsed from each file's name via sample_metadata.parse_sample_id.

Usage:
    python build_variant_matrix.py vcf_dir/ --out-matrix variant_matrix_af.tsv --out-metadata samples_metadata.tsv
"""

import argparse
import glob
import os

import pandas as pd

from sample_metadata import KNOWN_SUFFIXES, build_metadata_table


def _sample_id_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    for suffix in KNOWN_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _parse_info(info: str) -> dict:
    fields = {}
    for item in info.split(";"):
        if "=" in item:
            key, _, value = item.partition("=")
            fields[key] = value
        else:
            fields[item] = True
    return fields


def _first_ann(ann_value: str):
    """Return (gene, effect) from the first ANN entry, or (None, None)."""
    first = ann_value.split(",")[0]
    parts = first.split("|")
    if len(parts) < 4:
        return None, None
    effect = parts[1] or None
    gene = parts[3] or None
    return gene, effect


def parse_vcf(path: str, pass_only: bool = True):
    """Yield (chrom, pos, ref, alt, af, gene, effect) for each variant record."""
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            chrom, pos, _id, ref, alt, _qual, filt, info = fields[:8]
            if pass_only and filt not in ("PASS", "."):
                continue
            info_fields = _parse_info(info)
            af = float(info_fields.get("AF", "nan"))
            gene = effect = None
            if "ANN" in info_fields:
                gene, effect = _first_ann(info_fields["ANN"])
            yield chrom, int(pos), ref, alt, af, gene, effect


def build_matrix(vcf_paths, pass_only: bool = True):
    variant_chrom_pos_ref_alt = {}
    variant_gene_effect = {}  # variant_key -> (gene, effect), filled from any annotated file
    af_by_variant_sample = {}  # variant_key -> {sample_id: af}
    sample_ids = []

    for path in vcf_paths:
        sample_id = _sample_id_from_path(path)
        sample_ids.append(sample_id)
        for chrom, pos, ref, alt, af, gene, effect in parse_vcf(path, pass_only=pass_only):
            variant_key = f"{chrom}:{pos}:{ref}>{alt}"
            variant_chrom_pos_ref_alt[variant_key] = (chrom, pos, ref, alt)
            if gene and variant_key not in variant_gene_effect:
                variant_gene_effect[variant_key] = (gene, effect)
            af_by_variant_sample.setdefault(variant_key, {})[sample_id] = af

    rows = []
    for variant_key, (chrom, pos, ref, alt) in sorted(
        variant_chrom_pos_ref_alt.items(), key=lambda kv: (kv[1][0], kv[1][1])
    ):
        gene, effect = variant_gene_effect.get(variant_key, (None, None))
        row = {
            "variant": variant_key,
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "gene": gene,
            "effect": effect,
        }
        row.update(af_by_variant_sample[variant_key])
        rows.append(row)

    sample_cols = sorted(set(sample_ids))
    matrix = pd.DataFrame(rows, columns=["variant", "chrom", "pos", "ref", "alt", "gene", "effect"] + sample_cols)
    return matrix, sample_cols


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vcf_dir", help="Directory containing per-sample .vcf files")
    parser.add_argument("--out-matrix", default="variant_matrix_af.tsv", help="Output matrix TSV path")
    parser.add_argument("--out-metadata", default="samples_metadata.tsv", help="Output sample metadata TSV path")
    parser.add_argument(
        "--include-non-pass",
        action="store_true",
        help="Include variants that did not pass filters (default: PASS only)",
    )
    args = parser.parse_args()

    vcf_paths = sorted(glob.glob(os.path.join(args.vcf_dir, "*.vcf")))
    if not vcf_paths:
        raise SystemExit(f"No .vcf files found in {args.vcf_dir!r}")

    matrix, sample_ids = build_matrix(vcf_paths, pass_only=not args.include_non_pass)
    matrix.to_csv(args.out_matrix, sep="\t", index=False)
    print(f"Wrote {len(matrix)} variants x {len(sample_ids)} samples to {args.out_matrix}")

    metadata = build_metadata_table(sample_ids)
    metadata.to_csv(args.out_metadata, sep="\t")
    print(f"Wrote metadata for {len(metadata)} samples to {args.out_metadata}")


if __name__ == "__main__":
    main()
