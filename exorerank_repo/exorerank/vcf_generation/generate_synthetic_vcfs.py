#!/usr/bin/env python3
"""Build deterministic, Exomiser-compatible synthetic benchmark VCFs.

Implemented background mode: ClinVar benign/likely-benign variants (optionally VUS).
The README's proposed 1000 Genomes background is NOT implemented here; requesting
--background-source 1000g fails clearly rather than silently changing methodology.
"""
from __future__ import annotations

import argparse, gzip, hashlib, json, random, re, sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

CLINVAR_URLS = {
    "GRCh37": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37/weekly/clinvar.vcf.gz",
    "GRCh38": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/clinvar.vcf.gz",
}

def parse_info(text):
    return {p.split("=", 1)[0]: p.split("=", 1)[1] for p in text.split(";") if "=" in p}

def clean_chromosome(chrom):
    return chrom.removeprefix("chr")

def clean_filename(value):
    value = re.sub(r'[^A-Za-z0-9._-]+', "_", value.strip())
    return re.sub(r"_+", "_", value).strip("_.") or "variant"

def significance(info):
    return {x.strip().replace(" ", "_") for x in re.split(r"[|,/]", info.get("CLNSIG", "")) if x}

def gene_from_info(info):
    return info.get("GENEINFO", "").split("|")[0].split(":")[0].strip()

def is_pathogenic(info, include_likely):
    sig = significance(info)
    return ("Pathogenic" in sig or "Pathogenic/Likely_pathogenic" in sig
            or (include_likely and "Likely_pathogenic" in sig)) and not ({"Benign", "Likely_benign"} & sig)

def is_background(info, include_vus):
    sig = significance(info)
    return bool({"Benign", "Likely_benign"} & sig) or (include_vus and "Uncertain_significance" in sig)

def open_source(source):
    if source.startswith(("http://", "https://")):
        response = urlopen(source)
        return gzip.GzipFile(fileobj=response) if source.endswith(".gz") else response
    handle = open(source, "rb")
    return gzip.GzipFile(fileobj=handle) if source.endswith(".gz") else handle

def iter_records(source):
    with open_source(source) as stream:
        for raw in stream:
            line = raw.decode("utf-8").rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            chrom, pos, vid, ref, alt, _, _, info_text = parts[:8]
            if "," in alt:  # keep simple biallelic records for unambiguous GT/AD
                continue
            try:
                pos = int(pos)
            except ValueError:
                continue
            info = parse_info(info_text)
            gene = gene_from_info(info)
            if not gene:
                continue
            yield {"chrom": clean_chromosome(chrom), "pos": pos, "id": vid or ".",
                   "ref": ref, "alt": alt, "info": info_text, "parsed_info": info, "gene": gene}

def reservoir_add(items, seen, item, limit, rng):
    seen += 1
    if len(items) < limit:
        items.append(item)
    else:
        index = rng.randrange(seen)
        if index < limit:
            items[index] = item
    return seen

def collect_catalogs(source, pathogenic_limit, background_limit, include_likely, include_vus, rng):
    pathogenic, background = [], []
    p_seen = b_seen = 0
    for record in iter_records(source):
        if is_pathogenic(record["parsed_info"], include_likely):
            p_seen = reservoir_add(pathogenic, p_seen, record, pathogenic_limit, rng)
        elif is_background(record["parsed_info"], include_vus):
            b_seen = reservoir_add(background, b_seen, record, background_limit, rng)
    if not pathogenic:
        raise ValueError("No pathogenic ClinVar records with a gene symbol were found.")
    if not background:
        raise ValueError("No ClinVar benign/likely-benign background records with a gene symbol were found.")
    return pathogenic, background

def variant_key(record):
    return record["chrom"], record["pos"], record["ref"], record["alt"]

def quality_fields(pathogenic, rng):
    if pathogenic:
        ref_depth, alt_depth = rng.randint(40, 80), rng.randint(35, 75)
        return "0/1", rng.randint(90, 99), ref_depth + alt_depth, ref_depth, alt_depth, rng.randint(200, 500), "PASS"
    tier = rng.choices(("low", "medium", "high"), weights=(60, 30, 10))[0]
    if tier == "low":
        gt, ref_depth, alt_depth, gq, qual, flt = rng.choice(("0/0", "0/1")), rng.randint(5, 15), rng.randint(0, 8), rng.randint(10, 30), rng.randint(5, 30), rng.choice(("LowQual", "LowDP", "LowGQ"))
    elif tier == "medium":
        gt, ref_depth, alt_depth, gq, qual, flt = rng.choice(("0/0", "0/1", "1/1")), rng.randint(15, 30), rng.randint(0, 25), rng.randint(30, 60), rng.randint(30, 100), rng.choice(("PASS", "PASS", "LowGQ"))
    else:
        gt, ref_depth, alt_depth, gq, qual, flt = rng.choice(("0/0", "0/1", "1/1")), rng.randint(25, 60), rng.randint(0, 45), rng.randint(60, 89), rng.randint(100, 200), "PASS"
    return gt, gq, ref_depth + alt_depth, ref_depth, alt_depth, qual, flt

def sort_key(record):
    chrom = record["chrom"]
    return (int(chrom) if chrom.isdigit() else {"X": 23, "Y": 24, "MT": 25, "M": 25}.get(chrom, 26), record["pos"])

def vcf_header(sample, build):
    return "\n".join((
        "##fileformat=VCFv4.2", "##source=ExoRerank_Stage1_ClinVarBackground",
        "##reference=" + build,
        '##FILTER=<ID=LowQual,Description="Low quality synthetic background call">',
        '##FILTER=<ID=LowDP,Description="Low depth synthetic background call">',
        '##FILTER=<ID=LowGQ,Description="Low genotype quality synthetic background call">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">',
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">',
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + sample,
    )) + "\n"

def write_case(outdir, case_number, pathogenic, background, build, num_variants, seed, source, sample_prefix):
    rng = random.Random(seed + case_number)
    candidates = [item for item in background if variant_key(item) != variant_key(pathogenic)]
    if len(candidates) < num_variants - 1:
        raise ValueError("Background catalog is smaller than --num-variants - 1 after removing the causal variant.")
    records = rng.sample(candidates, num_variants - 1) + [pathogenic]
    records.sort(key=sort_key)
    token = clean_filename(pathogenic["gene"])
    identifier = clean_filename(pathogenic["parsed_info"].get("ALLELEID", str(pathogenic["pos"])))
    stem = f"{token}_case{case_number:04d}_allele{identifier}"
    path = outdir / (stem + ".vcf")
    sample = f"{sample_prefix}{case_number:04d}"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(vcf_header(sample, build))
        for record in records:
            gt, gq, dp, ad_ref, ad_alt, qual, flt = quality_fields(record is pathogenic, rng)
            handle.write(f'{record["chrom"]}\t{record["pos"]}\t{record["id"]}\t{record["ref"]}\t{record["alt"]}\t{qual}\t{flt}\t{record["info"]}\tGT:GQ:DP:AD\t{gt}:{gq}:{dp}:{ad_ref},{ad_alt}\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1, "case_number": case_number, "vcf": path.name, "sha256": digest,
        "seed": seed, "case_seed": seed + case_number, "reference_build": build,
        "background_source": "clinvar", "background_source_note": "ClinVar benign/likely-benign variants; this is not 1000 Genomes.",
        "clinvar_source": source, "causal_variant": {k: pathogenic[k] for k in ("gene", "chrom", "pos", "id", "ref", "alt", "info")},
        "variant_count": len(records), "sample": sample, "chromosome_prefix_removed": True,
        "filename_contract": "first underscore-delimited token is the causal gene symbol",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (outdir / (stem + ".metadata.json")).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", choices=sorted(CLINVAR_URLS), default="GRCh38")
    parser.add_argument("--background-source", choices=("clinvar", "1000g"), default="clinvar")
    parser.add_argument("--clinvar-vcf", help="Local .vcf/.vcf.gz or URL; defaults to the current ClinVar weekly VCF for --build.")
    parser.add_argument("--output-dir", "--outdir", dest="output_dir", default="synthetic_vcfs")
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--num-variants", type=int, default=50000)
    parser.add_argument("--pathogenic-catalog-size", type=int, default=500)
    parser.add_argument("--background-catalog-size", type=int, default=200000)
    parser.add_argument("--include-likely", action="store_true")
    parser.add_argument("--include-vus", action="store_true")
    parser.add_argument("--seed", type=int, default=1, help="Deterministic seed (default: 1).")
    parser.add_argument("--sample-prefix", default="SYNTH")
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    if args.background_source == "1000g":
        raise SystemExit("1000 Genomes background is not implemented in this repository. Use --background-source clinvar (documented mode) or provide a 1000 Genomes adapter before claiming that methodology.")
    if min(args.num_cases, args.num_variants, args.pathogenic_catalog_size, args.background_catalog_size) < 1:
        raise SystemExit("Case, variant, and catalog sizes must be positive.")
    source = args.clinvar_vcf or CLINVAR_URLS[args.build]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    catalog_rng = random.Random(args.seed)
    pathogenic, background = collect_catalogs(source, args.pathogenic_catalog_size, args.background_catalog_size, args.include_likely, args.include_vus, catalog_rng)
    cases = [write_case(outdir, number, catalog_rng.choice(pathogenic), background, args.build, args.num_variants, args.seed, source, args.sample_prefix) for number in range(1, args.num_cases + 1)]
    manifest = {"schema_version": 1, "seed": args.seed, "reference_build": args.build, "background_source": "clinvar", "background_source_note": "ClinVar benign/likely-benign variants (not 1000 Genomes).", "clinvar_source": source, "catalog_counts": {"pathogenic": len(pathogenic), "background": len(background)}, "cases": cases, "created_utc": datetime.now(timezone.utc).isoformat()}
    (outdir / "run_metadata.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Generated {len(cases)} VCF(s) in {outdir} using ClinVar background; run_metadata.json records provenance.")
if __name__ == "__main__":
    main()
