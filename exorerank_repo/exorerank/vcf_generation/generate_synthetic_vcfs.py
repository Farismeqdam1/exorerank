#!/usr/bin/env python3
"""
generate_synthetic_vcfs.py  —  PLACEHOLDER
==========================================
>>> ADD YOUR VCF-GENERATION SCRIPT HERE <<<

This is the Stage-1 step of the pipeline: build synthetic test VCFs by spiking
ClinVar pathogenic variants into 1000 Genomes background exomes, producing one
case per known causative variant with a matched HPO phenotype profile.

The downstream re-ranking code (../pipeline/) expects, for each case:
  • one VCF file containing background variation + one spiked ClinVar variant
  • the HPO term set for that case's disease (passed to Exomiser)
  • a filename whose FIRST underscore-delimited token is the causative gene
    symbol, e.g.  ABCA4_Severe-retinal-dystrophy.vcf
    (the analysis scripts extract the target gene from this token)

Document here, for reproducibility (Bioinformatics requirement):
  - 1000 Genomes sample(s)/superpopulation used, and reference build (GRCh37/38)
  - ClinVar release date and filters (e.g. review status >= 2 stars; P/LP only)
  - spike-in method / tool, and how zygosity was set (dominant / recessive /
    compound-heterozygous)
  - how HPO terms were assigned per case (e.g. via gene->disease->HPO annotations)
  - total number of synthetic samples generated

Once you paste your script here, keep the output filename convention above so the
rest of the pipeline works unchanged.
"""

if __name__ == "__main__":
    raise SystemExit(
        "This is a placeholder. Paste your VCF-generation script into "
        "vcf_generation/generate_synthetic_vcfs.py"
    )
