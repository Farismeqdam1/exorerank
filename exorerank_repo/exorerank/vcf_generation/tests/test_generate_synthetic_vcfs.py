import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "generate_synthetic_vcfs.py"
FIXTURE = Path(__file__).parent / "fixtures" / "clinvar_tiny.vcf"

class SyntheticVcfSmokeTests(unittest.TestCase):
    def run_generator(self, destination, seed=17):
        return subprocess.run([sys.executable, str(SCRIPT), "--clinvar-vcf", str(FIXTURE), "--output-dir", str(destination), "--num-cases", "1", "--num-variants", "3", "--pathogenic-catalog-size", "10", "--background-catalog-size", "10", "--seed", str(seed)], text=True, capture_output=True, check=True)

    def test_fixture_output_obeys_exomiser_and_filename_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.run_generator(output)
            vcf = next(output.glob("*.vcf"))
            self.assertIn(vcf.name.split("_", 1)[0], {"CFTR", "BRCA1"})
            lines = vcf.read_text().splitlines()
            self.assertTrue(any(line.startswith("#CHROM") and "GT:GQ:DP:AD" not in line for line in lines))
            records = [line.split("\t") for line in lines if not line.startswith("#")]
            self.assertEqual(len(records), 3)
            self.assertTrue(all(not item[0].startswith("chr") for item in records))
            self.assertTrue(all(item[8] == "GT:GQ:DP:AD" for item in records))
            metadata = json.loads(next(output.glob("*.metadata.json")).read_text())
            self.assertEqual(metadata["background_source"], "clinvar")
            self.assertIn("not 1000 Genomes", metadata["background_source_note"])

    def test_seed_makes_vcf_bytes_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "one", Path(temporary) / "two"
            self.run_generator(first, seed=9)
            self.run_generator(second, seed=9)
            self.assertEqual(next(first.glob("*.vcf")).read_bytes(), next(second.glob("*.vcf")).read_bytes())

    def test_unimplemented_1000_genomes_mode_is_explicit(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--background-source", "1000g"], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not implemented", result.stderr)

if __name__ == "__main__":
    unittest.main()
