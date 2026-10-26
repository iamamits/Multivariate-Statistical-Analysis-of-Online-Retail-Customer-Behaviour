"""
Computational Statistics — Sequential Pipeline Runner
Coursework: Multivariate Statistical Analysis of Online Retail Customer Behaviour
Dataset: Online Retail II (N = 5,878 eligible customers)

This runner executes all five syllabus units sequentially in order:
  1. Unit 1: Multivariate Normal Distribution (MVN)
  2. Unit 2: Linear Discriminant Analysis (LDA)
  3. Unit 3: Principal Component Analysis (PCA)
  4. Unit 4: Factor Analysis (FA)
  5. Unit 5: K-Means Cluster Analysis

Pre-requisite:
  phase1_output/customer_master.csv must exist.
  (Phase 1 data cleaning pipeline is NOT run automatically).
"""

from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
CUSTOMER_MASTER = PROJECT_ROOT / "phase1_output" / "customer_master.csv"

# Canonical unit scripts in syllabus order
UNITS = [
    {
        "unit": 1,
        "name": "Multivariate Normal Distribution (MVN)",
        "script": PROJECT_ROOT / "unit1_mvn" / "unit1_mvn_analysis.py",
        "output_dir": PROJECT_ROOT / "unit1_mvn" / "outputs",
    },
    {
        "unit": 2,
        "name": "Linear Discriminant Analysis (LDA)",
        "script": PROJECT_ROOT / "unit2_discriminant_analysis" / "unit2_discriminant_analysis.py",
        "output_dir": PROJECT_ROOT / "unit2_discriminant_analysis" / "outputs",
    },
    {
        "unit": 3,
        "name": "Principal Component Analysis (PCA)",
        "script": PROJECT_ROOT / "unit3_pca" / "unit3_pca_analysis.py",
        "output_dir": PROJECT_ROOT / "unit3_pca" / "outputs",
    },
    {
        "unit": 4,
        "name": "Factor Analysis (FA)",
        "script": PROJECT_ROOT / "unit4_factor_analysis" / "unit4_factor_analysis.py",
        "output_dir": PROJECT_ROOT / "unit4_factor_analysis" / "outputs",
    },
    {
        "unit": 5,
        "name": "K-Means Cluster Analysis",
        "script": PROJECT_ROOT / "unit5_cluster_analysis" / "unit5_cluster_analysis.py",
        "output_dir": PROJECT_ROOT / "unit5_cluster_analysis" / "outputs",
    },
]


def check_prerequisites():
    """Verify that the prepared customer dataset exists before starting."""
    if not CUSTOMER_MASTER.exists():
        print(f"ERROR: Prepared customer dataset not found at:\n  {CUSTOMER_MASTER}")
        print("Please ensure Phase 1 has been executed and customer_master.csv is present.")
        sys.exit(1)

    # Verify that all canonical unit scripts exist
    for u in UNITS:
        if not u["script"].exists():
            print(f"ERROR: Missing canonical script for Unit {u['unit']} at:\n  {u['script']}")
            sys.exit(1)


def main():
    print("=" * 75)
    print("COMPUTATIONAL STATISTICS — FULL COURSEWORK PIPELINE RUNNER")
    print("=" * 75)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Input Dataset: {CUSTOMER_MASTER.relative_to(PROJECT_ROOT)}")
    print(f"Python Binary: {sys.executable}\n")

    check_prerequisites()

    total_start_time = time.time()
    successful_units = 0

    for u in UNITS:
        unit_num = u["unit"]
        unit_name = u["name"]
        script_path = u["script"]
        rel_script = script_path.relative_to(PROJECT_ROOT)

        print("\n" + "#" * 75)
        print(f"RUNNING UNIT {unit_num}: {unit_name}")
        print(f"Script: {rel_script}")
        print("#" * 75)

        start_time = time.time()

        # Execute unit script with current Python interpreter and working directory as project root
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"\n[FAILURE] Unit {unit_num} ({unit_name}) failed with exit code {result.returncode}.")
            print(f"Execution halted after {elapsed:.2f} seconds.")
            sys.exit(result.returncode)

        successful_units += 1
        print(f"\n[SUCCESS] Unit {unit_num} completed successfully in {elapsed:.2f} seconds.")

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 75)
    print("ALL 5 UNITS EXECUTED SUCCESSFULLY — PIPELINE COMPLETE")
    print("=" * 75)
    print(f"Completed {successful_units}/5 units in {total_elapsed:.2f} seconds.")
    print("All CSV and plot artifacts are updated in their respective unit outputs/ folders.")


if __name__ == "__main__":
    main()
