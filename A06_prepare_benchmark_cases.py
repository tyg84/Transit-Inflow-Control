"""Copy one generated reference case to the benchmark input directories."""

from pathlib import Path
import shutil


SOURCE_CASE = Path("data/reference")
BENCHMARK_CASES = ("equity_efficiency_MLR", "BYO", "DE")


def prepare_benchmark_cases():
    if not SOURCE_CASE.is_dir():
        raise FileNotFoundError(
            "data/reference is missing; run A01 through A05 before this script."
        )

    for case_name in BENCHMARK_CASES:
        target = Path("data") / case_name
        target.mkdir(parents=True, exist_ok=True)
        for source_file in SOURCE_CASE.glob("*.csv"):
            shutil.copy2(source_file, target / source_file.name)
        print(f"Prepared {target} from {SOURCE_CASE}.")


if __name__ == "__main__":
    prepare_benchmark_cases()
