"""End-to-end runner: ingest -> score -> report. One command, in order."""
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest import ingest  # noqa: E402
from src.analyze import run as analyze_run  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("=== Stage 1: ingest ===")
    ingest()
    print("\n=== Stage 2: score ===")
    analyze_run()
    print("\n=== Stage 3: build static report ===")
    subprocess.run(
        [sys.executable, str(Path(__file__).parent / "03_report.py")], check=True
    )
    print("\n=== All done. ===")
    print("Open output/report.html, or run:  streamlit run dashboard/app.py")
