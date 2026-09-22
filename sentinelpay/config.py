from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
RANDOM_SEED = 42

for directory in (DATA_DIR, ARTIFACT_DIR, REPORT_DIR):
    directory.mkdir(exist_ok=True)
