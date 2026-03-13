from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = ROOT / "tests" / "baselines"
MANIFEST_PATH = ROOT / "tests" / "baseline_manifest.json"


@pytest.fixture(scope="session")
def project_root():
    return ROOT


@pytest.fixture(scope="session")
def baselines_dir():
    return BASELINES_DIR


@pytest.fixture(scope="session")
def manifest_path():
    return MANIFEST_PATH
