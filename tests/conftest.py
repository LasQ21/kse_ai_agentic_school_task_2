"""Джерела артефактів для тестів сервісу: синтетичний зразок (завжди) і справжні data/ та models/
(лише якщо вони є локально). Тести, що беруть фікстуру `artifacts`, проганяються на кожному з них."""
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from src.data import PROCESSED_DIR, RAW_DIR, load_vadimkin  # noqa: E402
from src.service import SERVICE_MODEL, load_service_model  # noqa: E402


@dataclass
class Artifacts:
    name: str
    dataset: Path
    raw_csv: Path
    service_model: Path

    def bundle(self) -> dict:
        return load_service_model(self.service_model)

    def raw(self) -> pd.DataFrame:
        return load_vadimkin(self.raw_csv).drop_duplicates()


def _real() -> Artifacts | None:
    csvs = sorted((RAW_DIR / "vadimkin").glob("official_data_uk_*.csv"))
    if SERVICE_MODEL.exists() and (PROCESSED_DIR / "dataset.parquet").exists() and csvs:
        return Artifacts("real", PROCESSED_DIR / "dataset.parquet", csvs[-1], SERVICE_MODEL)
    return None


def pytest_generate_tests(metafunc):
    if "artifacts" in metafunc.fixturenames:
        metafunc.parametrize("artifacts", ["sample"] + (["real"] if _real() else []),
                             indirect=True, scope="session")


@pytest.fixture(scope="session")
def sample_artifacts(tmp_path_factory) -> Artifacts:
    import sample_data
    p = sample_data.build(tmp_path_factory.mktemp("sample"))
    return Artifacts("sample", p["dataset"], p["raw_csv"], p["service_model"])


@pytest.fixture(scope="session")
def artifacts(request, sample_artifacts) -> Artifacts:
    return sample_artifacts if request.param == "sample" else _real()
