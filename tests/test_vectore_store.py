from core.vector_store import VectorStore
from core.utils import read_yaml, ROOT


def test_vs_add_query():
    settings = read_yaml(ROOT / "config" / "settings.yaml")
    vs = VectorStore(settings)
    vs.add("Cape Coast Castle historic fortress", {"timestamp": "2025-01-01T00:00:00Z"})
    res = vs.query("castle")
    assert isinstance(res, list)
