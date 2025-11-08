import pathlib
from core.context_builder import ContextBuilder
from core.utils import ROOT, read_yaml


def test_analyze_images_smoke():
    settings = read_yaml(ROOT / "config" / "settings.yaml")
    cb = ContextBuilder(settings)

    sample = list((ROOT / "data" / "images").glob("*.jpg"))
    if not sample:
        assert True  # no images = skip in CI
        return

    out = cb.analyze_images([sample[0]])
    assert isinstance(out, dict)
