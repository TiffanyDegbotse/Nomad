from core.tts_engine import TTSEngine
from core.utils import read_yaml, ROOT


def test_tts_smoke():
    settings = read_yaml(ROOT / "config" / "settings.yaml")
    tts = TTSEngine(settings)
    out = tts.synth("SightSense test synthesis.")
    assert out.exists()
