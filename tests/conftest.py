import json
from pathlib import Path

import pytest


@pytest.fixture
def light_settings():
    from yabitsu.forest_light import Settings

    path = Path(__file__).parent / "fixtures/trees_light.json"
    return Settings(**json.loads(path.read_text(encoding="utf-8")))
