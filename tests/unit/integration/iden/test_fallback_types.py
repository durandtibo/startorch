from __future__ import annotations

from unittest.mock import patch

import pytest

from startorch.integration.iden.fallback_types import BaseDataGenerator


def test_base_data_generator() -> None:
    class DataGenerator(BaseDataGenerator): ...

    with (
        patch("startorch.utils.imports.is_iden_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'iden' package is required but not installed."),
    ):
        DataGenerator()
