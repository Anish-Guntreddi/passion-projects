from __future__ import annotations

import pytest

from minipaged.requests.request import reset_request_id_counter


@pytest.fixture(autouse=True)
def _reset_request_id_counter():
    """Every test starts with a fresh request-id counter, so ids are
    reproducible within a test regardless of test execution order."""
    reset_request_id_counter()
    yield
