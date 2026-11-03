"""Pytest: put a linked Ecomodel on sys.path when available (merge analyse needs BaseSurface)."""
from __future__ import annotations

import os

import pytest

from instance_annotator.ecomodel_link import EcomodelLinkError, ensure_ecomodel_on_path


def pytest_configure(config):
    try:
        ensure_ecomodel_on_path(
            os.environ.get("INSTANCE_ANNOTATOR_ECOMODEL") or None,
            persist=False,
            allow_prompt=False,
        )
    except EcomodelLinkError:
        pass


@pytest.fixture
def ecomodel_required():
    try:
        return ensure_ecomodel_on_path(allow_prompt=False, persist=False)
    except EcomodelLinkError as exc:
        pytest.skip(str(exc))
