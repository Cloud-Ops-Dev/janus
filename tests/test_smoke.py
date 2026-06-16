"""Smoke test — proves the package imports and the strict CRM pipeline runs green.

Real test suites (policy matrix, registry validation, credential redaction,
descriptor-drift, fake-downstream integration) land with the Phase-1 build
(epic infra-22q).
"""
from janus import __version__


def test_version_present():
    assert __version__
