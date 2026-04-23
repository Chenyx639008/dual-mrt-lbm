"""Tests for the main module."""

from lbm_mrt import __version__


def test_version():
    """Check that the version is acceptable."""
    assert isinstance(__version__, str)
