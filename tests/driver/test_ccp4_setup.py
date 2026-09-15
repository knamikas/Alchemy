"""Test CCP4 installation verification and its error contract."""

from __future__ import annotations

import pytest

from driver import ccp4_setup


def test_missing_ccp4_tools_are_named_with_a_remedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The message must say which tools are absent and how to fix it."""

    def missing_edstats(tool: str, path: str | None = None) -> str | None:
        del path
        return None if tool == "edstats" else "/x"

    monkeypatch.setattr("driver.ccp4_setup.shutil.which", missing_edstats)

    with pytest.raises(ccp4_setup.Ccp4SetupError) as excinfo:
        ccp4_setup.verify_ccp4({"PATH": "/x"})

    message = str(excinfo.value)
    assert "edstats" in message
    assert "--configure-ccp4" in message, "the error should name the remedy"


def test_a_complete_ccp4_installation_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    def installed_tool(tool: str, path: str | None = None) -> str:
        del path
        return f"/opt/{tool}"

    monkeypatch.setattr("driver.ccp4_setup.shutil.which", installed_tool)
    ccp4_setup.verify_ccp4({"PATH": "/opt"})


def test_tool_availability_agrees_with_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CCP4 availability and verification use the same tool set.

    If ``ccp4_tools_available`` and ``verify_ccp4`` disagree, the driver can
    accept an installation that the setup helper rejects.
    """

    def missing_fft(tool: str, path: str | None = None) -> str | None:
        del path
        return None if tool == "fft" else "/x"

    monkeypatch.setattr("driver.ccp4_setup.shutil.which", missing_fft)

    assert not ccp4_setup.ccp4_tools_available({"PATH": "/x"})
    with pytest.raises(ccp4_setup.Ccp4SetupError):
        ccp4_setup.verify_ccp4({"PATH": "/x"})


def test_a_library_caller_never_has_to_catch_systemexit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ccp4_setup`` raises an ordinary exception, not ``SystemExit``.

    ``SystemExit`` derives from ``BaseException``, so a caller outside a CLI
    process -- a notebook, a service -- cannot contain it with ``except
    Exception``. The CLI's conversion to an exit is ``test_cli_and_config``.
    """

    def no_tools(_tool: str, path: str | None = None) -> None:
        del path
        return None

    monkeypatch.setattr("driver.ccp4_setup.shutil.which", no_tools)

    with pytest.raises(Exception) as excinfo:
        ccp4_setup.verify_ccp4({"PATH": "/x"})
    assert not isinstance(excinfo.value, SystemExit)
