from pathlib import Path

import pytest

from reachy_mini_language_tutor import prompts
from reachy_mini_language_tutor.config import config


@pytest.mark.parametrize(
    ("language_file", "expected"),
    [
        (None, "en"),
        ("", "en"),
        ("  \n", "en"),
        ("it\n", "it"),
        (" FR \n", "fr"),
        ("auto\n", None),
        (" AUTO \n", None),
    ],
)
def test_transcription_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, language_file: str | None, expected: str | None
) -> None:
    """Preserve language defaults and omit the hint only for automatic detection."""
    profile = tmp_path / "test_profile"
    profile.mkdir()
    if language_file is not None:
        (profile / "language.txt").write_text(language_file, encoding="utf-8")
    monkeypatch.setattr(prompts, "PROFILES_DIRECTORY", tmp_path)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", profile.name)

    settings = prompts.get_session_transcription()
    if expected is None:
        assert settings == {"model": "gpt-4o-transcribe"}
    else:
        assert settings == {"model": "gpt-4o-transcribe", "language": expected}


def test_english_for_italians_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load the actual bilingual profile and verify its session configuration."""
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "english_for_italians")
    assert prompts.get_session_language() == "auto"
    assert prompts.get_session_transcription() == {"model": "gpt-4o-transcribe"}
    assert prompts.get_session_voice() == "cedar"
    assert prompts.get_profile_proactive_mode()

    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
    assert prompts.get_session_transcription() == {"model": "gpt-4o-transcribe", "language": "en"}
    assert not prompts.get_profile_proactive_mode()
