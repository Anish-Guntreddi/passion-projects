"""Phase 6 release-metadata consistency checks.

These are cheap regression guards against the *documentation* artifacts
(``pyproject.toml``, ``CHANGELOG.md``, ``docs/decisions/*.md``,
``README.md``) drifting apart from each other or from what's actually on
disk -- not a substitute for the manual fresh-clone verification recorded
in ``docs/reproducibility.md`` (spec Phase 6 exit criterion), which
exercises the real install/test/train/generate path end to end.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _pyproject_version() -> str:
    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def _changelog_latest_version() -> str:
    text = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(r"^## \[(\d+\.\d+\.\d+)\]", text, flags=re.MULTILINE)
    assert match is not None, "CHANGELOG.md has no '## [X.Y.Z]' version heading"
    return match.group(1)


def test_pyproject_version_matches_changelog_latest_entry() -> None:
    assert _pyproject_version() == _changelog_latest_version()


def test_changelog_latest_entry_is_first_in_file() -> None:
    # Keep a Changelog orders entries newest-first; assert the version we
    # just matched against pyproject.toml is the *first* heading, not
    # merely present somewhere in the file (which would silently accept a
    # stale/misordered changelog).
    text = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", text, flags=re.MULTILINE)
    assert headings, "no version headings found in CHANGELOG.md"
    assert headings[0] == _pyproject_version()


def test_adr_numbering_has_no_gaps_and_starts_at_0001() -> None:
    decisions_dir = PROJECT_ROOT / "docs" / "decisions"
    numbers = sorted(
        int(m.group(1))
        for path in decisions_dir.glob("*.md")
        if (m := re.match(r"(\d{4})-", path.name)) is not None
    )
    assert numbers, "no ADR files found under docs/decisions/"
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"ADR numbering has gaps or duplicates: {numbers}"
    )


def test_committed_sample_generation_report_exists_and_is_nonempty() -> None:
    # Phase 6 (ADR 0011, decision 2): sample generations are committed
    # under benchmarks/, not the gitignored artifacts/ path.
    md_path = PROJECT_ROOT / "benchmarks" / "sample_generations" / "toy_sample_report.md"
    json_path = PROJECT_ROOT / "benchmarks" / "sample_generations" / "toy_sample_report.json"
    assert md_path.is_file() and md_path.stat().st_size > 0
    assert json_path.is_file() and json_path.stat().st_size > 0
    assert "Decoding settings:" in md_path.read_text(encoding="utf-8")


def test_readme_referenced_benchmark_plots_all_exist_on_disk() -> None:
    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    referenced = set(re.findall(r"benchmarks/plots/[\w.-]+\.png", readme_text))
    assert referenced, "expected README.md to reference at least one benchmarks/plots/*.png"
    for relative_path in referenced:
        assert (PROJECT_ROOT / relative_path).is_file(), f"missing {relative_path}"


def test_readme_status_line_claims_all_six_phases() -> None:
    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "0–6" in readme_text or "0-6" in readme_text
    assert "not yet implemented" not in readme_text
