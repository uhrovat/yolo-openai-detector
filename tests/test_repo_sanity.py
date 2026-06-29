"""Sanity checks for the governance scaffold.

Replaced/extended by real tests in WO-01+. Keeps CI green and verifies the
durable project structure is present.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "NOTICE.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    ".env.example",
    "docs/architecture.md",
    "docs/openai-compatibility.md",
    "docs/response-schema.md",
    "docs/requirements.md",
    "docs/non-goals.md",
    "docs/work-orders/README.md",
    "docs/work-orders/PR1-inference-core.md",
]

REQUIRED_DIRS = ["app", "scripts", "models", "tests", "tests/fixtures"]


def test_required_files_exist():
    missing = [f for f in REQUIRED_FILES if not (ROOT / f).is_file()]
    assert not missing, f"missing required files: {missing}"


def test_required_dirs_exist():
    missing = [d for d in REQUIRED_DIRS if not (ROOT / d).is_dir()]
    assert not missing, f"missing required dirs: {missing}"


def test_claude_md_mirrors_agents_md():
    # CLAUDE.md must contain the full AGENTS.md body (faithful mirror).
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    # Compare from the first H1 of AGENTS.md onward.
    anchor = "# AGENTS.md — Project Constitution"
    assert anchor in agents and anchor in claude
    assert agents[agents.index(anchor):] in claude, "CLAUDE.md is out of sync with AGENTS.md"
