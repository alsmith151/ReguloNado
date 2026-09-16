"""Guards against accidentally publishing machine- or cluster-specific paths."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_tracked_text_files_do_not_contain_private_cluster_paths() -> None:
    repository = Path(__file__).parents[1]
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    forbidden = (
        "/" + "ceph" + "/",
        "/" + "project" + "/" + "milne_group",
        "/" + "Users" + "/" + "asmith",
    )
    violations: list[str] = []
    for raw_path in tracked:
        if not raw_path:
            continue
        relative = Path(raw_path.decode())
        path = repository / relative
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for marker in forbidden:
            if marker in text:
                violations.append(f"{relative}: contains {marker!r}")

    assert not violations, "Private paths found in tracked files:\n" + "\n".join(violations)
